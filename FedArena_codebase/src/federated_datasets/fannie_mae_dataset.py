import numpy as np
import pandas as pd
import torch
from omegaconf import open_dict, OmegaConf
from torch.utils.data import Dataset
from hydra.utils import instantiate
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, QuantileTransformer

from utils.preprocessing_cache import preprocessing_cache

from .federated_dataset import FederatedDataset

# Explicit feature allowlist — mirrors CAT_ALL/NUM_ALL in scripts/prepare_fannie_mae.py
# (prepare prints the same classification in its "source columns -> fate" report).
# No dtype auto-detection anywhere: these lists are the single word on what is a
# feature, and the closed-world assert in _fit_transform_features turns any drift
# between this file and the parquet into a loud init-time error.
CAT_FEATURES = [
    "channel", "first_time", "purpose", "prop_type", "occupancy", "state", "msa",
    "zip3", "mi_type", "special_program", "valuation_method", "high_balance",
]
NUM_FEATURES = [
    "orig_rate", "orig_upb", "orig_term", "ltv", "cltv", "dti", "fico", "fico_co",
    "n_borrowers", "n_units", "mi_pct",
]


class FannieMaeDataset(FederatedDataset):
    """Fannie Mae single-family loan performance — OOT federated benchmark.

    Binary classification: default (90-day delinquency or bad terminal Zero
    Balance code) within the first 84 months of loan life. FL client =
    ``seller`` (originating bank). Built by ``scripts/prepare_fannie_mae.py``.

    SINGLE parquet file with a **precomputed** ``split`` column: train rows are
    70% of the 2012Q1 cohort per client, val rows are sampled from 2012Q2 and
    test rows from 2012Q3 (out-of-time protocol, run_oot-equivalent). Because
    the split is a scientific decision baked offline, ``train_val_test_split``
    here simply partitions by that column and IGNORES the frac arguments —
    ``apply_client_server_phase_and_holdout`` then derives ``phase`` and carves
    the server val/test pool from it as usual (server_pool_enabled=True flow).

    Features are EXPLICIT: the module-level CAT_FEATURES/NUM_FEATURES mirror the
    prepare script's allowlist; ``*_isnull`` flags are derived from NUM_FEATURES
    by a printed rule. At init the file's columns must be exactly bookkeeping +
    allowlist (+ flags) — anything else raises, so no column can reach the model
    unseen and the label ingredients (age_d*, zb_*) physically never ship in the
    parquet at all.
    """

    def __init__(
        self,
        cfg,
        mode,
        data_sources,
        base_path,
        client_column="seller",
        split_column="split",
        num_policy="standard",
        **kwargs,
    ):
        self.client_column = client_column
        self.split_column = split_column
        self.num_policy = num_policy

        Dataset.__init__(self)
        self.cfg = cfg
        self.mode = mode
        self.distribution = instantiate(cfg.distribution)
        self.data_sources = data_sources
        self.name = self.__class__.__name__
        self.init_df()

    def load_map_files(self):
        # ONE file, already containing seller/split/target/features — the three
        # quarters were merged offline by prepare_fannie_mae.py, so nothing has
        # to be tagged or concatenated here (test_map_file is null).
        df = pd.DataFrame()
        for path in self.data_sources[f"{self.loading_mode}_map_file"]:
            print(f"[DEBUG] Loading {path}...", flush=True)
            if path.endswith(".parquet"):
                df = pd.concat([df, pd.read_parquet(path)])
            elif path.endswith(".csv"):
                try:
                    import pyarrow.csv
                    chunk = pyarrow.csv.read_csv(path).to_pandas()
                except ImportError:
                    chunk = pd.read_csv(path, low_memory=False)
                df = pd.concat([df, chunk])
            else:
                df = pd.concat([df, pd.read_csv(path, low_memory=False)])
        print(f"[DEBUG] Loaded {len(df)} rows", flush=True)
        return df

    # ------------------------------------------------------------------
    # init_df — same shape as home_credit: no feature engineering (prepare
    # did it all), fit scaler/encoder AFTER the phase column exists.
    # ------------------------------------------------------------------

    def init_df(self):
        if not self.df_exist():
            self.downloading()

        self.loading_mode = self.get_loading_mode()
        self.data = self.load_map_files()

        # test_map_file is null → no-op; kept for parity with the other datasets.
        if self.mode == "train" and OmegaConf.select(
            self.cfg, "federated_params.server_pool_enabled", default=False
        ):
            self._merge_test_data_for_pool()

        if "trust_dataset" in self.cfg:
            self.parse_trust()

        self.define_num_classes()

        # Per-client split + phase labeling BEFORE scaler/encoder fit. With the
        # server pool on, apply_client_server_phase_and_holdout calls OUR
        # train_val_test_split per client → phase mirrors the baked split.
        if self.mode == "train":
            self.split_to_clients()
            if OmegaConf.select(
                self.cfg, "federated_params.server_pool_enabled", default=False
            ):
                self.apply_client_server_phase_and_holdout()

        # Stage 2: fit on phase=='train' rows only, transform all.
        self._fit_transform_features()

        self.orig_data = self.data

    def preprocessing(self):
        # No-op: features are fully engineered offline by prepare_fannie_mae.py.
        pass

    # ------------------------------------------------------------------
    # Stage 2 — fit OrdinalEncoder + StandardScaler/QuantileTransformer on
    # phase=='train' rows (i.e. Q1-train only — Q2/Q3 must not leak into stats).
    # ------------------------------------------------------------------

    def _fit_transform_features(self):
        non_feature = {
            "target", self.client_column, self.split_column,
            # FL bookkeeping columns
            "client", "phase", "server_pool", "_tmp_id",
        }

        if self.mode == "train":
            # Explicit allowlist, intersected with the variant file (prepare
            # drops per-variant constants); _isnull flags derive from NUM_FEATURES.
            cat_cols = [c for c in CAT_FEATURES if c in self.data.columns]
            num_cols = [c for c in NUM_FEATURES if c in self.data.columns]
            isnull_cols = [
                f"{c}_isnull" for c in NUM_FEATURES
                if f"{c}_isnull" in self.data.columns
            ]
            num_cols = num_cols + isnull_cols

            # Closed world: the file may contain NOTHING beyond allowlist +
            # bookkeeping. A new/renamed column must be classified by hand
            # (here and in prepare_fannie_mae.py), never picked up silently.
            extra = (
                set(self.data.columns) - set(cat_cols) - set(num_cols) - non_feature
            )
            assert not extra, (
                f"Unexpected columns in the parquet: {sorted(extra)}. Classify "
                f"them in CAT_FEATURES/NUM_FEATURES (and prepare_fannie_mae.py) "
                f"or drop them from the file."
            )
            absent = [
                c for c in CAT_FEATURES + NUM_FEATURES
                if c not in self.data.columns
            ]
            print(f"[features] cat ({len(cat_cols)}): {cat_cols}", flush=True)
            print(f"[features] num ({len(num_cols)}, incl. {len(isnull_cols)} "
                  f"_isnull flags): {num_cols}", flush=True)
            if absent:
                print(f"[features] allowlisted but absent in this variant "
                      f"(constant, dropped by prepare): {absent}", flush=True)

            # Fit ONLY on pure training rows (phase from the baked split). In the
            # OOT setup this is not just hygiene: fitting on val/test would let
            # the scaler peek at the FUTURE quarters' distributions.
            if "phase" in self.data.columns:
                train_mask = (self.data["phase"] == "train").values
            elif self.split_column in self.data.columns:
                # server_pool disabled → no phase column; the baked split still
                # tells us which rows are train.
                train_mask = (self.data[self.split_column] == "train").values
            else:
                train_mask = np.ones(len(self.data), dtype=bool)

            n_train_rows = int(train_mask.sum())
            print(
                f"[DEBUG] fit scaler/encoder on {n_train_rows} train rows "
                f"({len(self.data) - n_train_rows} val+test+pool rows excluded)",
                flush=True,
            )

            # --- OrdinalEncoder: fit on TRAIN rows only ---
            unknown_value = np.iinfo("int64").max - 3
            encoded_missing_value = -1
            if cat_cols:
                print("[DEBUG] OrdinalEncoder fitting on train rows...", flush=True)
                encoder = OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=unknown_value,
                    encoded_missing_value=encoded_missing_value,
                    dtype="int64",
                ).fit(self.data.loc[train_mask, cat_cols])
                print("[DEBUG] OrdinalEncoder transform on all rows...", flush=True)
                encoded = encoder.transform(self.data[cat_cols]).astype(np.int64)
                # Remap -1 (missing) and unknown_value to valid indices [0..card+1]
                max_known = np.array([len(cats) - 1 for cats in encoder.categories_])
                for col_idx in range(encoded.shape[1]):
                    mask_unknown = encoded[:, col_idx] == unknown_value
                    mask_missing = encoded[:, col_idx] == encoded_missing_value
                    encoded[mask_unknown, col_idx] = max_known[col_idx] + 1
                    encoded[mask_missing, col_idx] = max_known[col_idx] + 2
                self.data[cat_cols] = encoded
                # +1 for unknown values at test time, +1 for encoded_missing_value
                cat_cardinalities = [len(cats) + 2 for cats in encoder.categories_]
                ordinal_categories = [list(cats) for cats in encoder.categories_]
            else:
                cat_cardinalities = []
                ordinal_categories = []

            # --- Numerical: fillna, then fit on TRAIN rows only.
            #     num_policy: "standard" (StandardScaler) | "noisy_quantile"
            #     (QuantileTransformer→normal; tiny 1e-5 noise on train breaks
            #     ties so features with few unique values transform smoothly). ---
            self.data[num_cols] = (
                self.data[num_cols].fillna(0).astype(np.float32)
            )
            seed = self.cfg.random_state
            if getattr(self, "num_policy", "standard") == "noisy_quantile":
                n_tr = max(int(train_mask.sum()), 1)
                print("[DEBUG] noisy-quantile fit on train rows...", flush=True)
                num_transformer = QuantileTransformer(
                    n_quantiles=max(min(n_tr // 30, 1000), 10),
                    output_distribution="normal",
                    subsample=10**9,
                    random_state=seed,
                )
                X_tr = self.data.loc[train_mask, num_cols].to_numpy(np.float32)
                X_tr = X_tr + np.random.RandomState(seed).normal(
                    0.0, 1e-5, X_tr.shape
                ).astype(np.float32)
                num_transformer.fit(X_tr)
            else:
                print("[DEBUG] StandardScaler fit on train rows...", flush=True)
                num_transformer = StandardScaler().fit(
                    self.data.loc[train_mask, num_cols]
                )
            self.data[num_cols] = num_transformer.transform(
                self.data[num_cols]
            ).astype(np.float32)

            print("[DEBUG] Writing cache...", flush=True)
            preprocessing_cache.clear()
            cache_update = {
                "num_cols": num_cols,
                "cat_cols": cat_cols,
                "cat_cardinalities": cat_cardinalities,
                "ordinal_categories": ordinal_categories,
                "num_transformer": num_transformer,
            }
            if isinstance(num_transformer, StandardScaler):
                cache_update["scaler_mean"] = num_transformer.mean_.tolist()
                cache_update["scaler_scale"] = num_transformer.scale_.tolist()
            preprocessing_cache.update(cache_update)

            with open_dict(self.cfg):
                self.cfg.model.n_num_features = len(num_cols)
                self.cfg.model.cat_cardinalities = cat_cardinalities

        else:
            # Test mode: use cached parameters from the train run. (With the
            # server pool on, a separate test dataset is never instantiated —
            # kept for completeness, same as home_credit.)
            cache = preprocessing_cache
            assert cache, "Train dataset must be created before test dataset"
            num_cols = list(cache["num_cols"])
            cat_cols = list(cache["cat_cols"])

            if cat_cols:
                unknown_value = np.iinfo("int64").max - 3
                encoded_missing_value = -1
                categories = [list(cats) for cats in cache["ordinal_categories"]]
                encoder = OrdinalEncoder(
                    categories=categories,
                    handle_unknown="use_encoded_value",
                    unknown_value=unknown_value,
                    encoded_missing_value=encoded_missing_value,
                    dtype="int64",
                )
                dummy = np.array([[cats[0] for cats in categories]])
                encoder.fit(dummy)
                encoded = encoder.transform(self.data[cat_cols])
                max_known = np.array([len(cats) - 1 for cats in categories])
                for col_idx in range(encoded.shape[1]):
                    mask_unknown = encoded[:, col_idx] == unknown_value
                    mask_missing = encoded[:, col_idx] == encoded_missing_value
                    encoded[mask_unknown, col_idx] = max_known[col_idx] + 1
                    encoded[mask_missing, col_idx] = max_known[col_idx] + 2
                self.data[cat_cols] = encoded.astype(np.int64)

            self.data[num_cols] = (
                self.data[num_cols].fillna(0).astype(np.float32)
            )
            num_transformer = cache.get("num_transformer")
            if num_transformer is not None:
                self.data[num_cols] = num_transformer.transform(
                    self.data[num_cols]
                ).astype(np.float32)
            else:
                mean = np.array(cache["scaler_mean"])
                scale = np.array(cache["scaler_scale"])
                scale[scale == 0] = 1.0
                self.data[num_cols] = (
                    (self.data[num_cols].values - mean) / scale
                ).astype(np.float32)

        self.num_cols = num_cols
        self.cat_cols = cat_cols

        self.data[num_cols] = np.nan_to_num(
            self.data[num_cols].values
        ).astype(np.float32)
        self.data["target"] = self.data["target"].astype(int)

    def split_to_clients(self):
        print("[DEBUG] split_to_clients start...", flush=True)
        print(f"Used distribution is: {self.distribution.__class__.__name__}")
        self.data = self.distribution.split_to_clients(
            self.data,
            self.cfg.federated_params.amount_of_clients,
            self.cfg.random_state,
        )
        actual_clients = self.data["client"].nunique()
        with open_dict(self.cfg):
            self.cfg.federated_params.amount_of_clients = actual_clients
        print(f"Amount of clients set to: {actual_clients}")

    # NOTE: train_val_split is NOT overridden — the base stratified version is
    # only used by the trust-dataset branch, which this dataset doesn't use.

    @staticmethod
    def train_val_test_split(
        df, random_state, train_frac=0.7, val_frac=0.15, test_frac=0.15
    ):
        """Partition by the precomputed ``split`` column; fracs are IGNORED.

        The OOT split (train=Q1 70%, val=Q2 sample, test=Q3 sample) was baked
        into the parquet by scripts/prepare_fannie_mae.py. Called per client from
        apply_client_server_phase_and_holdout (federated_dataset.py:285), which
        reads ``_tmp_id`` from the returned val/test frames to write phase and
        carve the server pool — so all columns must be preserved.
        """
        train_df = df[df["split"] == "train"].reset_index(drop=True)
        val_df = df[df["split"] == "val"].reset_index(drop=True)
        test_df = df[df["split"] == "test"].reset_index(drop=True)
        return train_df, val_df, test_df

    # ------------------------------------------------------------------
    # Fast __getitem__ via materialized numpy buffers (home_credit pattern)
    # ------------------------------------------------------------------

    def _ensure_feature_buffers(self):
        """Rebuild cached numpy rows when self.data is replaced (client split, train/val).

        Hot path __getitem__ then only indexes dense arrays — no per-sample pandas iloc.
        """
        key = (id(self.data), len(self.data))
        if getattr(self, "_feature_buf_key", None) == key:
            return
        self._feature_buf_key = key
        n = len(self.data)
        n_num = len(self.num_cols)
        n_cat = len(self.cat_cols)
        if n_num:
            self._buf_x_num = np.ascontiguousarray(
                self.data[self.num_cols].to_numpy(dtype=np.float32, copy=False)
            )
        else:
            self._buf_x_num = np.empty((n, 0), dtype=np.float32)
        if n_cat:
            self._buf_x_cat = np.ascontiguousarray(
                self.data[self.cat_cols].to_numpy(dtype=np.int64, copy=False)
            )
        else:
            self._buf_x_cat = None
        self._buf_y = np.ascontiguousarray(
            self.data["target"].to_numpy(copy=False)
        )

    def __getitem__(self, index):
        self._ensure_feature_buffers()
        x_num = torch.as_tensor(self._buf_x_num[index])
        if self._buf_x_cat is not None:
            x_cat = torch.as_tensor(self._buf_x_cat[index])
        else:
            x_cat = torch.zeros(0, dtype=torch.long)
        label = int(self._buf_y[index])
        return index, ([x_num, x_cat], label)
