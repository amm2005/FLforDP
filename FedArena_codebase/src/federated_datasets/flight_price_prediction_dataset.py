import numpy as np
import pandas as pd
import torch
from omegaconf import open_dict, OmegaConf
from torch.utils.data import Dataset
from hydra.utils import instantiate
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

from utils.preprocessing_cache import preprocessing_cache

from .federated_dataset import FederatedDataset


class FlightPricePredictionDataset(FederatedDataset):
    """Flight price (EaseMyTrip) regression dataset for federated learning.

    SINGLE-FILE dataset (``Clean_Dataset.csv``): there is no predefined
    train/test file and no usable calendar/time axis (``days_left`` is just
    days-to-departure in the 50-day collection). So this is the *single-file*
    counterpart to covid19 (two predefined files + temporal split).

    Flow (relies on ``server_pool_enabled: True``):
      1. Load the one file as the whole dataset (``test_map_file: null`` →
         ``_merge_test_data_for_pool`` is a no-op).
      2. Feature-engineer once on the full frame (incl. route aggregates) — no
         per-instance inconsistency because a separate ``mode='test'`` dataset is
         never instantiated when the server pool is on.
      3. ``split_to_clients`` by ``airline`` → per-client **i.i.d.** (random)
         train/val/test split via ``train_val_test_split``.
      4. Server pool carves global val/test out of each client's local val/test.

    The ONLY differences vs covid19 are: ``test_map_file`` null vs a real file,
    and ``train_val_test_split`` i.i.d. vs temporal. Everything else (server
    pool, phase-aware encoder/scaler fit, target standardization) is shared.

    Target: ``price`` → ``target_transform`` (log1p by default) → standardized on
    ``phase=='train'`` rows. RMSE reported in log-space; R2 is scale-invariant.
    """

    def __init__(
        self,
        cfg,
        mode,
        data_sources,
        base_path,
        client_column="airline",
        time_column=None,          # no time axis → i.i.d. split
        target_column="price",
        target_transform="log1p",
        **kwargs,
    ):
        self.client_column = client_column
        self.time_column = time_column
        self.target_column = target_column
        self.target_transform = target_transform

        Dataset.__init__(self)
        self.cfg = cfg
        self.mode = mode
        self.distribution = instantiate(cfg.distribution)
        self.data_sources = data_sources
        self.name = self.__class__.__name__
        self.init_df()

    # ------------------------------------------------------------------
    # IO (parquet + fast pyarrow csv)
    # ------------------------------------------------------------------

    def load_map_files(self):
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
    # init_df — fit scaler/encoder AFTER phase column is built
    # ------------------------------------------------------------------

    def init_df(self):
        if not self.df_exist():
            self.downloading()

        self.loading_mode = self.get_loading_mode()
        self.data = self.load_map_files()

        # Single-file dataset: test_map_file is null, so this is a no-op
        # (kept for parity with covid/home_credit — harmless if no test file).
        if self.mode == "train" and OmegaConf.select(
            self.cfg, "federated_params.server_pool_enabled", default=False
        ):
            self._merge_test_data_for_pool()

        # Stage 1: feature engineering — no fitting; all rows.
        self._feature_engineering()

        if "trust_dataset" in self.cfg:
            self.parse_trust()

        self.define_num_classes()

        # Per-client split + phase labeling BEFORE scaler/encoder fit.
        if self.mode == "train":
            self.split_to_clients()
            if OmegaConf.select(
                self.cfg, "federated_params.server_pool_enabled", default=False
            ):
                self.apply_client_server_phase_and_holdout()

        # Stage 2: fit on phase=='train' rows only, transform all.
        self._fit_transform_features()

        self.orig_data = self.data

    # ------------------------------------------------------------------
    # Stage 1 — feature engineering (no fitting). All rows.
    # ------------------------------------------------------------------

    def _feature_engineering(self):
        # Drop the unnamed CSV index column if present (Clean_Dataset has one).
        drop_cols = [
            c for c in self.data.columns
            if c == "" or str(c).startswith("Unnamed")
        ]
        if drop_cols:
            self.data = self.data.drop(columns=drop_cols)

        stops_map = {"zero": 0, "one": 1, "two_or_more": 2}
        time_order = {
            "Early_Morning": 0, "Morning": 1, "Afternoon": 2,
            "Evening": 3, "Night": 4, "Late_Night": 5,
        }

        d = self.data

        d["dep_time_order"] = d["departure_time"].map(time_order)
        d["arr_time_order"] = d["arrival_time"].map(time_order)

        d["route"] = (
            d["source_city"].astype(str) + "_" + d["destination_city"].astype(str)
        )
        d["stops_num"] = d["stops"].map(stops_map)               # number of stops
        d["is_direct"] = (d["stops_num"] == 0).astype(int)       # direct flight
        d["is_multi_stop"] = (d["stops_num"] >= 2).astype(int)   # >=2 stops
        d["early_booking_30"] = (d["days_left"] >= 30).astype(int)  # early booking
        d["duration_per_stop"] = d["duration"] / (d["stops_num"] + 1)  # avg leg
        d["is_overnight"] = (
            d["arr_time_order"] < d["dep_time_order"]
        ).astype(int)                                            # overnight flight

        # Cyclical time-of-day encoding (6 bins → period 6). Late_Night sits next
        # to Early_Morning on the circle, which a raw 0..5 ordinal gets wrong.
        # These replace departure_time/arrival_time (categorical) and the raw
        # *_time_order ordinals — all of which are excluded in non_feature.
        period = 6
        d["dep_time_sin"] = np.sin(2 * np.pi * d["dep_time_order"] / period)
        d["dep_time_cos"] = np.cos(2 * np.pi * d["dep_time_order"] / period)
        d["arr_time_sin"] = np.sin(2 * np.pi * d["arr_time_order"] / period)
        d["arr_time_cos"] = np.cos(2 * np.pi * d["arr_time_order"] / period)

        # Route-level aggregates. NOTE: computed on the FULL frame across all
        # clients (airlines)
        d["n_airlines_on_route"] = (
            d.groupby("route")["airline"].transform("nunique")
        )
        d["n_classes_on_route"] = (
            d.groupby("route")["class"].transform("nunique")
        )
        d["duration_rank_on_route"] = (
            d.groupby("route")["duration"].rank(pct=True)
        )
        # d["n_flights_on_route"] = d.groupby("route")["flight"].transform("nunique")

        # --- Target: target_column -> target_transform (log1p by default) ---
        # The raw target column is NOT dropped — it is excluded from features
        # via the non_feature set in _fit_transform_features.
        if "target" not in d.columns:
            if self.target_column not in d.columns:
                raise ValueError(
                    f"target_column='{self.target_column}' not found in data "
                    f"columns: {list(d.columns)}"
                )
            d["target"] = self._transform_target(d[self.target_column])

        self.data = d

    def _transform_target(self, series):
        """Map the raw target column to the model target via string dispatch.

        ``self.target_transform`` is a plain string (or None) — no subclasses,
        just a lookup table. Extend ``transforms`` to add new options.

          - None / "none" : identity (float32 cast)
          - "log1p"       : log1p with NaN→0 and negative-clip safety
        """
        transforms = {
            "none": lambda s: s.astype(np.float32),
            "log1p": lambda s: np.log1p(
                s.fillna(0).clip(lower=0)
            ).astype(np.float32),
        }
        name = (
            str(self.target_transform).lower()
            if self.target_transform is not None
            else "none"
        )
        if name not in transforms:
            raise ValueError(
                f"Unknown target_transform={self.target_transform!r}. "
                f"Supported: {sorted(transforms)}."
            )
        return transforms[name](series)

    # ------------------------------------------------------------------
    # Stage 2 — fit OrdinalEncoder + StandardScaler on phase=='train' rows
    # ------------------------------------------------------------------

    def _fit_transform_features(self):
        non_feature = {
            "target",
            self.target_column,            # raw target ("price") — never a feature
            self.time_column,              # None for flight — harmless
            self.client_column,            # "airline" (client id source)
            "flight",                      # high-cardinality ID — not a feature
            "stops",                       # redundant with stops_num
            # time-of-day → fed in as sin/cos cyclical features instead
            "departure_time", "arrival_time",
            "dep_time_order", "arr_time_order",  # intermediates for is_overnight + sin/cos
            # FL bookkeeping columns
            "client", "phase", "server_pool", "_tmp_id",
        }

        if self.mode == "train":
            all_cols = [c for c in self.data.columns if c not in non_feature]
            cat_cols = [
                c for c in all_cols
                if self.data[c].dtype == object
                or str(self.data[c].dtype) == "category"
            ]
            num_cols = [c for c in all_cols if c not in cat_cols]

            # Fit ONLY on pure training rows. Val and test (incl. server pool)
            # are excluded so neither distribution leaks into scaler stats or
            # encoder categories.
            if "phase" in self.data.columns:
                train_mask = (self.data["phase"] == "train").values
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
                max_known = np.array([len(cats) - 1 for cats in encoder.categories_])
                for col_idx in range(encoded.shape[1]):
                    mask_unknown = encoded[:, col_idx] == unknown_value
                    mask_missing = encoded[:, col_idx] == encoded_missing_value
                    encoded[mask_unknown, col_idx] = max_known[col_idx] + 1
                    encoded[mask_missing, col_idx] = max_known[col_idx] + 2
                self.data[cat_cols] = encoded
                cat_cardinalities = [len(cats) + 2 for cats in encoder.categories_]
                ordinal_categories = [list(cats) for cats in encoder.categories_]
            else:
                cat_cardinalities = []
                ordinal_categories = []

            # --- StandardScaler: fit on TRAIN rows only ---
            print("[DEBUG] Numerical fillna + StandardScaler fit on train rows...", flush=True)
            self.data[num_cols] = (
                self.data[num_cols].fillna(0).astype(np.float32)
            )
            scaler = StandardScaler()
            scaler.fit(self.data.loc[train_mask, num_cols])
            self.data[num_cols] = scaler.transform(
                self.data[num_cols]
            ).astype(np.float32)

            # --- Standardize regression target on TRAIN rows only ---
            target_mean = float(self.data.loc[train_mask, "target"].mean())
            target_std = float(self.data.loc[train_mask, "target"].std())
            if target_std == 0.0 or not np.isfinite(target_std):
                target_std = 1.0
            self.data["target"] = (
                (self.data["target"].values - target_mean) / target_std
            ).astype(np.float32)
            print(
                f"[DEBUG] Target standardized: mean={target_mean:.4f}, "
                f"std={target_std:.4f}",
                flush=True,
            )

            preprocessing_cache.clear()
            preprocessing_cache.update({
                "num_cols": num_cols,
                "cat_cols": cat_cols,
                "cat_cardinalities": cat_cardinalities,
                "ordinal_categories": ordinal_categories,
                "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist(),
                "num_classes": 1,  # regression
                "target_mean": target_mean,
                "target_std": target_std,
            })

            with open_dict(self.cfg):
                self.cfg.model.n_num_features = len(num_cols)
                self.cfg.model.cat_cardinalities = cat_cardinalities

        else:
            # Test mode: use cached parameters from train run.
            cache = preprocessing_cache
            assert cache, "Train dataset must be created before test dataset"
            num_cols = list(cache["num_cols"])
            cat_cols = list(cache["cat_cols"])

            unknown_value = np.iinfo("int64").max - 3
            encoded_missing_value = -1
            if cat_cols:
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
            mean = np.array(cache["scaler_mean"])
            scale = np.array(cache["scaler_scale"])
            scale[scale == 0] = 1.0
            self.data[num_cols] = (
                (self.data[num_cols].values - mean) / scale
            ).astype(np.float32)

            target_mean = float(cache.get("target_mean", 0.0))
            target_std = float(cache.get("target_std", 1.0))
            if target_std == 0.0 or not np.isfinite(target_std):
                target_std = 1.0
            self.data["target"] = (
                (self.data["target"].values - target_mean) / target_std
            ).astype(np.float32)

        self.num_cols = num_cols
        self.cat_cols = cat_cols

        # Final cleanup: defang any remaining NaN (e.g. from constant columns).
        self.data[num_cols] = np.nan_to_num(
            self.data[num_cols].values
        ).astype(np.float32)
        self.data["target"] = self.data["target"].astype(np.float32)

    # ------------------------------------------------------------------
    # Misc — define_num_classes, split_to_clients, i.i.d. split
    # ------------------------------------------------------------------

    def define_num_classes(self):
        """Regression: num_classes=1."""
        self.num_classes = 1
        with open_dict(self.cfg):
            self.cfg.training_params.num_classes = self.num_classes
        if self.cfg.model.num_classes != self.num_classes:
            with open_dict(self.cfg):
                self.cfg.model.num_classes = self.num_classes

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

    @staticmethod
    def train_val_test_split(
        df, random_state, train_frac=0.7, val_frac=0.15, test_frac=0.15
    ):
        """i.i.d. (random) 3-way split — flight prices have no usable time axis.

        Deterministic given ``random_state`` (called per client inside
        ``apply_client_server_phase_and_holdout``).
        """
        n = len(df)
        if n < 3:
            empty = pd.DataFrame(columns=df.columns)
            return df.reset_index(drop=True), empty, empty
        shuffled = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        n_train = n - n_val - n_test
        return (
            shuffled.iloc[:n_train].reset_index(drop=True),
            shuffled.iloc[n_train:n_train + n_val].reset_index(drop=True),
            shuffled.iloc[n_train + n_val:].reset_index(drop=True),
        )

    # ------------------------------------------------------------------
    # Fast __getitem__ via materialized numpy buffers
    # ------------------------------------------------------------------

    def _ensure_feature_buffers(self):
        """Materialize numpy buffers from self.data once per unique df instance."""
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
            self.data["target"].to_numpy(dtype=np.float32, copy=False)
        )

    def __getitem__(self, index):
        self._ensure_feature_buffers()
        x_num = torch.as_tensor(self._buf_x_num[index])
        if self._buf_x_cat is not None:
            x_cat = torch.as_tensor(self._buf_x_cat[index])
        else:
            x_cat = torch.zeros(0, dtype=torch.long)
        label = float(self._buf_y[index])
        return index, ([x_num, x_cat], label)
