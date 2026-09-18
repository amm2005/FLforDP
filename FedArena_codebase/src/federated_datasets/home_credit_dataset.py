import numpy as np
import pandas as pd
import torch
from omegaconf import open_dict, OmegaConf
from torch.utils.data import Dataset
from hydra.utils import instantiate
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, QuantileTransformer

from utils.preprocessing_cache import preprocessing_cache

from .federated_dataset import FederatedDataset


class HomeCreditDataset(FederatedDataset):
    """Home Credit federated dataset with phase-aware feature fitting."""

    def __init__(
        self,
        cfg,
        mode,
        data_sources,
        base_path,
        client_column="mode_credor_3940957M",
        time_column="week_num",
        num_policy="standard",
        **kwargs,
    ):
        self.client_column = client_column
        self.time_column = time_column
        self.num_policy = num_policy

        Dataset.__init__(self)
        self.cfg = cfg
        self.mode = mode
        self.distribution = instantiate(cfg.distribution)
        self.data_sources = data_sources
        self.name = self.__class__.__name__
        self.init_df()

    def load_map_files(self):
        df = pd.DataFrame()
        for path in self.data_sources[f"{self.loading_mode}_map_file"]:
            print(f"[DEBUG] Loading {path}...", flush=True)
            if path.endswith(".parquet"):
                df = pd.concat([df, pd.read_parquet(path)])
            elif path.endswith(".csv"):
                try:
                    import pyarrow.csv
                    tbl = pyarrow.csv.read_csv(path)
                    chunk = tbl.to_pandas()
                except ImportError:
                    chunk = pd.read_csv(path, low_memory=False)
                df = pd.concat([df, chunk])
            else:
                df = pd.concat([df, pd.read_csv(path, low_memory=False)])
        print(f"[DEBUG] Loaded {len(df)} rows", flush=True)
        return df

    def init_df(self):
        if not self.df_exist():
            self.downloading()

        self.loading_mode = self.get_loading_mode()
        self.data = self.load_map_files()

        if self.mode == "train" and OmegaConf.select(
            self.cfg, "federated_params.server_pool_enabled", default=False
        ):
            self._merge_test_data_for_pool()

        if "trust_dataset" in self.cfg:
            self.parse_trust()

        self.define_num_classes()

        if self.mode == "train":
            self.split_to_clients()
            if OmegaConf.select(
                self.cfg, "federated_params.server_pool_enabled", default=False
            ):
                self.apply_client_server_phase_and_holdout()

        self._fit_transform_features()

        self.orig_data = self.data

    def preprocessing(self):
        pass

    def _fit_transform_features(self):
        non_feature = {
            "target", "case_id", self.time_column, self.client_column,
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

            # Fit only on phase=='train' rows to avoid val/test leakage.
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
                # Remap missing/unknown encodings to valid trailing indices.
                max_known = np.array([len(cats) - 1 for cats in encoder.categories_])
                for col_idx in range(encoded.shape[1]):
                    mask_unknown = encoded[:, col_idx] == unknown_value
                    mask_missing = encoded[:, col_idx] == encoded_missing_value
                    encoded[mask_unknown, col_idx] = max_known[col_idx] + 1
                    encoded[mask_missing, col_idx] = max_known[col_idx] + 2
                self.data[cat_cols] = encoded
                # +1 for unknown at test, +1 for encoded_missing_value.
                cat_cardinalities = [
                    len(cats) + 2 for cats in encoder.categories_
                ]
                ordinal_categories = [
                    list(cats) for cats in encoder.categories_
                ]
            else:
                cat_cardinalities = []
                ordinal_categories = []

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
            cache = preprocessing_cache
            assert cache, "Train dataset must be created before test dataset"
            num_cols = list(cache["num_cols"])
            cat_cols = list(cache["cat_cols"])

            if cat_cols:
                unknown_value = np.iinfo("int64").max - 3
                encoded_missing_value = -1
                categories = [
                    list(cats) for cats in cache["ordinal_categories"]
                ]
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

    @staticmethod
    def train_val_split(df, train_val_prop, random_state):
        """Temporal split by week_num; stratified fallback if column absent."""
        time_col = "week_num"
        if time_col in df.columns:
            sorted_df = df.sort_values(time_col)
            n_val = max(1, int(len(sorted_df) * train_val_prop))
            n_train = len(sorted_df) - n_val
            train_df = sorted_df.iloc[:n_train].reset_index(drop=True)
            val_df = sorted_df.iloc[n_train:].reset_index(drop=True)
            return train_df, val_df
        return FederatedDataset.train_val_split(df, train_val_prop, random_state)

    @staticmethod
    def train_val_test_split(
        df, random_state, train_frac=0.7, val_frac=0.15, test_frac=0.15
    ):
        """Temporal 3-way split by week_num."""
        time_col = "week_num"
        sorted_df = (
            df.sort_values(time_col).reset_index(drop=True)
            if time_col in df.columns
            else df.reset_index(drop=True)
        )
        n = len(sorted_df)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        n_train = n - n_val - n_test
        train_df = sorted_df.iloc[:n_train]
        val_df = sorted_df.iloc[n_train : n_train + n_val]
        test_df = sorted_df.iloc[n_train + n_val :]
        return (
            train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True),
        )

    def _ensure_feature_buffers(self):
        """Cache dense numpy buffers of features/targets for fast __getitem__."""
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
