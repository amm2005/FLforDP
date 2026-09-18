import numpy as np
import pandas as pd
import torch
from omegaconf import open_dict, OmegaConf
from torch.utils.data import Dataset
from hydra.utils import instantiate
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

from utils.preprocessing_cache import preprocessing_cache

from .federated_dataset import FederatedDataset


class Covid19Dataset(FederatedDataset):
    """COVID-19 daily new cases regression dataset for federated learning."""

    def __init__(
        self,
        cfg,
        mode,
        data_sources,
        base_path,
        client_column="country",
        time_column="date",
        target_column="daily_new_cases",
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

        self._feature_engineering()

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

    def _feature_engineering(self):
        """Calendar, log1p, lags, rolling stats, and days_since_first_case features."""
        self.data[self.time_column] = pd.to_datetime(
            self.data[self.time_column], format="%Y-%m-%d"
        )
        self.data = self.data.sort_values(
            [self.client_column, self.time_column]
        ).reset_index(drop=True)

        day_of_week = self.data[self.time_column].dt.dayofweek
        month = self.data[self.time_column].dt.month
        self.data["week"] = self.data[self.time_column].dt.isocalendar().week.astype(int)
        self.data["year"] = self.data[self.time_column].dt.year
        self.data["day_of_week_sin"] = np.sin(2 * np.pi * day_of_week / 7.0)
        self.data["day_of_week_cos"] = np.cos(2 * np.pi * day_of_week / 7.0)
        self.data["month_sin"] = np.sin(2 * np.pi * month / 12.0)
        self.data["month_cos"] = np.cos(2 * np.pi * month / 12.0)
        self.data["is_weekend"] = (day_of_week >= 5).astype(int)

        if "target" not in self.data.columns:
            if self.target_column not in self.data.columns:
                raise ValueError(
                    f"target_column='{self.target_column}' not found in data "
                    f"columns: {list(self.data.columns)}"
                )
            self.data["target"] = self._transform_target(
                self.data[self.target_column]
            )

        def _logged_series(col):
            return np.log1p(self.data[col].fillna(0).clip(lower=0))

        cid = self.data[self.client_column]

        if "daily_new_cases" in self.data.columns:
            src = _logged_series("daily_new_cases")
            grp = src.groupby(cid)
            for lag in [1, 7, 14]:
                self.data[f"log_daily_new_cases_lag{lag}"] = grp.shift(lag)
            shifted = grp.shift(1).groupby(cid)  # shift(1) prevents leakage
            for window in [3, 7, 14]:
                self.data[f"log_daily_new_cases_rmean_{window}"] = (
                    shifted.rolling(window, min_periods=1).mean()
                    .reset_index(level=0, drop=True)
                )
            self.data["log_daily_new_cases_rstd_7"] = (
                shifted.rolling(7, min_periods=1).std()
                .reset_index(level=0, drop=True)
            )

        if "daily_new_deaths" in self.data.columns:
            src = _logged_series("daily_new_deaths")
            grp = src.groupby(cid)
            for lag in [1, 7]:
                self.data[f"log_daily_new_deaths_lag{lag}"] = grp.shift(lag)
            shifted = grp.shift(1).groupby(cid)
            self.data["log_daily_new_deaths_rmean_7"] = (
                shifted.rolling(7, min_periods=1).mean()
                .reset_index(level=0, drop=True)
            )

        if "cumulative_total_cases" in self.data.columns:
            src = _logged_series("cumulative_total_cases")
            grp = src.groupby(cid)
            self.data["log_cumulative_total_cases_lag1"] = grp.shift(1)
            self.data["log_cumulative_total_cases_lag14"] = grp.shift(14)

        if "cumulative_total_deaths" in self.data.columns:
            src = _logged_series("cumulative_total_deaths")
            grp = src.groupby(cid)
            self.data["log_cumulative_total_deaths_lag1"] = grp.shift(1)
            self.data["log_cumulative_total_deaths_lag14"] = grp.shift(14)

        if "active_cases" in self.data.columns:
            src = _logged_series("active_cases")
            grp = src.groupby(cid)
            self.data["log_active_cases_lag1"] = grp.shift(1)
            self.data["log_active_cases_lag7"] = grp.shift(7)
            shifted = grp.shift(1).groupby(cid)
            self.data["log_active_cases_rmean_7"] = (
                shifted.rolling(7, min_periods=1).mean()
                .reset_index(level=0, drop=True)
            )

        if self.mode == "train":
            train_country_anchors = (
                self.data.groupby(self.client_column, sort=False)[self.time_column]
                .min()
            )
            self.data["days_since_first_case"] = (
                self.data[self.time_column]
                - self.data[self.client_column].map(train_country_anchors)
            ).dt.days
            self._train_country_anchors = train_country_anchors
        else:
            cache = preprocessing_cache
            anchors = (cache.get("country_anchor_dates") if cache else None) or {}
            anchor_ts = pd.to_datetime(
                self.data[self.client_column].astype(str).map(anchors),
                errors="coerce",
            )
            fallback = self.data.groupby(self.client_column)[self.time_column].transform("min")
            anchor_ts = anchor_ts.fillna(fallback)
            self.data["days_since_first_case"] = (
                self.data[self.time_column] - anchor_ts
            ).dt.days
            self._train_country_anchors = None

    def _transform_target(self, series):
        """Map raw target column to model target via string dispatch."""
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

    def _fit_transform_features(self):
        non_feature = {
            "target",
            self.target_column,
            self.time_column,
            self.client_column,
            "client", "phase", "server_pool", "_tmp_id",
            "daily_new_cases", "daily_new_deaths",
            "cumulative_total_cases", "cumulative_total_deaths", "active_cases",
            "log_daily_new_cases", "log_daily_new_deaths",
            "log_cumulative_total_cases", "log_cumulative_total_deaths",
            "log_active_cases",
            "active_ratio",
        }

        if self.mode == "train":
            all_cols = [c for c in self.data.columns if c not in non_feature]
            cat_cols = [
                c for c in all_cols
                if self.data[c].dtype == object
                or str(self.data[c].dtype) == "category"
            ]
            num_cols = [c for c in all_cols if c not in cat_cols]

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

            print("[DEBUG] Numerical fillna + StandardScaler fit on train rows...", flush=True)
            self.data[num_cols] = (
                self.data[num_cols].fillna(0).astype(np.float32)
            )
            scaler = StandardScaler()
            scaler.fit(self.data.loc[train_mask, num_cols])
            self.data[num_cols] = scaler.transform(
                self.data[num_cols]
            ).astype(np.float32)

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

            anchors_iso = {}
            if getattr(self, "_train_country_anchors", None) is not None:
                anchors_iso = {
                    str(k): (v.isoformat() if hasattr(v, "isoformat") else str(v))
                    for k, v in self._train_country_anchors.items()
                }
            preprocessing_cache.clear()
            preprocessing_cache.update({
                "num_cols": num_cols,
                "cat_cols": cat_cols,
                "cat_cardinalities": cat_cardinalities,
                "ordinal_categories": ordinal_categories,
                "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist(),
                "num_classes": 1,
                "country_anchor_dates": anchors_iso,
                "target_mean": target_mean,
                "target_std": target_std,
            })

            with open_dict(self.cfg):
                self.cfg.model.n_num_features = len(num_cols)
                self.cfg.model.cat_cardinalities = cat_cardinalities

        else:
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

        self.data[num_cols] = np.nan_to_num(
            self.data[num_cols].values
        ).astype(np.float32)
        self.data["target"] = self.data["target"].astype(np.float32)

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
    def train_val_split(df, train_val_prop, random_state):
        """Temporal split by date; falls back to stratified if column absent."""
        time_col = "date"
        if time_col in df.columns:
            sorted_df = df.sort_values(by=time_col)
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
        """Temporal 3-way split by date (train earlier -> val -> test)."""
        time_col = "date"
        sorted_df = (
            df.sort_values(by=time_col).reset_index(drop=True)
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
