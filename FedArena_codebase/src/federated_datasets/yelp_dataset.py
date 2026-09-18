import numpy as np
import pandas as pd
import torch
from omegaconf import open_dict, OmegaConf
from torch.utils.data import Dataset
from hydra.utils import instantiate
from transformers import AutoTokenizer, DataCollatorWithPadding
from sklearn.preprocessing import LabelEncoder

from utils.preprocessing_cache import preprocessing_cache

from .federated_dataset import FederatedDataset


class TextClassificationDataset(FederatedDataset):
    def __init__(
        self,
        cfg,
        mode,
        data_sources,
        base_path,
        client_column="PUBLISHER",
        text_column="TITLE",
        target_column="CATEGORY",
        model_name="prajjwal1/bert-tiny",
        max_len=512,
        num_classes=5,
        **kwargs,
    ):
        self.client_column = client_column
        self.text_column = text_column
        self.target_column = target_column
        self.model_name = model_name
        self.max_len = max_len
        self._num_classes_fixed = int(num_classes)

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
                    chunk = pyarrow.csv.read_csv(path).to_pandas()
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

        self.orig_data = self.data

    def _feature_engineering(self):
        self.le = LabelEncoder()
        self.data = self.data[[self.client_column, self.target_column, self.text_column]]
        self.data[self.target_column] = self.le.fit_transform(self.data[self.target_column])

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        self.collator = DataCollatorWithPadding(self.tokenizer)
        tokenized = self.tokenizer(
            self.data[self.text_column].tolist(),
            truncation=True,
            padding=False,
            max_length=self.max_len
        )
        self.data["input_ids"] = tokenized["input_ids"]
        self.data["attention_mask"] = tokenized["attention_mask"]

    def define_num_classes(self):
        """Multi-class with fixed num_classes."""
        self.num_classes = self._num_classes_fixed
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
        n = len(df)
        df = df.sample(frac=1, random_state=random_state)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        n_train = n - n_val - n_test
        train_df = df.iloc[:n_train]
        val_df = df.iloc[n_train : n_train + n_val]
        test_df = df.iloc[n_train + n_val :]
        return (
            train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True),
        )

    def _ensure_feature_buffers(self):
        key = (id(self.data), len(self.data))
        if getattr(self, "_feature_buf_key", None) == key:
            return

        self._feature_buf_key = key

        feature_cols = ['input_ids', 'attention_mask']
        target_col = self.target_column

        for col in feature_cols:
            if col not in self.data.columns:
                setattr(self, f"_buf_{col}", None)
                continue

            values = self.data[col].to_numpy(copy=False)
            if values.dtype != object:
                values = values.astype(object, copy=False)
            setattr(self, f"_buf_{col}", values)

        if target_col in self.data.columns:
            self._buf_y = np.ascontiguousarray(
                self.data[target_col].to_numpy(copy=False)
            )
        else:
            self._buf_y = None

    def __getitem__(self, index):
        self._ensure_feature_buffers()
        return {
            'input_ids': self._buf_input_ids[index],
            'attention_mask': self._buf_attention_mask[index],
            'labels': self._buf_y[index] if self._buf_y is not None else None
        }
