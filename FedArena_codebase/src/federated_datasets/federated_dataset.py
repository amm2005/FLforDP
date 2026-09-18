import os
import copy
import warnings
import numpy as np
import pandas as pd
from omegaconf import open_dict, OmegaConf
from torch.utils.data import Dataset
from hydra.utils import instantiate
from sklearn.model_selection import train_test_split
from hydra.core.hydra_config import HydraConfig

from utils.dataset_utils import update_data_sources


class FederatedDataset(Dataset):
    def __init__(self, cfg, mode, data_sources, base_path):
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        self.distribution = instantiate(cfg.distribution)
        self.data_sources = data_sources

        assert (
            base_path == cfg.train_dataset.base_path
        ), f"We need to duplicate `base_path` changes to all considering datasets. You {self.mode} dataset contains base_path={base_path}"

        if self.df_exist():
            self.data_sources = update_data_sources(base_path, data_sources)
        self.name = self.__class__.__name__
        self.init_df()

    def init_df(self):
        if not self.df_exist():
            self.downloading()

        self.loading_mode = self.get_loading_mode()
        self.data = self.load_map_files()

        if self.mode == "train" and OmegaConf.select(
            self.cfg, "federated_params.server_pool_enabled", default=False
        ):
            self._merge_test_data_for_pool()

        self.preprocessing()

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

    def df_exist(self):
        if f"{self.mode}_map_file" not in self.data_sources:
            return True

        return self.data_sources[f"{self.mode}_map_file"] is not None

    def downloading(self):
        assert getattr(
            self, "target_dir", False
        ), "To update the dataset during download, the `self.target_dir` attribute is required."

        train_map_path = os.path.join(self.target_dir, "train_map_file.csv")
        test_map_path = os.path.join(self.target_dir, "test_map_file.csv")
        self.data_sources["train_map_file"] = [train_map_path]
        self.data_sources["test_map_file"] = [test_map_path]
        with open_dict(self.cfg):
            self.cfg.train_dataset.data_sources.train_map_file = [train_map_path]
            self.cfg.test_dataset.data_sources.test_map_file = [test_map_path]

    def get_loading_mode(self):
        loading_mode = self.mode

        if self.mode == "trust":
            if "trust_map_file" in self.cfg.trust_dataset.data_sources:
                loading_mode = "trust"
            else:
                if hasattr(self.cfg.trust_dataset, "trust_base_data_part"):
                    self.trust_base_data_part = (
                        self.cfg.trust_dataset.trust_base_data_part
                    )
                else:
                    print(f"We will use train as trust dataset by default.")
                    print(
                        f"You can configure this by setting `+trust_dataset.trust_base_data_part=<train/test>`\n"
                    )
                    self.trust_base_data_part = "train"

                loading_mode = self.trust_base_data_part
        return loading_mode

    def load_map_files(self):
        df = pd.DataFrame()
        for directories in self.data_sources[f"{self.loading_mode}_map_file"]:
            df = pd.concat([df, pd.read_csv(directories, low_memory=False)])

        return df

    def _merge_test_data_for_pool(self):
        """Load test_map_file and concat into train data when server_pool_enabled."""
        if "test_map_file" not in self.data_sources:
            return
        if self.data_sources["test_map_file"] is None:
            return

        saved = self.loading_mode
        self.loading_mode = "test"
        try:
            test_data = self.load_map_files()
        finally:
            self.loading_mode = saved

        n_before = len(self.data)
        self.data = pd.concat([self.data, test_data], ignore_index=True)
        print(
            f"[server_pool] Merged test_map_file: {n_before} train + "
            f"{len(test_data)} test = {len(self.data)} total rows.",
            flush=True,
        )

    def parse_trust(self):
        if self.cfg.train_dataset["_target_"] == self.cfg.trust_dataset["_target_"]:
            if self.mode == "train":
                return self.repeating_trust_df()
        else:
            if self.mode == "trust":
                return self.separate_trust_df()

    def repeating_trust_df(self):
        df = self.data

        num_trust_samples = 500
        if "num_trust_samples" in self.cfg.trust_dataset:
            num_trust_samples = self.cfg.trust_dataset.num_trust_samples
        else:
            print(f"We will trust dataset with {num_trust_samples} by default.")
            print(
                f"You can configure this number by setting `+trust_dataset.num_trust_samples=number`\n"
            )

        val_prop = num_trust_samples / len(df)

        train_df, trust_df = self.train_val_split(
            df, val_prop, random_state=self.cfg.random_state
        )

        save_dir = HydraConfig.get().runtime.output_dir
        save_path = os.path.join(save_dir, "trust_map_file.csv")
        trust_df.to_csv(save_path)
        print(f"New trust map-file saved in: {save_path}\n")

        with open_dict(self.cfg):
            self.cfg.trust_dataset.data_sources.trust_map_file = [save_path]

        self.data = train_df

    def separate_trust_df(self):
        df = self.data
        num_trust_samples = 500
        if "num_trust_samples" in self.cfg.trust_dataset:
            num_trust_samples = self.cfg.trust_dataset.num_trust_samples
        else:
            print(f"We will trust dataset with {num_trust_samples} by default.")
            print(
                f"You can configure this number by setting `+trust_dataset.num_trust_samples=number`\n"
            )

        train_val_prop = num_trust_samples / len(df)

        _, trust_df = self.train_val_split(
            df, train_val_prop, random_state=self.cfg.random_state
        )

        save_dir = HydraConfig.get().runtime.output_dir
        save_path = os.path.join(save_dir, "trust_map_file.csv")
        print(f"New trust map-file saved in: {save_path}\n")
        trust_df.to_csv(save_path)
        with open_dict(self.cfg):
            self.cfg.trust_dataset.data_sources.trust_map_file = [save_path]

        self.data = trust_df

    def preprocessing(self):
        pass

    def define_num_classes(self):
        if isinstance(self.data.iloc[0]["target"], list):
            self.num_classes = len(self.data.iloc[0]["target"])
        else:
            self.num_classes = pd.Series(
                np.concatenate(
                    self.data["target"]
                    .apply(lambda x: x if isinstance(x, list) else [x])
                    .values
                )
            ).nunique()

        with open_dict(self.cfg):
            self.cfg.training_params.num_classes = self.num_classes
        if self.cfg.model.num_classes != self.num_classes:
            with open_dict(self.cfg):
                self.cfg.model.num_classes = self.num_classes

    def split_to_clients(self):
        print(f"Used distribution is: {self.distribution.__class__.__name__}")
        self.data = self.distribution.split_to_clients(
            self.data,
            self.cfg.federated_params.amount_of_clients,
            self.cfg.random_state,
        )

    def apply_client_server_phase_and_holdout(self):
        """Label each row with phase ('train'/'val'/'test') and server_pool flag."""
        fp = self.cfg.federated_params
        train_frac = float(OmegaConf.select(fp, "client_train_frac", default=0.7))
        val_frac = float(OmegaConf.select(fp, "client_val_frac", default=0.15))
        test_frac = float(OmegaConf.select(fp, "client_test_frac", default=0.15))
        server_val_frac = float(OmegaConf.select(fp, "server_val_frac", default=0.5))
        server_test_frac = float(OmegaConf.select(fp, "server_test_frac", default=0.5))

        df = self.data.reset_index(drop=True)
        n = len(df)

        df["_tmp_id"] = np.arange(n, dtype=np.int64)

        phase_arr = np.full(n, "train", dtype=object)
        server_pool_arr = np.zeros(n, dtype=bool)

        rng = np.random.RandomState(self.cfg.random_state)

        for client_id in sorted(df["client"].unique()):
            client_mask = df["client"].values == client_id
            client_df = df.loc[client_mask].copy()

            if len(client_df) == 0:
                continue

            train_df, val_df, test_df = type(self).train_val_test_split(
                client_df,
                random_state=self.cfg.random_state,
                train_frac=train_frac,
                val_frac=val_frac,
                test_frac=test_frac,
            )

            if len(val_df) > 0:
                val_ids = val_df["_tmp_id"].values
                phase_arr[val_ids] = "val"
                n_srv_val = max(0, int(len(val_ids) * server_val_frac))
                if n_srv_val > 0:
                    srv_val_ids = rng.choice(val_ids, size=n_srv_val, replace=False)
                    server_pool_arr[srv_val_ids] = True

            if len(test_df) > 0:
                test_ids = test_df["_tmp_id"].values
                phase_arr[test_ids] = "test"
                n_srv_test = max(0, int(len(test_ids) * server_test_frac))
                if n_srv_test > 0:
                    srv_test_ids = rng.choice(test_ids, size=n_srv_test, replace=False)
                    server_pool_arr[srv_test_ids] = True
                    
            print(f"Client {client_id}: train_df: {train_df.shape}, val_df (before pooling): {val_df.shape}, test_df (before pooling): {test_df.shape}")

        df["phase"] = phase_arr
        df["server_pool"] = server_pool_arr
        df = df.drop(columns=["_tmp_id"])

        n_pool_val = int(server_pool_arr[phase_arr == "val"].sum())
        n_pool_test = int(server_pool_arr[phase_arr == "test"].sum())
        print(
            f"[server_pool] val={n_pool_val} rows, test={n_pool_test} rows "
            f"pooled across {df['client'].nunique()} clients.",
            flush=True,
        )

        self.data = df

    def get_server_val_dataset(self):
        """Shallow-copy view of pooled server val rows across all clients."""
        obj = copy.copy(self)
        mask = self.orig_data["server_pool"] & (self.orig_data["phase"] == "val")
        obj.data = self.orig_data[mask].reset_index(drop=True)
        obj._feature_buf_key = None
        return obj

    def get_server_test_dataset(self):
        """Shallow-copy view of pooled server test rows across all clients."""
        obj = copy.copy(self)
        mask = self.orig_data["server_pool"] & (self.orig_data["phase"] == "test")
        obj.data = self.orig_data[mask].reset_index(drop=True)
        obj._feature_buf_key = None
        return obj

    def to_client_side(self, rank):
        self.rank = rank
        if "server_pool" in self.orig_data.columns:
            client_mask = (self.orig_data["client"] == rank) & (
                ~self.orig_data["server_pool"]
            )
        else:
            client_mask = self.orig_data["client"] == rank
        self.data = self.orig_data[client_mask].reset_index(drop=True)
        return self

    def dataset_split(self, train_val_prop):
        train_data, valid_data = self.train_val_split(
            self.data, train_val_prop, self.cfg.random_state
        )
        valid_dataset = copy.deepcopy(self)
        valid_dataset.data = valid_data
        valid_dataset.mode = "valid"
        self.data = train_data
        return valid_dataset

    @staticmethod
    def train_val_split(df, train_val_prop, random_state):
        df = df.copy()
        is_multilabel = (
            isinstance(df["target"].iloc[0], list) and len(df["target"].iloc[0]) > 1
        )

        if is_multilabel:
            df.loc[:, "strat_target"] = df["target"].apply(lambda x: tuple(x))
        else:
            df.loc[:, "strat_target"] = df["target"]

        value_counts = df["strat_target"].value_counts()
        major_keys = value_counts[value_counts >= 2].index
        major_classes_df = df[df["strat_target"].isin(major_keys)].copy()
        minor_classes_df = df[~df["strat_target"].isin(major_keys)].copy()
        n_major_classes = len(major_keys)

        if (
            len(major_classes_df) == 0
            or train_val_prop * len(major_classes_df) < n_major_classes
        ):
            train_df, valid_df = train_test_split(
                major_classes_df,
                test_size=train_val_prop,
                random_state=random_state,
            )
            train_df = pd.concat([train_df, minor_classes_df], ignore_index=True)

            return train_df.reset_index(drop=True), valid_df.reset_index(drop=True)

        stratify = major_classes_df["strat_target"]
        min_freq = stratify.value_counts().min()
        max_allowed_ratio = (min_freq - 1) / min_freq
        if max_allowed_ratio < train_val_prop:
            train_val_prop = max_allowed_ratio

        train_df, valid_df = train_test_split(
            major_classes_df,
            test_size=train_val_prop,
            stratify=stratify,
            random_state=random_state,
        )
        train_df = pd.concat([train_df, minor_classes_df], ignore_index=True)

        return train_df.reset_index(drop=True), valid_df.reset_index(drop=True)

    @staticmethod
    def train_val_test_split(df, random_state, train_frac=0.7, val_frac=0.15, test_frac=0.15):
        """Fallback 3-way stratified split for datasets without a time column."""
        val_test_frac = val_frac + test_frac
        if len(df) < 3 or val_test_frac <= 0:
            empty = pd.DataFrame(columns=df.columns)
            return df.reset_index(drop=True), empty, empty

        train_df, val_test_df = FederatedDataset.train_val_split(
            df, val_test_frac, random_state
        )
        if len(val_test_df) < 2:
            empty = pd.DataFrame(columns=df.columns)
            return train_df.reset_index(drop=True), val_test_df.reset_index(drop=True), empty

        test_frac_of_remainder = test_frac / val_test_frac
        val_df, test_df = FederatedDataset.train_val_split(
            val_test_df, test_frac_of_remainder, random_state
        )
        return (
            train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True),
        )

    def get_cfg(self):
        return self.cfg

    def __getitem__(self, index):
        raise NotImplementedError(
            f"You need to implement __getitem__ function " f"in {self.name} dataset!"
        )

    def __len__(self):
        return len(self.data)
