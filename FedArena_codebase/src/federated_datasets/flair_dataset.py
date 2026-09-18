"""FLAIR multi-label image classification dataset."""

import json
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from omegaconf import open_dict
from torch.utils.data import Dataset
from torchvision import transforms
from hydra.utils import instantiate

from .federated_dataset import FederatedDataset


# Alphabetical order fixes the output-head index of each coarse label.
FLAIR_LABELS_17 = sorted([
    "structure", "equipment", "material", "outdoor", "plant",
    "food", "animal", "liquid", "art", "interior_room",
    "light", "recreation", "celebration", "fire", "music",
    "games", "religion",
])
FLAIR_LABEL_TO_IDX = {label: i for i, label in enumerate(FLAIR_LABELS_17)}
NUM_FLAIR_LABELS = len(FLAIR_LABELS_17)


_IMG_CACHE = {}


class FLAIRDataset(FederatedDataset):
    """Federated dataset for FLAIR multi-label image classification."""

    def __init__(
        self,
        cfg,
        mode,
        data_sources,
        base_path,
        images_dir,
        client_column="user_id",
        n_clients_select=10,
        **kwargs,
    ):
        self.client_column = client_column
        self.images_dir = images_dir
        self.n_clients_select = int(n_clients_select)

        self.train_transform = self._set_up_transform(train=True)
        self.eval_transform = self._set_up_transform(train=False)

        Dataset.__init__(self)
        self.cfg = cfg
        self.mode = mode
        self.distribution = instantiate(cfg.distribution)
        self.data_sources = data_sources
        self.name = self.__class__.__name__
        self.init_df()

    @staticmethod
    def _set_up_transform(train):
        flair_mean = (0.4892, 0.4552, 0.4044)
        flair_std = (0.2738, 0.2667, 0.2809)
        if train:
            return transforms.Compose([
                transforms.RandomCrop(224, padding=0),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(flair_mean, flair_std),
            ])
        else:
            return transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(flair_mean, flair_std),
            ])

    def load_map_files(self):
        """Load metadata, keep the largest users, and encode coarse labels."""
        json_path = self.data_sources[f"{self.loading_mode}_map_file"][0]
        print(f"[FLAIR] Loading {json_path} ...", flush=True)

        with open(json_path, "r") as f:
            records = json.load(f)
        df = pd.DataFrame(records)
        print(f"[FLAIR] Loaded {len(df)} raw records.", flush=True)

        for drop_col in ("fine_grained_labels", "partition"):
            if drop_col in df.columns:
                df = df.drop(columns=[drop_col])

        top_users = (
            df[self.client_column]
            .value_counts()
            .nlargest(self.n_clients_select)
            .index.tolist()
        )
        df = df[df[self.client_column].isin(top_users)].reset_index(drop=True)

        unknown_labels = set()

        def to_multi_hot(labels_list):
            vec = [0] * NUM_FLAIR_LABELS
            for label in labels_list:
                idx = FLAIR_LABEL_TO_IDX.get(label)
                if idx is None:
                    unknown_labels.add(label)
                else:
                    vec[idx] = 1
            return vec

        df["target"] = df["labels"].apply(to_multi_hot)
        assert not unknown_labels, (
            f"[FLAIR] Found label strings not in FLAIR_LABELS_17: "
            f"{sorted(unknown_labels)}. Update FLAIR_LABELS_17 to match the exact "
            f"coarse-label strings in labels_and_metadata.json."
        )
        df = df.drop(columns=["labels"])

        print(
            f"[FLAIR] Selected top-{self.n_clients_select} users; "
            f"kept {len(df)} rows. Per-user counts:",
            flush=True,
        )
        counts = df[self.client_column].value_counts().sort_values(ascending=False)
        for uid, count in counts.items():
            print(f"  {uid}: {count}", flush=True)

        return df

    def _merge_test_data_for_pool(self):
        """Do not merge a test file; the server pool comes from the full JSON."""
        return

    def define_num_classes(self):
        """FLAIR is multi-label with 17 fixed classes — hard-code it."""
        self.num_classes = NUM_FLAIR_LABELS
        with open_dict(self.cfg):
            self.cfg.training_params.num_classes = self.num_classes
        if self.cfg.model.num_classes != self.num_classes:
            with open_dict(self.cfg):
                self.cfg.model.num_classes = self.num_classes

    def split_to_clients(self):
        """Assign one client to each selected Flickr user."""
        print(f"Used distribution is: {self.distribution.__class__.__name__}", flush=True)
        self.data = self.distribution.split_to_clients(
            self.data,
            self.cfg.federated_params.amount_of_clients,
            self.cfg.random_state,
        )
        actual_clients = self.data["client"].nunique()
        with open_dict(self.cfg):
            self.cfg.federated_params.amount_of_clients = actual_clients
        print(f"Amount of clients set to: {actual_clients}", flush=True)

    @staticmethod
    def train_val_split(df, train_val_prop, random_state):
        """Random val split. train_val_prop ∈ (0, 1) is the val fraction."""
        df_sorted = df.sort_values("image_id").reset_index(drop=True)
        rng = np.random.RandomState(random_state)
        perm = rng.permutation(len(df_sorted))
        n_val = max(1, int(len(df_sorted) * train_val_prop))
        n_train = len(df_sorted) - n_val
        train_idx = perm[:n_train]
        val_idx = perm[n_train:]
        return (
            df_sorted.iloc[train_idx].reset_index(drop=True),
            df_sorted.iloc[val_idx].reset_index(drop=True),
        )

    @staticmethod
    def train_val_test_split(
        df, random_state, train_frac=0.7, val_frac=0.15, test_frac=0.15
    ):
        """Split a client deterministically after sorting by image ID."""
        df_sorted = df.sort_values("image_id").reset_index(drop=True)
        rng = np.random.RandomState(random_state)
        perm = rng.permutation(len(df_sorted))
        n = len(df_sorted)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        n_train = n - n_val - n_test
        train_idx = perm[:n_train]
        val_idx = perm[n_train : n_train + n_val]
        test_idx = perm[n_train + n_val :]
        return (
            df_sorted.iloc[train_idx].reset_index(drop=True),
            df_sorted.iloc[val_idx].reset_index(drop=True),
            df_sorted.iloc[test_idx].reset_index(drop=True),
        )

    def __getitem__(self, index):
        row = self.data.iloc[index]

        image_path = os.path.join(self.images_dir, f"{row['image_id']}.jpg")
        image = _IMG_CACHE.get(image_path)
        if image is None:
            image = Image.open(image_path).convert("RGB")
            image.load()
            _IMG_CACHE[image_path] = image

        if "phase" in self.data.columns:
            use_eval = row["phase"] != "train"
        else:
            use_eval = self.mode != "train"
        transform = self.eval_transform if use_eval else self.train_transform
        image = transform(image)

        target = torch.as_tensor(row["target"], dtype=torch.float32)

        return index, ([image], target)
