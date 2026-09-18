"""Federated PETA carrying-attribute multi-label dataset."""

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


# 7 mutually non-exclusive carrying labels (multi-label); sorted for a stable
# label->index mapping. Do not reorder between runs.
CARRYING_LABELS = sorted([
    "carryingBackpack",
    "carryingMessengerBag",
    "carryingNothing",
    "carryingOther",
    "carryingPlasticBags",
    "carryingSuitcase",
    "carryingLuggageCase",
])
CARRYING_LABEL_TO_IDX = {lab: i for i, lab in enumerate(CARRYING_LABELS)}
NUM_CARRYING_LABELS = len(CARRYING_LABELS)  # 7

# Per-channel normalization computed on the full PETA dataset.
PETA_MEAN = (0.4487, 0.4319, 0.4177)
PETA_STD = (0.2261, 0.2184, 0.2122)


class PETADataset(FederatedDataset):
    """Federated PETA carrying-attribute multi-label dataset."""

    def __init__(
        self,
        cfg,
        mode,
        data_sources,
        base_path,
        client_column="source_dataset",
        **kwargs,
    ):
        self.client_column = client_column
        self.images_root = str(base_path)

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
        """PETA-normalized transforms (dataset stats, not ImageNet)."""
        if train:
            return transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(PETA_MEAN, PETA_STD),
            ])
        else:
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(PETA_MEAN, PETA_STD),
            ])

    def load_map_files(self):
        """Parse PETA annotations JSON and build the dataframe with multi-hot targets."""
        json_path = self.data_sources[f"{self.loading_mode}_map_file"][0]
        print(f"[PETA] Loading {json_path} ...", flush=True)

        with open(json_path, "r") as f:
            records = json.load(f)
        df = pd.DataFrame(records)
        print(f"[PETA] Loaded {len(df)} raw records.", flush=True)

        # Rename JSON "client" to avoid collision with the integer "client"
        # column created downstream by ColumnDistribution.
        if "client" in df.columns:
            df = df.rename(columns={"client": "source_dataset"})

        df["image_path"] = (
            self.images_root
            + os.sep + df["source_dataset"].astype(str)
            + os.sep + "archive"
            + os.sep + df["image_id"].astype(str)
        )

        def to_multi_hot(labels_list):
            vec = [0] * NUM_CARRYING_LABELS
            if isinstance(labels_list, (list, tuple)):
                for lab in labels_list:
                    idx = CARRYING_LABEL_TO_IDX.get(lab)
                    if idx is not None:
                        vec[idx] = 1
            return vec

        df["target"] = df["labels"].apply(to_multi_hot)

        n_total = len(df)
        target_mat = np.array(df["target"].tolist(), dtype=np.int64)  # (N, 7)
        pos_counts = target_mat.sum(axis=0)
        row_active = target_mat.sum(axis=1)
        n_multi = int((row_active > 1).sum())
        n_all_zero = int((row_active == 0).sum())

        print(
            f"[PETA] carrying MULTI-LABEL ({NUM_CARRYING_LABELS} attributes)",
            flush=True,
        )
        print(
            f"[PETA] rows: total={n_total}, "
            f"multi-active(>=2)={n_multi} ({100*n_multi/max(n_total,1):.1f}%), "
            f"no-carrying(all-zero)={n_all_zero} "
            f"({100*n_all_zero/max(n_total,1):.1f}%) — all kept.",
            flush=True,
        )
        print(f"[PETA] Per-attribute positive frequency:", flush=True)
        for i, lab in enumerate(CARRYING_LABELS):
            print(
                f"  [{i}] {lab}: {int(pos_counts[i])} "
                f"({100*pos_counts[i]/max(n_total,1):.1f}%)",
                flush=True,
            )

        print(f"[PETA] Per-source_dataset (client) counts:", flush=True)
        src_counts = df["source_dataset"].value_counts().sort_values(ascending=False)
        for src, cnt in src_counts.items():
            print(f"  {src}: {cnt}", flush=True)

        df = df.drop(columns=["labels"])
        return df

    def _merge_test_data_for_pool(self):
        """No-op: PETA shares one JSON across train/test paths."""
        return

    def define_num_classes(self):
        """Multi-label with 7 fixed carrying attributes."""
        self.num_classes = NUM_CARRYING_LABELS
        with open_dict(self.cfg):
            self.cfg.training_params.num_classes = self.num_classes
        if self.cfg.model.num_classes != self.num_classes:
            with open_dict(self.cfg):
                self.cfg.model.num_classes = self.num_classes

    def split_to_clients(self):
        """Assign integer client column via ColumnDistribution on source_dataset."""
        print(
            f"Used distribution is: {self.distribution.__class__.__name__}",
            flush=True,
        )
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
    def train_val_test_split(
        df, random_state, train_frac=0.7, val_frac=0.15, test_frac=0.15
    ):
        """Group-wise 3-way split by person_id (all frames of a person stay together)."""
        df_sorted = df.reset_index(drop=True)
        n = len(df_sorted)
        if n == 0:
            empty = df_sorted.iloc[0:0]
            return empty.copy(), empty.copy(), empty.copy()

        persons = df_sorted["person_id"].astype(str).values
        uniq = np.array(sorted(pd.unique(persons)))
        rng = np.random.RandomState(random_state)
        rng.shuffle(uniq)

        pos_by_person = {}
        for i, p in enumerate(persons):
            pos_by_person.setdefault(p, []).append(i)

        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))

        test_idx, val_idx, train_idx = [], [], []
        filled = 0
        for p in uniq:
            rows = pos_by_person[p]
            if filled < n_test:
                test_idx.extend(rows)
            elif filled < n_test + n_val:
                val_idx.extend(rows)
            else:
                train_idx.extend(rows)
            filled += len(rows)

        def take(idx):
            return df_sorted.iloc[sorted(idx)].reset_index(drop=True)

        return take(train_idx), take(val_idx), take(test_idx)

    def __getitem__(self, index):
        """Return (index, ([image_tensor], target_tensor)) with a 7-length multi-hot target."""
        row = self.data.iloc[index]

        image = Image.open(row["image_path"]).convert("RGB")

        if "phase" in self.data.columns:
            use_eval = row["phase"] != "train"
        else:
            use_eval = self.mode != "train"
        transform = self.eval_transform if use_eval else self.train_transform
        image = transform(image)  # (3, 224, 224) float

        target = torch.as_tensor(row["target"], dtype=torch.float32)
        return index, ([image], target)
