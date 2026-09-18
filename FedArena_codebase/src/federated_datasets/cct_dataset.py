"""Federated Caltech Camera Traps single-label image classification."""

import json
import os

import numpy as np
import pandas as pd
from PIL import Image
from omegaconf import open_dict
from torch.utils.data import Dataset
from torchvision import transforms
from hydra.utils import instantiate

from .federated_dataset import FederatedDataset


# Process-local cache of decoded RGB frames keyed by absolute path. Each JPEG is
# decoded once per process; the pre-transform PIL image is cached so train
# augmentation still varies per access.
_IMG_CACHE = {}


class CCTDataset(FederatedDataset):
    """Federated dataset for Caltech Camera Traps single-label classification."""

    def __init__(
        self,
        cfg,
        mode,
        data_sources,
        base_path,
        images_dir,
        client_column="location",
        n_clients_select=5,
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
        # Dataset-specific normalization (computed on CCT, not ImageNet).
        cct_mean = (0.3571, 0.3626, 0.3119)
        cct_std = (0.2137, 0.2114, 0.2011)
        if train:
            return transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.65, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(cct_mean, cct_std),
            ])
        else:
            # Squash whole frame to 224x224 so no animal is cropped out at eval.
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(cct_mean, cct_std),
            ])

    def load_map_files(self):
        # COCO Camera Traps JSON; the class label lives in
        # annotations.category_id and is joined onto the image rows.
        json_path = self.data_sources[f"{self.loading_mode}_map_file"][0]
        print(f"[CCT] Loading {json_path} ...", flush=True)

        with open(json_path, "r") as f:
            data = json.load(f)

        cat_name = {c["id"]: c["name"] for c in data["categories"]}
        empty_ids = {cid for cid, n in cat_name.items() if str(n).lower() == "empty"}

        # Frames with multiple annotations keep the last category in JSON order.
        img_cat = {}
        for annotation in data["annotations"]:
            img_cat[annotation["image_id"]] = annotation["category_id"]

        df = pd.DataFrame(data["images"]).rename(columns={"id": "image_id"})
        df["category_id"] = df["image_id"].map(img_cat)

        df = df[df["category_id"].notna()].copy()
        df["category_id"] = df["category_id"].astype(int)
        df = df[~df["category_id"].isin(empty_ids)]

        top = df["location"].value_counts().nlargest(self.n_clients_select).index
        df = df[df["location"].isin(top)].reset_index(drop=True)

        # Remap surviving category_ids to contiguous [0, K-1] for CrossEntropyLoss.
        present = sorted(df["category_id"].unique())
        cat2idx = {c: i for i, c in enumerate(present)}
        df["target"] = df["category_id"].map(cat2idx)
        self._num_classes = len(present)

        print(
            f"[CCT] Selected top-{self.n_clients_select} locations; "
            f"kept {len(df)} rows, {self._num_classes} classes.",
            flush=True,
        )
        counts = df[self.client_column].value_counts().sort_values(ascending=False)
        for loc, count in counts.items():
            print(f"  location {loc}: {count}", flush=True)

        return df.reset_index(drop=True)

    def _merge_test_data_for_pool(self):
        # No-op: CCT uses one JSON for both train and test map files.
        return

    def define_num_classes(self):
        # K computed authoritatively in load_map_files; propagate to model + training.
        self.num_classes = self._num_classes
        with open_dict(self.cfg):
            self.cfg.training_params.num_classes = self.num_classes
        if self.cfg.model.num_classes != self.num_classes:
            with open_dict(self.cfg):
                self.cfg.model.num_classes = self.num_classes

    def split_to_clients(self):
        # Map each camera location to one integer client id via the column distribution.
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
    def train_val_test_split(df, random_state, train_frac=0.7, val_frac=0.15, test_frac=0.15):
        # Sequence-aware 3-way split: whole seq_id bursts stay in one split so
        # near-duplicate frames never leak across the boundary.

        if len(df) < 3:
            empty = df.iloc[0:0]
            return df.reset_index(drop=True), empty, empty

        sizes = df.groupby("seq_id").size()
        n = len(df)

        seqs = np.array(sorted(sizes.index))
        rng = np.random.RandomState(random_state)
        seqs = seqs[rng.permutation(len(seqs))]

        # Assign each whole sequence by cumulative kept-frame fraction.
        assign = {}
        cum = 0
        for s in seqs:
            frac = cum / n
            if frac < test_frac:
                assign[s] = "test"
            elif frac < test_frac + val_frac:
                assign[s] = "val"
            else:
                assign[s] = "train"
            cum += int(sizes[s])

        phase = df["seq_id"].map(assign)
        return (
            df[phase == "train"].reset_index(drop=True),
            df[phase == "val"].reset_index(drop=True),
            df[phase == "test"].reset_index(drop=True),
        )

    def __getitem__(self, index):
        # Image served from _IMG_CACHE after a single decode per process.
        row = self.data.iloc[index]
        path = os.path.join(self.images_dir, row["file_name"])
        image = _IMG_CACHE.get(path)
        if image is None:
            image = Image.open(path).convert("RGB")
            image.load()
            _IMG_CACHE[path] = image

        if "phase" in self.data.columns:
            use_eval = row["phase"] != "train"
        else:
            use_eval = self.mode != "train"
        transform = self.eval_transform if use_eval else self.train_transform
        image = transform(image)

        target = int(row["target"])
        return index, ([image], target)
