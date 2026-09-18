"""FLamby Fed-ISIC2019 dataset (8-class dermoscopy, one client per center)."""

import copy
import os
import random

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from omegaconf import open_dict
from torch.utils.data import Dataset
from hydra.utils import instantiate

from .federated_dataset import FederatedDataset


_IMG_CACHE = {}


def _flamby_flip(image, **kwargs):
    """Replicate the random flip axis of the removed Albumentations A.Flip."""
    return cv2.flip(image, random.randint(-1, 1))


class FedISIC2019Dataset(FederatedDataset):
    """Federated dataset for FLamby Fed-ISIC2019 (8-class, single-label)."""

    def __init__(
        self,
        cfg,
        mode,
        data_sources,
        base_path,
        images_dir,
        client_column="center",
        **kwargs,
    ):
        self.client_column = client_column
        self.images_dir = images_dir

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
        """Apply FLamby's crop-200 augmentation and ImageNet normalization."""
        crop_size = 200
        if train:
            return A.Compose(
                [
                    A.RandomScale(scale_limit=0.07, p=0.5),
                    A.Rotate(limit=50, p=0.5),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.15, contrast_limit=0.10, p=0.5
                    ),
                    A.Lambda(image=_flamby_flip, p=0.5, name="FlambyFlip"),
                    A.Affine(shear=0.1, p=0.5),
                    A.RandomCrop(height=crop_size, width=crop_size, p=1.0),
                    A.CoarseDropout(
                        num_holes_range=(1, 8),
                        hole_height_range=(16, 16),
                        hole_width_range=(16, 16),
                        p=0.5,
                    ),
                    A.Normalize(),
                ]
            )
        return A.Compose(
            [
                A.CenterCrop(height=crop_size, width=crop_size, p=1.0),
                A.Normalize(),
            ]
        )

    def load_map_files(self):
        """Load the manifest and select the requested official fold."""
        dfs = []
        for csv_path in self.data_sources[f"{self.loading_mode}_map_file"]:
            print(f"[Fed-ISIC2019] Loading {csv_path} ...", flush=True)
            dfs.append(pd.read_csv(csv_path))
        df = pd.concat(dfs, ignore_index=True)
        df["target"] = df["target"].astype(int)
        df["center"] = df["center"].astype(int)
        fold = "train" if self.loading_mode == "train" else "test"
        df = df[df["fold"] == fold].reset_index(drop=True)
        print(f"[Fed-ISIC2019] fold={fold}: {len(df)} rows", flush=True)
        return df

    def _merge_test_data_for_pool(self):
        """Keep the official test fold out of the training pool."""
        return

    def get_server_test_dataset(self):
        """Return FLamby's official test fold instead of a pooled holdout."""
        obj = copy.copy(self)
        dfs = [pd.read_csv(p) for p in self.data_sources["test_map_file"]]
        test = pd.concat(dfs, ignore_index=True)
        test["target"] = test["target"].astype(int)
        test["center"] = test["center"].astype(int)
        obj.data = test[test["fold"] == "test"].reset_index(drop=True)
        obj.mode = "test"
        obj._feature_buf_key = None
        print(
            f"[Fed-ISIC2019] server test = official FLamby test fold: "
            f"{len(obj.data)} rows",
            flush=True,
        )
        return obj

    def define_num_classes(self):
        """Set the fixed eight-class output size."""
        self.num_classes = 8
        with open_dict(self.cfg):
            self.cfg.training_params.num_classes = self.num_classes
        if self.cfg.model.num_classes != self.num_classes:
            with open_dict(self.cfg):
                self.cfg.model.num_classes = self.num_classes

    def split_to_clients(self):
        """Assign one client to each acquisition center."""
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

    def __getitem__(self, index):
        row = self.data.iloc[index]
        path = os.path.join(self.images_dir, f"{row['image']}.jpg")
        image = _IMG_CACHE.get(path)
        if image is None:
            image = np.array(Image.open(path).convert("RGB"))
            _IMG_CACHE[path] = image

        if "phase" in self.data.columns:
            use_eval = row["phase"] != "train"
        else:
            use_eval = self.mode != "train"
        transform = self.eval_transform if use_eval else self.train_transform

        image = transform(image=image)["image"]
        image = np.transpose(image, (2, 0, 1))
        image = torch.as_tensor(image, dtype=torch.float32)

        target = int(row["target"])
        return index, ([image], target)
