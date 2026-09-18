#!/usr/bin/env python3
"""Build the Fed-ISIC2019 manifest used by the paper experiments.

First use the official FLamby scripts to download and preprocess the images:

    git clone https://github.com/owkin/FLamby.git ../FLamby
    python -m pip install -e ../FLamby
    python ../FLamby/flamby/datasets/fed_isic2019/dataset_creation_scripts/download_isic.py \
        --output-folder data/fed_isic2019
    python ../FLamby/flamby/datasets/fed_isic2019/dataset_creation_scripts/resize_images.py

Then run this script from the Federated-Research-LIB-dev root:

    python scripts/prepare_fed_isic2019.py \
        --split-file ../FLamby/flamby/datasets/fed_isic2019/dataset_creation_scripts/train_test_split

The official scripts produce the resized images. This script copies the
official split into the flat CSV format consumed by ``FedISIC2019Dataset`` and
checks that every referenced image exists.

Inputs:

    FLamby's dataset_creation_scripts/train_test_split
    data/fed_isic2019/ISIC_2019_Training_Input_preprocessed/*.jpg

Output:

    data/fed_isic2019/fed_isic2019_map.csv
"""

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "fed_isic2019"
KEEP_COLUMNS = ["image", "target", "center", "fold", "fold2"]
EXPECTED_ROWS = 23_247
EXPECTED_TEST_ROWS = 4_650
EXPECTED_CENTERS = set(range(6))
EXPECTED_TARGETS = set(range(8))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--split-file",
        type=Path,
        required=True,
        help="FLamby's dataset_creation_scripts/train_test_split file",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DATA_DIR / "ISIC_2019_Training_Input_preprocessed",
        help="directory produced by FLamby's resize_images.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_DIR / "fed_isic2019_map.csv",
        help="experiment manifest to write",
    )
    return parser.parse_args()


def load_official_split(split_file: Path) -> pd.DataFrame:
    """Load and validate the official FLamby split."""
    if not split_file.is_file():
        raise FileNotFoundError(
            f"FLamby split not found: {split_file}"
        )

    source = pd.read_csv(split_file)
    missing_columns = [
        column for column in KEEP_COLUMNS if column not in source.columns
    ]
    if missing_columns:
        raise ValueError(
            f"FLamby split is missing columns {missing_columns}. "
            f"Found: {list(source.columns)}"
        )

    manifest = source[KEEP_COLUMNS].copy()
    manifest["target"] = manifest["target"].astype(int)
    manifest["center"] = manifest["center"].astype(int)
    if manifest.isna().any().any():
        raise ValueError("FLamby split contains null values in required columns")
    if manifest["image"].duplicated().any():
        duplicates = manifest.loc[
            manifest["image"].duplicated(), "image"
        ].head(5).tolist()
        raise ValueError(f"FLamby split contains duplicate images: {duplicates}")
    if len(manifest) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS:,} FLamby rows, found {len(manifest):,}"
        )
    if set(manifest["fold"].unique()) != {"train", "test"}:
        raise ValueError(
            f"Unexpected fold values: {sorted(manifest['fold'].unique())}"
        )
    n_test = int((manifest["fold"] == "test").sum())
    if n_test != EXPECTED_TEST_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_TEST_ROWS:,} official test rows, found {n_test:,}"
        )
    if set(manifest["center"].unique()) != EXPECTED_CENTERS:
        raise ValueError(
            f"Expected centers 0..5, found {sorted(manifest['center'].unique())}"
        )
    if set(manifest["target"].unique()) != EXPECTED_TARGETS:
        raise ValueError(
            f"Expected targets 0..7, found {sorted(manifest['target'].unique())}"
        )
    expected_fold2 = (
        manifest["fold"]
        + "_"
        + manifest["center"].astype(str)
    )
    if not manifest["fold2"].astype(str).equals(expected_fold2):
        raise ValueError("fold2 must equal '<fold>_<center>' for every row")
    return manifest


def validate_images(manifest: pd.DataFrame, image_dir: Path) -> None:
    """Require every image referenced by the manifest."""
    if not image_dir.is_dir():
        raise FileNotFoundError(
            f"Preprocessed image directory not found: {image_dir}\n"
            "Run FLamby's download_isic.py and resize_images.py first."
        )

    present = {path.stem for path in image_dir.glob("*.jpg")}
    if not present:
        raise FileNotFoundError(f"No JPG images found in {image_dir}")

    missing = manifest.loc[~manifest["image"].isin(present), "image"]
    if not missing.empty:
        examples = ", ".join(f"{image}.jpg" for image in missing.head(5))
        raise FileNotFoundError(
            f"{len(missing):,}/{len(manifest):,} images are missing from "
            f"{image_dir}. Examples: {examples}"
        )


def write_manifest(manifest: pd.DataFrame, output: Path) -> None:
    """Write the validated manifest."""
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output, index=False)


def main() -> None:
    args = parse_args()
    manifest = load_official_split(args.split_file)
    validate_images(manifest, args.image_dir)
    write_manifest(manifest, args.output)

    print(f"Official split: {args.split_file}")
    print(f"Preprocessed images: {args.image_dir}")
    print(f"Rows: {len(manifest):,}")
    for center, count in (
        manifest["center"].value_counts().sort_index().items()
    ):
        print(f"  center {center}: {count:,}")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
