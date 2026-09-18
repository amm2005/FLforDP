#!/usr/bin/env python3
"""Resize an already-local CCT image tree without machine-specific paths.

The paper pipeline uses ``scripts/prepare_cct.py``, which downloads and resizes
the selected top-five images directly. This legacy helper is retained only for
users who already have a flat full-resolution ``data/cct/cct_images`` tree.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
TARGET = 256
QUALITY = 90


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--src",
        type=Path,
        default=ROOT / "data" / "cct" / "cct_images",
        help="flat source image directory",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=ROOT / "data" / "cct" / "cct_images_256",
        help="resized image directory",
    )
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()


def resize_one(source: Path, destination: Path) -> None:
    if destination.is_file():
        return
    try:
        with Image.open(source) as original:
            image = original.convert("RGB")
        width, height = image.size
        scale = TARGET / min(width, height)
        if scale < 1.0:
            resampling = getattr(Image, "Resampling", Image).BILINEAR
            image = image.resize(
                (
                    max(1, round(width * scale)),
                    max(1, round(height * scale)),
                ),
                resampling,
            )
        image.save(destination, "JPEG", quality=QUALITY)
    except Exception as error:
        print(f"FAIL {source.name}: {error}", flush=True)


def main() -> None:
    args = parse_args()
    if not args.src.is_dir():
        raise FileNotFoundError(f"CCT source directory not found: {args.src}")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")

    args.dst.mkdir(parents=True, exist_ok=True)
    files = [
        path
        for path in args.src.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        list(
            executor.map(
                lambda path: resize_one(path, args.dst / path.name),
                files,
            )
        )
    print(f"done: {len(files)} -> {args.dst}")


if __name__ == "__main__":
    main()
