#!/usr/bin/env python3
"""Prepare the Caltech Camera Traps subset used in the paper experiments."""

import argparse
import io
import json
import shutil
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]

ANNOTATION_URL = (
    "https://storage.googleapis.com/public-datasets-lila/"
    "caltechcameratraps/labels/caltech_camera_traps.json.zip"
)
IMAGE_BASE_URL = (
    "https://storage.googleapis.com/public-datasets-lila/"
    "caltech-unzipped/cct_images"
)
ARCHIVE_FILENAME = "cct_ann.json.zip"
ANNOTATION_FILENAME = "caltech_images_20210113.json"
PAPER_CLIENTS = 5
PAPER_LOCATION_COUNTS = {
    "38": 9_548,
    "46": 5_332,
    "114": 5_267,
    "100": 4_573,
    "57": 4_334,
}
PAPER_IMAGES = 29_054
PAPER_CLASSES = 15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data" / "cct",
        help="directory for the full JSON and resized paper images",
    )
    parser.add_argument(
        "--n-clients",
        type=int,
        default=PAPER_CLIENTS,
        help="number of largest non-empty camera locations to download",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="maximum shorter-side size for downloaded images",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=90,
        help="JPEG quality for resized images",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="number of concurrent image downloads",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="download attempts per image",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the selected subset without downloading images",
    )
    return parser.parse_args()


def download_file(url: str, destination: Path) -> None:
    """Download one file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, destination)


def ensure_annotation_json(out_dir: Path) -> Path:
    """Download and extract the complete CCT annotation JSON when needed."""
    json_path = out_dir / ANNOTATION_FILENAME
    if json_path.is_file():
        return json_path

    archive_path = out_dir / ARCHIVE_FILENAME
    if not archive_path.is_file():
        print(f"Downloading annotations: {ANNOTATION_URL}", flush=True)
        download_file(ANNOTATION_URL, archive_path)

    with zipfile.ZipFile(archive_path) as archive:
        json_members = [
            name for name in archive.namelist()
            if Path(name).name == ANNOTATION_FILENAME
        ]
        if not json_members:
            json_members = [
                name for name in archive.namelist()
                if name.lower().endswith(".json")
            ]
        if len(json_members) != 1:
            raise ValueError(
                "Expected one annotation JSON in "
                f"{archive_path}, found: {json_members}"
            )

        with archive.open(json_members[0]) as source, json_path.open("wb") as target:
            shutil.copyfileobj(source, target)

    return json_path


def select_images(
    json_path: Path,
    n_clients: int,
) -> tuple[list[dict], list[str], Counter, int]:
    """Select top locations using the exact policy used by the paper loader."""
    with json_path.open("r") as file:
        data = json.load(file)

    category_names = {
        category["id"]: category["name"]
        for category in data["categories"]
    }
    empty_ids = {
        category_id
        for category_id, name in category_names.items()
        if str(name).lower() == "empty"
    }

    # Some images have several annotations; dict assignment keeps the last in JSON order, matching the paper runs.
    image_category = {}
    for annotation in data["annotations"]:
        image_category[annotation["image_id"]] = annotation["category_id"]

    non_empty = [
        image
        for image in data["images"]
        if image.get("id") in image_category
        and image_category[image["id"]] not in empty_ids
    ]
    location_counts = Counter(
        str(image["location"])
        for image in non_empty
    )
    top_locations = [
        location
        for location, _ in location_counts.most_common(n_clients)
    ]
    top_set = set(top_locations)
    selected = [
        image
        for image in non_empty
        if str(image["location"]) in top_set
    ]
    selected_classes = {
        image_category[image["id"]]
        for image in selected
    }

    return selected, top_locations, location_counts, len(selected_classes)


def validate_paper_subset(
    selected: list[dict],
    locations: list[str],
    location_counts: Counter,
    n_classes: int,
    n_clients: int,
) -> None:
    """Fail if the default source no longer produces the paper subset."""
    if n_clients != PAPER_CLIENTS:
        return

    actual_counts = {
        location: location_counts[location]
        for location in locations
    }
    if actual_counts != PAPER_LOCATION_COUNTS:
        raise ValueError(
            "The annotation JSON does not produce the paper's top-five "
            f"locations. Found: {actual_counts}"
        )
    if len(selected) != PAPER_IMAGES or n_classes != PAPER_CLASSES:
        raise ValueError(
            "Unexpected paper subset: "
            f"images={len(selected):,}, classes={n_classes}; "
            f"expected images={PAPER_IMAGES:,}, classes={PAPER_CLASSES}."
        )


def safe_relative_path(file_name: str) -> Path:
    """Validate an annotation path before joining it to the output directory."""
    relative = Path(file_name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe image file_name in annotations: {file_name!r}")
    return relative


def resized_dimensions(
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    """Return the dimensions produced by the paper resize."""
    scale = image_size / min(width, height)
    if scale >= 1.0:
        return width, height
    return (
        max(1, round(width * scale)),
        max(1, round(height * scale)),
    )


def output_is_valid(path: Path, expected_size: tuple[int, int]) -> bool:
    """Return whether an existing output is the expected RGB JPEG."""
    if not path.is_file():
        return False
    try:
        with Image.open(path) as image:
            image.load()
            return (
                image.format == "JPEG"
                and image.mode == "RGB"
                and image.size == expected_size
            )
    except Exception:
        return False


def resize_for_experiment(image: Image.Image, image_size: int) -> Image.Image:
    """Match the offline resize used for the paper images."""
    image = image.convert("RGB")
    width, height = image.size
    output_size = resized_dimensions(width, height, image_size)
    if output_size == image.size:
        return image

    resampling = getattr(Image, "Resampling", Image).BILINEAR
    return image.resize(output_size, resampling)


def prepare_image(
    image_record: dict,
    images_dir: Path,
    image_size: int,
    jpeg_quality: int,
    retries: int,
) -> tuple[str, str]:
    """Download and resize one image, or reuse a valid existing output."""
    file_name = image_record["file_name"]
    relative = safe_relative_path(file_name)
    destination = images_dir / relative
    expected_size = resized_dimensions(
        int(image_record["width"]),
        int(image_record["height"]),
        image_size,
    )
    if output_is_valid(destination, expected_size):
        return file_name, "existing"

    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded_name = urllib.parse.quote(file_name, safe="/")
    url = f"{IMAGE_BASE_URL}/{encoded_name}"
    last_error = None

    for _ in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                payload = response.read()
            with Image.open(io.BytesIO(payload)) as source:
                output = resize_for_experiment(source, image_size)
                output.save(destination, "JPEG", quality=jpeg_quality)
            return file_name, "written"
        except Exception as error:
            last_error = error
            if destination.exists():
                destination.unlink()

    return file_name, f"failed: {last_error}"


def prepare_images(
    selected: list[dict],
    images_dir: Path,
    image_size: int,
    jpeg_quality: int,
    workers: int,
    retries: int,
) -> None:
    """Prepare all selected images and fail if any output is missing."""
    images_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    written = 0
    existing = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                prepare_image,
                image,
                images_dir,
                image_size,
                jpeg_quality,
                retries,
            )
            for image in selected
        ]
        for index, future in enumerate(as_completed(futures), start=1):
            file_name, status = future.result()
            if status == "written":
                written += 1
            elif status == "existing":
                existing += 1
            else:
                failures.append((file_name, status))

            if index % 500 == 0 or index == len(futures):
                print(
                    f"Prepared {index:,}/{len(futures):,} "
                    f"(new={written:,}, existing={existing:,}, "
                    f"failed={len(failures):,})",
                    flush=True,
                )

    if failures:
        details = "\n".join(
            f"  {file_name}: {status}"
            for file_name, status in failures[:20]
        )
        raise RuntimeError(
            f"Failed to prepare {len(failures):,} CCT images. "
            f"First failures:\n{details}"
        )


def main() -> None:
    args = parse_args()
    if args.n_clients <= 0:
        raise ValueError("--n-clients must be positive")
    if args.image_size <= 0:
        raise ValueError("--image-size must be positive")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 1 and 100")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.retries <= 0:
        raise ValueError("--retries must be positive")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = ensure_annotation_json(args.out_dir)
    selected, locations, counts, n_classes = select_images(
        json_path,
        args.n_clients,
    )
    validate_paper_subset(
        selected,
        locations,
        counts,
        n_classes,
        args.n_clients,
    )

    print(f"Annotations: {json_path}")
    print(f"Selected locations: {locations}")
    for location in locations:
        print(f"  location {location}: {counts[location]:,} images")
    print(f"Images: {len(selected):,}")
    print(f"Classes: {n_classes}")

    if args.dry_run:
        print("Dry run complete; no images were downloaded.")
        return

    images_dir = args.out_dir / f"cct_images_{args.image_size}"
    prepare_images(
        selected,
        images_dir,
        args.image_size,
        args.jpeg_quality,
        args.workers,
        args.retries,
    )
    print(f"Prepared images: {images_dir}")


if __name__ == "__main__":
    main()
