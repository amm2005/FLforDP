    #!/usr/bin/env python3
"""Export the raw Kaggle tables required by the Home Credit Bureau-B task."""

import argparse
import shutil
import sys
from pathlib import Path

import polars as pl


COMPETITION_ROOT = Path(
    "/kaggle/input/competitions/home-credit-credit-risk-model-stability"
)
DEFAULT_OUTPUT_DIR = Path("/kaggle/working/home_credit_bureau_b")

TABLE_FILES = (
    "train_credit_bureau_b_1.parquet",
    "train_credit_bureau_b_2.parquet",
    "train_person_1.parquet",
    "train_static_cb_0.parquet",
    "train_static_0_0.parquet",
    "train_static_0_1.parquet",
    "train_base.parquet",
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        type=Path,
        default=COMPETITION_ROOT,
        help="competition directory containing parquet_files/train/",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="output directory; a ZIP archive is created beside it",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing output directory and ZIP",
    )
    return parser.parse_args(argv)


def find_source_files(train_dir: Path) -> list[Path]:
    """Resolve every required table, failing if the competition input is incomplete."""
    sources = [train_dir / name for name in TABLE_FILES]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing Kaggle table(s):\n  " + "\n  ".join(missing)
        )
    return sources


def export_tables(train_dir: Path, output_dir: Path) -> None:
    """Copy the contract table and filter supporting tables by case_id."""
    sources = find_source_files(train_dir)
    bureau_path = train_dir / "train_credit_bureau_b_1.parquet"

    case_ids = (
        pl.read_parquet(bureau_path, columns=["case_id"])
        .get_column("case_id")
        .unique()
        .sort()
    )
    print(f"Bureau-B borrowers: {len(case_ids):,}")

    if output_dir.exists():
        raise FileExistsError(
            f"Output already exists: {output_dir}. "
            "Choose another --out or pass --overwrite."
        )
    output_dir.mkdir(parents=True)

    for path in sources:
        table = pl.scan_parquet(path)
        if path != bureau_path:
            table = table.filter(pl.col("case_id").is_in(case_ids))
        filtered = table.collect()
        filtered.write_parquet(
            output_dir / path.name,
            compression="lz4",
        )
        print(f"{path.name}: {filtered.height:,} rows")


def main(argv=None) -> None:
    args = parse_args(argv)
    train_dir = args.root / "parquet_files" / "train"
    archive_path = args.out.with_suffix(".zip")
    existing = [path for path in (args.out, archive_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Output already exists; pass --overwrite to replace:\n  "
            + "\n  ".join(str(path) for path in existing)
        )
    if args.overwrite:
        if args.out.is_dir():
            shutil.rmtree(args.out)
        elif args.out.exists():
            args.out.unlink()
        if archive_path.exists():
            archive_path.unlink()

    export_tables(train_dir, args.out)
    archive = shutil.make_archive(
        str(args.out),
        "zip",
        root_dir=args.out,
    )
    print(f"Download: {archive}")


if __name__ == "__main__" and "ipykernel" not in sys.modules:
    main()
