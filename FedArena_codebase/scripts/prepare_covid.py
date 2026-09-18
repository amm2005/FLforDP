#!/usr/bin/env python3
"""Build the COVID-19 CSV files used in the paper experiments."""

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DAILY_FILENAME = "worldometer_coronavirus_daily_data.csv"
SUMMARY_FILENAME = "worldometer_coronavirus_summary_data.csv"

CLIENT_COLUMN = "country"
POPULATION_COLUMN = "population"
DATE_COLUMN = "date"
N_CLIENTS = 10
TRAIN_RATIO = 0.8

OUTPUT_COLUMNS = [
    "date",
    "country",
    "cumulative_total_cases",
    "daily_new_cases",
    "active_cases",
    "cumulative_total_deaths",
    "daily_new_deaths",
]

EXPECTED_COUNTRIES = {
    "Bangladesh",
    "Brazil",
    "China",
    "India",
    "Indonesia",
    "Mexico",
    "Nigeria",
    "Pakistan",
    "Russia",
    "USA",
}
EXPECTED_ROWS = 8_224
EXPECTED_TRAIN_ROWS = 6_579
EXPECTED_TEST_ROWS = 1_645


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=ROOT / "data" / "covid19" / "raw",
        help="directory containing the two complete Kaggle CSV files",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data" / "covid19",
        help="directory for train_df.csv and test_df.csv",
    )
    return parser.parse_args()


def find_input_files(raw_dir: Path) -> tuple[Path, Path]:
    """Return the two Kaggle files, failing if either is missing."""
    daily_path = raw_dir / DAILY_FILENAME
    summary_path = raw_dir / SUMMARY_FILENAME
    missing = [path for path in (daily_path, summary_path) if not path.is_file()]
    if missing:
        details = "\n  ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing raw COVID-19 file(s):\n  {details}")
    return daily_path, summary_path


def select_paper_population(
    daily_path: Path,
    summary_path: Path,
) -> tuple[pd.DataFrame, list[str]]:
    """Select the ten most populous countries from the complete daily table."""
    summary = pd.read_csv(
        summary_path,
        usecols=[CLIENT_COLUMN, POPULATION_COLUMN],
    )
    if summary[CLIENT_COLUMN].duplicated().any():
        raise ValueError("The summary table contains duplicate country rows.")

    summary[POPULATION_COLUMN] = pd.to_numeric(
        summary[POPULATION_COLUMN],
        errors="coerce",
    )
    top_countries = (
        summary.sort_values(POPULATION_COLUMN, ascending=False)
        [CLIENT_COLUMN]
        .head(N_CLIENTS)
        .tolist()
    )

    if len(top_countries) != N_CLIENTS or set(top_countries) != EXPECTED_COUNTRIES:
        raise ValueError(
            "The Kaggle summary does not produce the paper's ten countries. "
            f"Found: {top_countries}"
        )

    daily = pd.read_csv(daily_path, usecols=OUTPUT_COLUMNS)
    selected = daily[daily[CLIENT_COLUMN].isin(top_countries)].copy()
    selected = selected[OUTPUT_COLUMNS]

    if selected.duplicated([CLIENT_COLUMN, DATE_COLUMN]).any():
        raise ValueError("The daily table contains duplicate country/date rows")
    if len(selected) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS:,} rows for the paper population, "
            f"found {len(selected):,}. Check the Kaggle dataset version."
        )
    return selected, top_countries


def split_by_time_ratio(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reproduce the notebook's global chronological 80/20 split."""
    ordered = data.copy()
    ordered[DATE_COLUMN] = pd.to_datetime(
        ordered[DATE_COLUMN],
        format="%Y-%m-%d",
        errors="raise",
    )
    ordered = ordered.sort_values(DATE_COLUMN)

    split_index = int(len(ordered) * TRAIN_RATIO)
    train = ordered.iloc[:split_index].copy()
    test = ordered.iloc[split_index:].copy()

    if len(train) != EXPECTED_TRAIN_ROWS or len(test) != EXPECTED_TEST_ROWS:
        raise ValueError(
            "Unexpected split sizes: "
            f"train={len(train):,}, test={len(test):,}."
        )
    return train, test


def write_outputs(
    train: pd.DataFrame,
    test: pd.DataFrame,
    out_dir: Path,
) -> tuple[Path, Path]:
    """Write the CSV schema loaded by the experiments."""
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train_df.csv"
    test_path = out_dir / "test_df.csv"
    train.to_csv(train_path, index=False, date_format="%Y-%m-%d")
    test.to_csv(test_path, index=False, date_format="%Y-%m-%d")
    return train_path, test_path


def main() -> None:
    args = parse_args()
    daily_path, summary_path = find_input_files(args.raw_dir)
    selected, top_countries = select_paper_population(daily_path, summary_path)
    train, test = split_by_time_ratio(selected)
    train_path, test_path = write_outputs(train, test, args.out_dir)

    print(f"Countries: {', '.join(top_countries)}")
    print(f"Rows: {len(selected):,}")
    print(f"Train: {train_path} ({len(train):,} rows)")
    print(f"Test: {test_path} ({len(test):,} rows)")


if __name__ == "__main__":
    main()
