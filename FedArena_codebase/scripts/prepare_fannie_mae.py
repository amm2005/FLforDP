"""Build the experiment-ready Fannie Mae benchmark parquet from raw files."""

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]

QUARTERS = ("2012Q1", "2012Q2", "2012Q3")
DELINQUENCY_DEPTH = 90
HORIZON_MONTHS = 84
TOP_N_BEFORE_EXCLUSION = 10
EXCLUDED_SELLER = "Wells Fargo Bank, N.A."
RANDOM_SEED = 42
DEFAULT_CHUNK_SIZE = 1_000_000

BAD_ZERO_BALANCE_CODES = {"02", "03", "09", "15", "97", "98"}

CAT_FEATURES = [
    "channel",
    "first_time",
    "purpose",
    "prop_type",
    "occupancy",
    "state",
    "msa",
    "zip3",
    "mi_type",
    "special_program",
    "valuation_method",
    "high_balance",
]

NUM_FEATURES = [
    "orig_rate",
    "orig_upb",
    "orig_term",
    "ltv",
    "cltv",
    "dti",
    "fico",
    "fico_co",
    "n_borrowers",
    "n_units",
    "mi_pct",
]

# One-based positions from the Fannie Mae loan-performance file glossary.
RAW_POSITIONS = {
    "loan_id": 2,
    "channel": 4,
    "seller": 5,
    "orig_rate": 8,
    "orig_upb": 10,
    "orig_term": 13,
    "loan_age": 16,
    "ltv": 20,
    "cltv": 21,
    "n_borrowers": 22,
    "dti": 23,
    "fico": 24,
    "fico_co": 25,
    "first_time": 26,
    "purpose": 27,
    "prop_type": 28,
    "n_units": 29,
    "occupancy": 30,
    "state": 31,
    "msa": 32,
    "zip3": 33,
    "mi_pct": 34,
    "dq_status": 40,
    "zb_code": 44,
    "mi_type": 73,
    "special_program": 79,
    "valuation_method": 86,
    "high_balance": 87,
    "fico_fallback": 111,
}

RAW_USECOLS = sorted(position - 1 for position in RAW_POSITIONS.values())
RAW_COLUMN_NAMES = {position - 1: name for name, position in RAW_POSITIONS.items()}
ORIGINATION_COLUMNS = [
    "loan_id",
    "seller",
    *CAT_FEATURES,
    *NUM_FEATURES,
    "fico_fallback",
]
NUMERIC_SOURCE_COLUMNS = [*NUM_FEATURES, "fico_fallback"]

BOOKKEEPING_COLUMNS = ["seller", "split", "target"]
LABEL_INGREDIENTS = {
    "loan_id",
    "loan_age",
    "dq_status",
    "age_d90",
    "first_age",
    "zb_code",
    "zb_age",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=ROOT / "data" / "fannie_mae" / "raw",
        help="directory containing 2012Q1.csv, 2012Q2.csv, and 2012Q3.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output parquet; by default it is written to data/fannie_mae/",
    )
    parser.add_argument(
        "--val-mode",
        choices=("oot100", "oot15"),
        default="oot100",
        help="size of each later-quarter pool relative to the client train set",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="number of monthly records read from a raw file at once",
    )
    args = parser.parse_args()
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")
    return args


def find_raw_files(raw_dir: Path) -> dict[str, Path]:
    """Return the three quarterly files and fail before processing if one is missing."""
    paths = {quarter: raw_dir / f"{quarter}.csv" for quarter in QUARTERS}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing raw Fannie Mae file(s):\n  " + "\n  ".join(missing)
        )
    return paths


def iter_raw_chunks(path: Path, chunk_size: int):
    """Read only the raw columns needed to build the benchmark."""
    reader = pd.read_csv(
        path,
        sep="|",
        header=None,
        usecols=RAW_USECOLS,
        dtype=str,
        quoting=csv.QUOTE_NONE,
        na_values=["", "XX"],
        keep_default_na=False,
        on_bad_lines="warn",
        chunksize=chunk_size,
    )
    for chunk in reader:
        yield chunk.rename(columns=RAW_COLUMN_NAMES)


def collapse_quarter(path: Path, chunk_size: int) -> pd.DataFrame:
    """Collapse monthly records into one origination row and label history per loan."""
    origination_parts = []
    age_parts = []
    zero_balance_parts = []

    for chunk_index, chunk in enumerate(iter_raw_chunks(path, chunk_size), start=1):
        chunk["loan_age"] = pd.to_numeric(chunk["loan_age"], errors="coerce")
        chunk["dq_status"] = pd.to_numeric(chunk["dq_status"], errors="coerce")

        origination_parts.append(
            chunk.drop_duplicates("loan_id", keep="first")[ORIGINATION_COLUMNS]
        )

        grouped = chunk.groupby("loan_id", sort=False)
        ages = grouped["loan_age"].min().rename("first_age").to_frame()
        ages["age_d90"] = (
            chunk["loan_age"]
            .where(chunk["dq_status"] >= 3)
            .groupby(chunk["loan_id"], sort=False)
            .min()
        )
        age_parts.append(ages.reset_index())

        zero_balance = chunk.loc[
            chunk["zb_code"].notna(), ["loan_id", "zb_code", "loan_age"]
        ].rename(columns={"loan_age": "zb_age"})
        if not zero_balance.empty:
            zero_balance_parts.append(zero_balance)

        if chunk_index % 10 == 0:
            print(f"  processed {chunk_index * chunk_size:,} monthly records")

    if not origination_parts:
        raise ValueError(f"Raw file is empty: {path}")

    origination = pd.concat(origination_parts, ignore_index=True).drop_duplicates(
        "loan_id", keep="first"
    )
    ages = (
        pd.concat(age_parts, ignore_index=True)
        .groupby("loan_id", as_index=False)
        .agg(first_age=("first_age", "min"), age_d90=("age_d90", "min"))
    )

    if zero_balance_parts:
        zero_balance = pd.concat(zero_balance_parts, ignore_index=True).drop_duplicates(
            "loan_id", keep="last"
        )
    else:
        zero_balance = pd.DataFrame(columns=["loan_id", "zb_code", "zb_age"])

    loans = origination.merge(ages, on="loan_id", how="left").merge(
        zero_balance, on="loan_id", how="left"
    )

    for column in NUMERIC_SOURCE_COLUMNS:
        loans[column] = pd.to_numeric(loans[column], errors="coerce")
    loans["fico"] = loans["fico"].fillna(loans["fico_fallback"])
    return loans.drop(columns="fico_fallback")


def add_target(loans: pd.DataFrame) -> pd.DataFrame:
    """Keep loans observed from origination and add the benchmark target."""
    loans = loans.loc[loans["first_age"] <= 3].copy()
    bad_end = loans["zb_code"].isin(BAD_ZERO_BALANCE_CODES)
    loans["target"] = (
        (loans["age_d90"] <= HORIZON_MONTHS)
        | (bad_end & (loans["zb_age"] <= HORIZON_MONTHS))
    ).astype("int8")
    return loans


def select_paper_clients(
    train_quarter: pd.DataFrame,
    val_quarter: pd.DataFrame,
    test_quarter: pd.DataFrame,
) -> list[str]:
    """Select the benchmark sellers that are present in all three quarters."""
    train_sizes = train_quarter.loc[
        train_quarter["seller"] != "Other", "seller"
    ].value_counts()
    active = [
        seller
        for seller in train_sizes.index
        if (val_quarter["seller"] == seller).any()
        and (test_quarter["seller"] == seller).any()
    ]
    selected = active[:TOP_N_BEFORE_EXCLUSION]

    if len(selected) != TOP_N_BEFORE_EXCLUSION:
        raise ValueError(
            f"Expected {TOP_N_BEFORE_EXCLUSION} active sellers, found {len(selected)}"
        )
    if EXCLUDED_SELLER not in selected:
        raise ValueError(
            f"Expected excluded seller {EXCLUDED_SELLER!r} in the top "
            f"{TOP_N_BEFORE_EXCLUSION}"
        )

    clients = [seller for seller in selected if seller != EXCLUDED_SELLER]
    print(f"clients ({len(clients)}): {clients}")
    return clients


def select_features(
    train_quarter: pd.DataFrame, clients: list[str]
) -> tuple[list[str], list[str]]:
    """Drop allowlisted features that are constant in the training population."""
    train_rows = train_quarter[train_quarter["seller"].isin(clients)]
    categorical = [
        column for column in CAT_FEATURES if train_rows[column].nunique(dropna=True) > 1
    ]
    numerical = [
        column for column in NUM_FEATURES if train_rows[column].nunique(dropna=True) > 1
    ]
    dropped = [
        column
        for column in CAT_FEATURES + NUM_FEATURES
        if column not in categorical + numerical
    ]
    print(
        f"features: {len(categorical)} categorical, {len(numerical)} numerical; "
        f"constant columns dropped: {dropped}"
    )
    return categorical, numerical


def _safe_split(frame: pd.DataFrame, **kwargs):
    """Use a stratified split when the client has enough positive examples."""
    try:
        return train_test_split(
            frame,
            stratify=frame["target"],
            random_state=RANDOM_SEED,
            **kwargs,
        )
    except ValueError:
        return train_test_split(frame, random_state=RANDOM_SEED, **kwargs)


def _sample_pool(pool: pd.DataFrame, size: int) -> pd.DataFrame:
    """Sample a later-quarter client pool without exceeding its size."""
    size = min(int(size), len(pool))
    if size >= len(pool):
        return pool
    sample, _ = _safe_split(pool, train_size=size)
    return sample


def build_oot_splits(
    train_quarter: pd.DataFrame,
    val_quarter: pd.DataFrame,
    test_quarter: pd.DataFrame,
    clients: list[str],
    val_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the per-client Q1/Q2/Q3 out-of-time split."""
    pool_fraction = 1.0 if val_mode == "oot100" else 0.15
    parts = []
    records = []

    for seller in clients:
        seller_train = train_quarter[train_quarter["seller"] == seller]
        train, _ = _safe_split(seller_train, test_size=0.30)
        pool_size = len(train) * pool_fraction
        val = _sample_pool(val_quarter[val_quarter["seller"] == seller], pool_size)
        test = _sample_pool(test_quarter[test_quarter["seller"] == seller], pool_size)

        for frame, split in ((train, "train"), (val, "val"), (test, "test")):
            part = frame.copy()
            part["split"] = split
            parts.append(part)

        records.append(
            {
                "seller": seller,
                "n_train": len(train),
                "n_val": len(val),
                "n_test": len(test),
                "pos_train": int(train["target"].sum()),
                "pos_val": int(val["target"].sum()),
                "pos_test": int(test["target"].sum()),
            }
        )

    summary = pd.DataFrame(records).set_index("seller").sort_values("n_test")
    return pd.concat(parts, ignore_index=True), summary


def assemble_output(
    combined: pd.DataFrame,
    categorical: list[str],
    numerical: list[str],
) -> pd.DataFrame:
    """Keep only bookkeeping columns and allowlisted origination features."""
    if combined.empty:
        raise ValueError("No Fannie Mae rows remain after client selection and splitting")
    if combined["loan_id"].duplicated().any():
        raise ValueError("A loan_id appears more than once in the prepared population")

    output = combined[BOOKKEEPING_COLUMNS].copy()
    for column in categorical:
        values = combined[column].astype(object)
        output[column] = values.where(pd.notna(values), np.nan)
    for column in numerical:
        output[column] = combined[column].astype("float32")
    return output


def validate_output(
    output: pd.DataFrame,
    clients: list[str],
    categorical: list[str],
    numerical: list[str],
) -> None:
    """Check client, split, target, and feature invariants before writing."""
    expected_columns = BOOKKEEPING_COLUMNS + categorical + numerical
    if list(output.columns) != expected_columns:
        raise ValueError(
            f"Unexpected output columns: {list(output.columns)}; "
            f"expected {expected_columns}"
        )

    actual_clients = set(output["seller"].dropna().unique())
    if actual_clients != set(clients) or len(actual_clients) != 9:
        raise ValueError(f"Expected the nine benchmark clients, found {actual_clients}")
    if EXCLUDED_SELLER in actual_clients:
        raise ValueError(f"Excluded seller is present: {EXCLUDED_SELLER}")

    if set(output["split"].unique()) != {"train", "val", "test"}:
        raise ValueError("Output must contain train, val, and test rows")
    if not output["target"].isin([0, 1]).all():
        raise ValueError("Target must be binary")

    incomplete = [
        seller
        for seller, frame in output.groupby("seller")
        if set(frame["split"].unique()) != {"train", "val", "test"}
    ]
    if incomplete:
        raise ValueError(f"Clients without all three splits: {incomplete}")

    leaked = set(output.columns) & LABEL_INGREDIENTS
    if leaked:
        raise ValueError(f"Label or identifier columns leaked into output: {leaked}")


def default_output_path(val_mode: str) -> Path:
    """Return the standard artifact path for the selected validation mode."""
    filename = (
        f"fannie_mae_d{DELINQUENCY_DEPTH}_h{HORIZON_MONTHS}_"
        f"{val_mode}_ex-wells.parquet"
    )
    return ROOT / "data" / "fannie_mae" / filename


def main() -> None:
    args = parse_args()
    raw_files = find_raw_files(args.raw_dir)

    quarters = {}
    for quarter in QUARTERS:
        print(f"\n=== {quarter}: {raw_files[quarter]} ===")
        loans = collapse_quarter(raw_files[quarter], args.chunk_size)
        quarters[quarter] = add_target(loans)
        print(
            f"{len(quarters[quarter]):,} eligible loans; "
            f"target rate {quarters[quarter]['target'].mean():.3%}"
        )

    train_quarter, val_quarter, test_quarter = (
        quarters[quarter] for quarter in QUARTERS
    )
    clients = select_paper_clients(train_quarter, val_quarter, test_quarter)
    categorical, numerical = select_features(train_quarter, clients)
    combined, summary = build_oot_splits(
        train_quarter,
        val_quarter,
        test_quarter,
        clients,
        args.val_mode,
    )

    print("\n=== per-client split summary ===")
    print(summary.to_string())

    output = assemble_output(combined, categorical, numerical)
    validate_output(output, clients, categorical, numerical)

    output_path = args.out or default_output_path(args.val_mode)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(output_path, index=False)
    print(
        f"\nwrote {len(output):,} rows x {len(output.columns)} columns "
        f"to {output_path}"
    )


if __name__ == "__main__":
    main()
