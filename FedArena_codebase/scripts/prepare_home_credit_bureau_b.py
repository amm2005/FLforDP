#!/usr/bin/env python3
"""Build the experiment-ready Home Credit Bureau-B parquet."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl


ROOT = Path(__file__).resolve().parents[1]

CLIENT_COL = "credor_3940957M"
PLACEHOLDER_CLIENT = "a55475b1"
MONTH_DAYS = 30.44
RETENTION_START = (2016, 2, 1)
HORIZON_MONTHS = 6
DPD_THRESHOLD = 30
DEFAULT_TOP_N_CLIENTS = 5

RAW_FILES = (
    "train_credit_bureau_b_1.parquet",
    "train_credit_bureau_b_2.parquet",
    "train_person_1.parquet",
    "train_static_cb_0.parquet",
    "train_static_0_0.parquet",
    "train_static_0_1.parquet",
    "train_base.parquet",
)

CAT_LAST = [
    "contracttype_653M",
    "purposeofcred_722M",
    "pmtmethod_731M",
    "subjectrole_326M",
    "subjectrole_43M",
    "periodicityofpmts_997M",
]
NUM_LAST = [
    "amount_1115A",
    "credlmt_3940954A",
    "installmentamount_833A",
    "installmentamount_644A",
    "instlamount_892A",
    "numberofinstls_810L",
]
CAT_PORTRAIT = [
    "sex_738L",
    "incometype_1044T",
    "language1_981M",
    "empladdr_district_926M",
    "maritalst_385M",
    "education_88M",
]
NUM_PORTRAIT = ["maininc_215A", "age"]

CAT_HIST = [f"hist_{column}" for column in CAT_LAST]
NUM_HIST = [f"hist_{column}" for column in NUM_LAST] + [
    "gap_days",
    "hist_planned_term_days",
    "hist_dpd_at_T",
    "hist_ov_at_T",
    "hist_had_dpd30_at_T",
    "hist_months_since_dpd",
    "hist_n_pmts_before_T",
]
EXTRA_NUM = ["n_prior_same_credor", "new_to_bank"]

CAT_FEATURES = CAT_LAST + CAT_PORTRAIT + CAT_HIST
NUM_FEATURES = NUM_LAST + NUM_PORTRAIT + NUM_HIST + EXTRA_NUM
FEATURES = CAT_FEATURES + NUM_FEATURES
BOOKKEEPING_COLUMNS = [CLIENT_COL, "split", "target"]
FORBIDDEN_OUTPUT_COLUMNS = {
    "case_id",
    "last_date",
    "date_decision",
    "contractdate_551D",
    "contractmaturitydate_151D",
    "dpdmax_851P",
    "pmts_date_1107D",
    "pmts_dpdvalue_108P",
    "pmts_pmtsoverdue_635A",
}

B1_READ = [
    "case_id",
    "num_group1",
    CLIENT_COL,
    "contractdate_551D",
    "contractmaturitydate_151D",
    "dpdmax_851P",
    *CAT_LAST,
    *NUM_LAST,
]
B2_READ = [
    "case_id",
    "num_group1",
    "pmts_date_1107D",
    "pmts_dpdvalue_108P",
    "pmts_pmtsoverdue_635A",
]
PERSON_READ = ["case_id", "num_group1", *CAT_PORTRAIT[:4]]
STATIC_CB_READ = [
    "case_id",
    "dateofbirth_337D",
    "maritalst_385M",
    "education_88M",
]
STATIC0_READ = ["case_id", "maininc_215A"]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=ROOT / "data" / "home_credit_bureau_b" / "raw_filtered",
        help="directory containing the seven filtered Kaggle parquet files",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output parquet; by default it is written to data/home_credit_bureau_b/",
    )
    parser.add_argument(
        "--top-n-clients",
        type=int,
        default=DEFAULT_TOP_N_CLIENTS,
        help="number of largest creditors to keep; use 9 for the auxiliary paper run",
    )
    args = parser.parse_args(argv)
    if args.top_n_clients <= 0:
        parser.error("--top-n-clients must be positive")
    return args


def validate_raw_files(raw_dir: Path) -> None:
    """Fail before processing if an exported Kaggle table is missing."""
    missing = [str(raw_dir / name) for name in RAW_FILES if not (raw_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing filtered Home Credit file(s):\n  " + "\n  ".join(missing)
        )


def load_raw_tables(raw_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load only columns needed to build the benchmark."""
    validate_raw_files(raw_dir)

    bureau = pl.read_parquet(
        raw_dir / "train_credit_bureau_b_1.parquet",
        columns=B1_READ,
    ).with_columns(
        pl.col("contractdate_551D").cast(pl.Date, strict=False),
        pl.col("contractmaturitydate_151D").cast(pl.Date, strict=False),
    )
    base = pl.read_parquet(
        raw_dir / "train_base.parquet",
        columns=["case_id", "date_decision"],
    ).with_columns(
        pl.col("date_decision").cast(pl.Date, strict=False),
    )
    bureau = bureau.join(
        base,
        on="case_id",
        how="left",
        maintain_order="left",
    )

    payments = pl.read_parquet(
        raw_dir / "train_credit_bureau_b_2.parquet",
        columns=B2_READ,
    ).with_columns(
        pl.col("pmts_date_1107D").cast(pl.Date, strict=False),
    )
    return bureau, payments


def build_contract_frames(
    bureau: pl.DataFrame,
    payments: pl.DataFrame,
) -> dict[str, pl.DataFrame]:
    """Select prediction units and their previous same-creditor contracts."""
    payment_span = payments.group_by("case_id", "num_group1").agg(
        pl.col("pmts_date_1107D").min().alias("pmt_min"),
        pl.col("pmts_date_1107D").max().alias("pmt_max"),
        pl.len().alias("n_pmts"),
    )
    dated = bureau.filter(pl.col("contractdate_551D").is_not_null())

    units = (
        dated.sort(["case_id", "contractdate_551D", "num_group1"])
        .group_by("case_id", maintain_order=True)
        .last()
        .filter(pl.col("dpdmax_851P").is_not_null())
    )
    unit_keys = units.select(
        "case_id",
        pl.col("contractdate_551D").alias("last_date"),
        pl.col(CLIENT_COL).alias("last_credor"),
    )
    earlier = (
        dated.join(unit_keys, on="case_id", how="inner")
        .filter(pl.col("contractdate_551D") < pl.col("last_date"))
    )
    same_creditor = earlier.filter(pl.col(CLIENT_COL) == pl.col("last_credor"))
    history = (
        same_creditor.sort(["case_id", "contractdate_551D", "num_group1"])
        .group_by("case_id", maintain_order=True)
        .last()
    )
    history_counts = (
        same_creditor.group_by("case_id")
        .len()
        .rename({"len": "n_prior_same_credor"})
    )
    return {
        "units": units,
        "payment_span": payment_span,
        "history": history,
        "history_counts": history_counts,
    }


def build_population(
    frames: dict[str, pl.DataFrame],
    payments: pl.DataFrame,
) -> pl.DataFrame:
    """Build the observable natural population and its six-month target."""
    population = (
        frames["units"]
        .join(
            frames["payment_span"],
            on=["case_id", "num_group1"],
            how="left",
            maintain_order="left",
        )
        .with_columns(
            (pl.col("date_decision") - pl.col("contractdate_551D"))
            .dt.total_days()
            .alias("age_d")
        )
        .filter(
            (pl.col("contractdate_551D") >= pl.date(*RETENTION_START))
            & (pl.col("age_d") >= HORIZON_MONTHS * MONTH_DAYS)
            & pl.col("n_pmts").is_not_null()
            & (
                (pl.col("pmt_min") - pl.col("contractdate_551D")).dt.total_days()
                <= 31
            )
            & (pl.col(CLIENT_COL) != PLACEHOLDER_CLIENT)
        )
    )

    labels = (
        payments.join(
            population.select("case_id", "num_group1", "contractdate_551D"),
            on=["case_id", "num_group1"],
            how="inner",
        )
        .filter(
            (pl.col("pmts_date_1107D") - pl.col("contractdate_551D"))
            .dt.total_days()
            <= HORIZON_MONTHS * MONTH_DAYS
        )
        .group_by("case_id")
        .agg(
            (pl.col("pmts_dpdvalue_108P").max() > DPD_THRESHOLD).alias("target")
        )
    )
    population = (
        population.join(labels, on="case_id", how="left", maintain_order="left")
        .with_columns(pl.col("target").fill_null(False))
    )
    print(
        f"eligible population: {population.height:,} rows, "
        f"positive rate {population['target'].mean():.3f}"
    )
    return population


def build_features(
    frames: dict[str, pl.DataFrame],
    population: pl.DataFrame,
    payments: pl.DataFrame,
    raw_dir: Path,
) -> pd.DataFrame:
    """Build origination, borrower, and prior same-creditor features."""
    history = frames["history"]
    history_payments = (
        payments.join(
            history.select("case_id", "num_group1", "last_date"),
            on=["case_id", "num_group1"],
            how="inner",
        )
        .filter(pl.col("pmts_date_1107D") <= pl.col("last_date"))
        .group_by("case_id")
        .agg(
            pl.col("pmts_dpdvalue_108P").max().alias("hist_dpd_at_T"),
            pl.col("pmts_pmtsoverdue_635A").max().alias("hist_ov_at_T"),
            pl.len().alias("hist_n_pmts_before_T"),
            pl.col("pmts_date_1107D")
            .filter(pl.col("pmts_dpdvalue_108P") > DPD_THRESHOLD)
            .max()
            .alias("last_bad_date"),
        )
    )
    history_features = (
        history.select(
            "case_id",
            pl.col("contractdate_551D").alias("hist_contractdate"),
            pl.col("contractmaturitydate_151D").alias("hist_maturity"),
            "last_date",
            *[pl.col(column).alias(f"hist_{column}") for column in CAT_LAST + NUM_LAST],
        )
        .join(history_payments, on="case_id", how="left", maintain_order="left")
        .with_columns(
            (pl.col("last_date") - pl.col("hist_contractdate"))
            .dt.total_days()
            .alias("gap_days"),
            (pl.col("hist_maturity") - pl.col("hist_contractdate"))
            .dt.total_days()
            .alias("hist_planned_term_days"),
            (pl.col("hist_dpd_at_T") > DPD_THRESHOLD)
            .cast(pl.Int8)
            .alias("hist_had_dpd30_at_T"),
            (
                (pl.col("last_date") - pl.col("last_bad_date")).dt.total_days()
                / MONTH_DAYS
            ).alias("hist_months_since_dpd"),
        )
        .drop("hist_contractdate", "hist_maturity", "last_date", "last_bad_date")
    )

    person = (
        pl.read_parquet(raw_dir / "train_person_1.parquet", columns=PERSON_READ)
        .filter(pl.col("num_group1") == 0)
        .drop("num_group1")
        .unique(subset="case_id")
    )
    static_cb = (
        pl.read_parquet(
            raw_dir / "train_static_cb_0.parquet",
            columns=STATIC_CB_READ,
        )
        .with_columns(pl.col("dateofbirth_337D").cast(pl.Date, strict=False))
        .unique(subset="case_id")
    )
    static_0 = pl.concat(
        [
            pl.read_parquet(
                raw_dir / f"train_static_0_{shard}.parquet",
                columns=STATIC0_READ,
            )
            for shard in (0, 1)
        ]
    ).unique(subset="case_id")

    table = (
        population.select(
            [
                "case_id",
                "date_decision",
                "target",
                pl.col(CLIENT_COL).alias("last_credor"),
                pl.col("contractdate_551D").alias("last_date"),
                *CAT_LAST,
                *NUM_LAST,
            ]
        )
        .join(
            frames["history_counts"],
            on="case_id",
            how="left",
            maintain_order="left",
        )
        .join(history_features, on="case_id", how="left", maintain_order="left")
        .join(person, on="case_id", how="left", maintain_order="left")
        .join(static_cb, on="case_id", how="left", maintain_order="left")
        .join(static_0, on="case_id", how="left", maintain_order="left")
        .with_columns(
            pl.col("n_prior_same_credor").fill_null(0),
            (pl.col("n_prior_same_credor").fill_null(0) == 0)
            .cast(pl.Int8)
            .alias("new_to_bank"),
            (
                (pl.col("date_decision") - pl.col("dateofbirth_337D"))
                .dt.total_days()
                / 365.25
            ).alias("age"),
        )
    )
    with_history = table.filter(pl.col("new_to_bank") == 0).height
    print(
        f"previous same-creditor history: {with_history:,} rows "
        f"({with_history / table.height:.1%})"
    )

    result = table.select(
        ["case_id", "last_credor", "target", "last_date", *FEATURES]
    ).to_pandas()
    result["target"] = result["target"].astype(int)
    return result


def split_time(
    client_rows: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split one creditor chronologically, breaking date ties by case_id."""
    client_rows = client_rows.sort_values(
        ["last_date", "case_id"]
    ).reset_index(drop=True)
    n_rows = len(client_rows)
    n_test = int(round(n_rows * (1 - train_frac - val_frac)))
    n_val = int(round(n_rows * val_frac))
    n_train = n_rows - n_val - n_test
    return (
        client_rows.iloc[:n_train],
        client_rows.iloc[n_train : n_train + n_val],
        client_rows.iloc[n_train + n_val :],
    )


def select_clients_and_bake_split(
    table: pd.DataFrame,
    top_n_clients: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep the largest creditors and add the offline temporal split."""
    clients = list(table["last_credor"].value_counts().head(top_n_clients).index)
    if len(clients) != top_n_clients:
        raise ValueError(
            f"Expected {top_n_clients} eligible creditors, found {len(clients)}"
        )
    table = table[table["last_credor"].isin(clients)]

    parts = []
    stats = []
    for client in clients:
        train, val, test = split_time(table[table["last_credor"] == client])
        for rows, split in ((train, "train"), (val, "val"), (test, "test")):
            rows = rows.copy()
            rows["split"] = split
            parts.append(rows)
        stats.append(
            {
                "credor": client,
                "n_train": len(train),
                "n_val": len(val),
                "n_test": len(test),
                "pos_train": int(train["target"].sum()),
                "pos_val": int(val["target"].sum()),
                "pos_test": int(test["target"].sum()),
            }
        )

    combined = pd.concat(parts, ignore_index=True)
    summary = pd.DataFrame(stats).set_index("credor")
    return combined, summary


def assemble_output(table: pd.DataFrame) -> pd.DataFrame:
    """Keep only experiment bookkeeping, target, and allowlisted features."""
    if table.empty:
        raise ValueError("No Home Credit rows remain after client selection")
    if table["case_id"].duplicated().any():
        raise ValueError("A case_id appears more than once in the prepared population")

    output = pd.DataFrame()
    output[CLIENT_COL] = table["last_credor"].astype(str)
    output["split"] = table["split"]
    output["target"] = table["target"].astype("int8")

    for column in CAT_FEATURES:
        values = table[column].astype(object)
        output[column] = values.where(pd.notna(values), np.nan)

    for column in NUM_FEATURES:
        if table[column].isna().any():
            output[f"{column}_isnull"] = table[column].isna().astype("int8")
        output[column] = table[column].astype("float32")

    flags = [column for column in output if column.endswith("_isnull")]
    expected = {CLIENT_COL, "split", "target", *FEATURES, *flags}
    assert set(output.columns) == expected
    assert all(flag.removesuffix("_isnull") in NUM_FEATURES for flag in flags)
    assert "hist_months_since_dpd_isnull" in output.columns
    return output


def validate_output(output: pd.DataFrame, top_n_clients: int) -> None:
    """Check the public artifact contract before replacing an existing file."""
    actual_clients = output[CLIENT_COL].dropna().nunique()
    if actual_clients != top_n_clients:
        raise ValueError(
            f"Expected {top_n_clients} creditors in output, found {actual_clients}"
        )
    if set(output["split"].unique()) != {"train", "val", "test"}:
        raise ValueError("Output must contain train, val, and test rows")
    if not output["target"].isin([0, 1]).all():
        raise ValueError("Target must be binary")
    incomplete = [
        client
        for client, frame in output.groupby(CLIENT_COL)
        if set(frame["split"].unique()) != {"train", "val", "test"}
    ]
    if incomplete:
        raise ValueError(f"Creditors without all three splits: {incomplete}")
    leaked = set(output.columns) & FORBIDDEN_OUTPUT_COLUMNS
    if leaked:
        raise ValueError(f"Identifier, date, or label columns leaked: {sorted(leaked)}")


def print_summary(output: pd.DataFrame, split_summary: pd.DataFrame) -> None:
    """Print the artifact counts needed for a quick reproducibility check."""
    print("\nrows and positives by client:")
    print(split_summary.to_string())
    print(
        f"\nartifact: {len(output):,} rows, {len(output.columns)} columns, "
        f"{output[CLIENT_COL].nunique()} clients, "
        f"positive rate {output['target'].mean():.3f}"
    )


def main(argv=None) -> None:
    args = parse_args(argv)
    print(
        f"Home Credit Bureau-B: natural, h{HORIZON_MONTHS}, "
        f"top-{args.top_n_clients}, threshold {DPD_THRESHOLD}"
    )

    bureau, payments = load_raw_tables(args.raw_dir)
    frames = build_contract_frames(bureau, payments)
    population = build_population(frames, payments)
    features = build_features(frames, population, payments, args.raw_dir)
    selected, split_summary = select_clients_and_bake_split(
        features,
        args.top_n_clients,
    )
    output = assemble_output(selected)
    validate_output(output, args.top_n_clients)

    suffix = "" if args.top_n_clients == DEFAULT_TOP_N_CLIENTS else (
        f"_top{args.top_n_clients}"
    )
    output_path = args.out or (
        ROOT
        / "data"
        / "home_credit_bureau_b"
        / f"bureau_b_h{HORIZON_MONTHS}_natural{suffix}.parquet"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(output_path, index=False)

    print_summary(output, split_summary)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
