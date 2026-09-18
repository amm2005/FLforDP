#!/usr/bin/env python3
"""Build the Expedia used in the paper experiments.

Download the original, unfiltered Expedia Hotel Recommendations competition
data from Kaggle:
https://www.kaggle.com/competitions/expedia-hotel-recommendations

The input directory must contain the complete ``train.csv`` and
``destinations.csv`` files. The competition ``test.csv`` is not used.

With the default settings, the script:
  1. reduces the 149 destination features to 10 PCA components;
  2. selects the 3,000,000 earliest training events by ``date_time``;
  3. keeps ``site_name`` clients with 6,000 to 100,000 events;
  4. builds the 34 model features; and
  5. writes the parquet loaded by the paper experiments.

``date_time`` is retained for the per-client temporal split performed by
``ExpediaDataset`` and is excluded from model features there.

Run from the repository root:
  python scripts/prepare_expedia.py

The defaults read and write the paper paths under
``data/expedia_hotel_recommendations``. Use ``--data-dir`` and ``--out`` only
for an explicitly different local variant.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

try:
    import pyarrow as pa
    import pyarrow.csv as pacsv
    _HAVE_PYARROW = True
except Exception:  # pragma: no cover
    _HAVE_PYARROW = False

CLIENT_COL = "site_name"
TARGET_COL = "hotel_cluster"
TIME_COL = "date_time"

SAMPLE_SIZE = 3_000_000
MIN_ROWS = 6_000
MAX_ROWS = 100_000
CHUNK_SIZE = 500_000
PCA_COMPONENTS = 10
RANDOM_STATE = 42
PAPER_CLIENTS = 12
PAPER_CLASSES = 100

# The exact feature set used in the paper experiments.
FEATURES_FINAL = [
    # Search and market
    "hotel_market",
    "hotel_country",
    "hotel_continent",
    "srch_destination_id",
    "srch_destination_type_id",

    # User context
    "posa_continent",
    "user_location_country",
    "user_location_region",
    "user_location_city",

    # Session
    "is_booking",
    "cnt",
    "channel",
    "is_package",

    # Trip
    "srch_adults_cnt",
    "srch_children_cnt",
    "srch_rm_cnt",
    "group_size",
    "occupancy",
    "stay_duration",
    "days_until_checkin",

    # Destination PCA
    "dest_pca_0",
    "dest_pca_1",
    "dest_pca_2",
    "dest_pca_3",
    "dest_pca_4",
    "dest_pca_5",
    "dest_pca_6",
    "dest_pca_7",
    "dest_pca_8",
    "dest_pca_9",

    # Distance
    "orig_destination_distance",

    # Geography
    "same_country",
    "same_continent",
    "international_trip",
]


def build_destination_pca(data_dir):
    """PCA(10) on destinations.csv latent features → dest_pca_0..9 + key."""
    data_dir = Path(data_dir)
    destinations_path = data_dir / "destinations.csv"
    train_path = data_dir / "train.csv"
    missing = [
        str(path)
        for path in (train_path, destinations_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing raw Expedia file(s):\n  " + "\n  ".join(missing)
        )

    destinations = pd.read_csv(destinations_path)
    d_cols = [c for c in destinations.columns if c.startswith("d")]
    if "srch_destination_id" not in destinations.columns:
        raise ValueError("destinations.csv is missing srch_destination_id")
    if len(d_cols) < PCA_COMPONENTS:
        raise ValueError(
            f"destinations.csv has {len(d_cols)} destination features; "
            f"at least {PCA_COMPONENTS} are required"
        )
    pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE)
    dest_pca = pca.fit_transform(destinations[d_cols].fillna(0))
    pca_cols = [f"dest_pca_{i}" for i in range(PCA_COMPONENTS)]
    dest_small = pd.DataFrame(dest_pca, columns=pca_cols)
    dest_small["srch_destination_id"] = destinations["srch_destination_id"]
    print(
        f"[prep] PCA {len(d_cols)} -> {PCA_COMPONENTS}; explained variance "
        f"{pca.explained_variance_ratio_.sum():.4f}",
        flush=True,
    )
    return dest_small, pca_cols


def _read_date_time_epoch(train_csv):
    """Read the full timestamp column as epoch seconds.

    Reading one column keeps memory use bounded for the full Kaggle train file.
    PyArrow is used when available, with pandas as the fallback.
    """
    if _HAVE_PYARROW:
        conv = pacsv.ConvertOptions(
            include_columns=["date_time"],
            column_types={"date_time": pa.timestamp("s")},
        )
        tbl = pacsv.read_csv(train_csv, convert_options=conv)
        ts64 = tbl.column("date_time").to_numpy(zero_copy_only=False)
        del tbl
    else:
        ts64 = pd.read_csv(
            train_csv, usecols=["date_time"], parse_dates=["date_time"]
        )["date_time"].values
    if np.isnat(ts64).any():
        raise ValueError("unparseable/NaT date_time present in train.csv")
    return ts64.astype("datetime64[s]").astype("int64")


def _earliest_keep_mask(ts, n):
    """Return a deterministic mask for the earliest ``n`` timestamps.

    Ties at the boundary are resolved by original CSV row order.
    """
    m = ts.shape[0]
    if n >= m:
        return np.ones(m, dtype=bool)
    part = np.argpartition(ts, n - 1)
    thr = ts[part[:n]].max()
    strictly = np.flatnonzero(ts < thr)          # kept unconditionally
    need = n - strictly.size
    tie_band = np.flatnonzero(ts == thr)         # ascending pos = original order
    keep_pos = np.concatenate([strictly, tie_band[:need]])
    keep_mask = np.zeros(m, dtype=bool)
    keep_mask[keep_pos] = True
    assert keep_mask.sum() == n, keep_mask.sum()
    return keep_mask


def load_sample(data_dir, dest_small, sample_size=None):
    """Select the earliest events and merge destination PCA features.

    The source file is not globally time-sorted, so the first pass finds the
    selected row positions and the second pass reads the full records in chunks.
    """
    if sample_size is None:
        sample_size = SAMPLE_SIZE
    train_csv = str(Path(data_dir) / "train.csv")

    # Find the earliest rows using only the timestamp column.
    ts = _read_date_time_epoch(train_csv)
    M = ts.shape[0]
    print(f"[prep] pass1: {M:,} timestamps (~{ts.nbytes / 1e6:.0f} MB)", flush=True)
    keep_mask = _earliest_keep_mask(ts, min(sample_size, M))
    n_keep = int(keep_mask.sum())
    del ts

    # Re-read the CSV in chunks and retain the selected records.
    off, parts = 0, []
    for chunk in pd.read_csv(train_csv, chunksize=CHUNK_SIZE):
        nrow = len(chunk)
        m = keep_mask[off:off + nrow]
        if m.any():
            parts.append(chunk.loc[m])
        off += nrow
    assert off == M, f"row-count mismatch: pass1={M} pass2={off}"
    df = pd.concat(parts, ignore_index=True)
    assert len(df) == n_keep, f"expected {n_keep} rows, got {len(df)}"

    df = df.merge(dest_small, on="srch_destination_id", how="left")
    print(
        f"[prep] earliest {len(df):,} rows; time window "
        f"{df[TIME_COL].min()} .. {df[TIME_COL].max()}; "
        f"shape after merge: {df.shape}",
        flush=True,
    )
    return df


def filter_clients_by_size(df, min_rows=None, max_rows=None, client_col=CLIENT_COL):
    """Keep clients within the inclusive size range in the selected window."""
    if min_rows is None and max_rows is None:
        return df
    counts = df[client_col].value_counts()
    keep = pd.Series(True, index=counts.index)
    if min_rows is not None:
        keep &= counts >= min_rows
    if max_rows is not None:
        keep &= counts <= max_rows
    keep_clients = counts.index[keep.values]
    out = df[df[client_col].isin(keep_clients)].reset_index(drop=True)
    print(
        f"[prep] client-size filter [{min_rows}, {max_rows}]: "
        f"{counts.size} -> {len(keep_clients)} clients, "
        f"{len(df):,} -> {len(out):,} rows",
        flush=True,
    )
    if len(out) == 0:
        raise ValueError(
            f"client-size filter [{min_rows}, {max_rows}] removed ALL rows "
            f"(client sizes in window: {int(counts.min())}..{int(counts.max())}). "
            f"Loosen the bounds or change SAMPLE_SIZE."
        )
    return out


def feature_engineering(df, pca_cols):
    """Build the engineered columns used by the experiment loader."""
    # Keep date_time as the temporal split key.
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df["srch_ci"] = pd.to_datetime(df["srch_ci"], errors="coerce")
    df["srch_co"] = pd.to_datetime(df["srch_co"], errors="coerce")

    # Explicit floats prevent durations from being treated as categorical.
    df["stay_duration"] = (
        (df["srch_co"] - df["srch_ci"]).dt.days.fillna(-1).astype("float64")
    )
    df["days_until_checkin"] = (
        (df["srch_ci"] - df[TIME_COL]).dt.days.fillna(-1).astype("float64")
    )

    df["group_size"] = df["srch_adults_cnt"] + df["srch_children_cnt"]
    df["occupancy"] = df["group_size"] / np.maximum(df["srch_rm_cnt"], 1)

    df["orig_destination_distance"] = df["orig_destination_distance"].fillna(0)

    df["same_country"] = (
        df["user_location_country"] == df["hotel_country"]
    ).astype(int)
    df["same_continent"] = (
        df["posa_continent"] == df["hotel_continent"]
    ).astype(int)
    df["international_trip"] = (df["same_country"] == 0).astype(int)

    # Missing IDs have no destination embedding.
    for col in pca_cols:
        df[col] = df[col].fillna(0)

    return df


def build_final(df):
    """Select the experiment schema and normalize identifier dtypes."""
    keep = FEATURES_FINAL + [TARGET_COL, CLIENT_COL, TIME_COL]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns after feature build: {missing}")
    out = df[keep].copy()
    out[TARGET_COL] = out[TARGET_COL].astype("int64")
    out[CLIENT_COL] = out[CLIENT_COL].astype("int64")
    return out


def validate_output(out, paper_defaults):
    """Validate schema, labels, clients, and the default paper population."""
    expected = FEATURES_FINAL + [TARGET_COL, CLIENT_COL, TIME_COL]
    if list(out.columns) != expected:
        raise ValueError(
            f"Unexpected Expedia output columns: {list(out.columns)}"
        )
    if out.empty:
        raise ValueError("Expedia preparation produced no rows")
    if out[[TARGET_COL, CLIENT_COL, TIME_COL]].isna().any().any():
        raise ValueError("Expedia target, client, and time columns must be non-null")
    if not out[TARGET_COL].between(0, PAPER_CLASSES - 1).all():
        raise ValueError("hotel_cluster labels must be in [0, 99]")
    if paper_defaults:
        n_clients = out[CLIENT_COL].nunique()
        n_classes = out[TARGET_COL].nunique()
        if n_clients != PAPER_CLIENTS or n_classes != PAPER_CLASSES:
            raise ValueError(
                "Default preparation does not reproduce the paper population: "
                f"clients={n_clients} (expected {PAPER_CLIENTS}), "
                f"classes={n_classes} (expected {PAPER_CLASSES})"
            )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data-dir",
        default="data/expedia_hotel_recommendations",
        help="Directory with the complete Kaggle train.csv and destinations.csv.",
    )
    ap.add_argument(
        "--out",
        default=(
            "data/expedia_hotel_recommendations/"
            "expedia_6000_100000.parquet"
        ),
        help="Output parquet loaded by the paper experiments.",
    )
    ap.add_argument(
        "--sample-size", type=int, default=SAMPLE_SIZE,
        help="earliest N rows by date_time (default: module SAMPLE_SIZE)",
    )
    ap.add_argument(
        "--min-rows", type=int, default=MIN_ROWS,
        help="minimum client size after earliest-window selection",
    )
    ap.add_argument(
        "--max-rows", type=int, default=MAX_ROWS,
        help="maximum client size after earliest-window selection",
    )
    args = ap.parse_args()
    if args.sample_size <= 0:
        ap.error("--sample-size must be positive")
    if args.min_rows <= 0:
        ap.error("--min-rows must be positive")
    if args.max_rows < args.min_rows:
        ap.error("--max-rows must be greater than or equal to --min-rows")

    dest_small, pca_cols = build_destination_pca(args.data_dir)
    df = load_sample(args.data_dir, dest_small, sample_size=args.sample_size)
    df = filter_clients_by_size(df, args.min_rows, args.max_rows)
    df = feature_engineering(df, pca_cols)
    out = build_final(df)
    paper_defaults = (
        args.sample_size == SAMPLE_SIZE
        and args.min_rows == MIN_ROWS
        and args.max_rows == MAX_ROWS
    )
    validate_output(out, paper_defaults)

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)

    print("=" * 60, flush=True)
    print(f"[prep] wrote {output_path}", flush=True)
    print(f"[prep] shape          : {out.shape}", flush=True)
    print(f"[prep] clients (site) : {out[CLIENT_COL].nunique()}", flush=True)
    print(f"[prep] classes        : {out[TARGET_COL].nunique()} "
          f"(min={out[TARGET_COL].min()}, max={out[TARGET_COL].max()})", flush=True)
    print(f"[prep] time range     : {out[TIME_COL].min()} .. {out[TIME_COL].max()}", flush=True)
    print(f"[prep] dtypes:\n{out.dtypes}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
