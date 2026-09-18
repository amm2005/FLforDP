#!/usr/bin/env python3
import argparse
import os

import polars as pl
from datasets import load_dataset


HF_NAME = "McAuley-Lab/Amazon-Reviews-2023"

JOIN_COL = "parent_asin"
CLIENT_COL = "store"
TARGET_COL = "rating"
TEXT_COL = "text"

FRACTION = 0.01


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--category",
        default="Toys_and_Games",
        help="Category of items.",
    )
    ap.add_argument(
        "--out",
        default="data/amazon/amazon_processed.parquet",
        help="Output parquet path.",
    )
    ap.add_argument(
        "--fraction", type=float, default=FRACTION,
        help="Fraction of full data",
    )
    ap.add_argument(
        "--max_clients", type=int, default=10,
        help="maximum number of clients",
    )
    ap.add_argument(
        "--min_samples_per_client", type=int, default=700,
        help="minumum number of samples per client",
    )
    args = ap.parse_args()


    reviews = load_dataset(HF_NAME, f"raw_review_{args.category}", split="full", trust_remote_code=True)
    reviews_df = pl.DataFrame({
        JOIN_COL: reviews[JOIN_COL],
        TARGET_COL: [int(lbl) for lbl in reviews[TARGET_COL]],
        TEXT_COL: reviews[TEXT_COL],
    })
    meta = load_dataset(HF_NAME, f"raw_meta_{args.category}", split="full", trust_remote_code=True)
    meta_df = pl.DataFrame({
        JOIN_COL: meta[JOIN_COL],
        CLIENT_COL: meta[CLIENT_COL],
    }).unique(subset=[JOIN_COL])
    out = reviews_df.join(
        meta_df,
        on=JOIN_COL,
        how="left"
    ).drop(JOIN_COL).sample(fraction=args.fraction, seed=42)

    unique_clients = out[CLIENT_COL].value_counts(sort=True)[CLIENT_COL].to_list()
    unique_clients = unique_clients[:min(len(unique_clients), args.max_clients)]
    processed_dataset = []
    for cid in unique_clients:
        client_data = out.filter(pl.col(CLIENT_COL) == cid)
        if len(client_data) < args.min_samples_per_client:
            print(f"⚠️ Client {cid} skipped (too few examples: {len(client_data)})")
            continue
        processed_dataset.append(client_data)
    if not processed_dataset:
        raise RuntimeError("No clients with enough data.")
    out = pl.concat(processed_dataset)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    out.write_parquet(args.out)

    print("=" * 60, flush=True)
    print(f"[prep] wrote {args.out}", flush=True)
    print(f"[prep] shape          : {out.shape}", flush=True)
    print(f"[prep] clients (site) : {out[CLIENT_COL].n_unique()}", flush=True)
    print(f"[prep] classes        : {out[TARGET_COL].n_unique()} "
          f"(min={out[TARGET_COL].min()}, max={out[TARGET_COL].max()})", flush=True)
    print(f"[prep] dtypes:\n{out.dtypes}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
