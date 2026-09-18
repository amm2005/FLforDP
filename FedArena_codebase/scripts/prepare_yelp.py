#!/usr/bin/env python3
import argparse
import os

import kagglehub
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        default="data/yelp/yelp_processed.parquet",
        help="Output parquet path.",
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


    path = kagglehub.dataset_download("yelp-dataset/yelp-dataset")
    dataset = pd.read_json(path + '/yelp_academic_dataset_review.json', lines=True, nrows=800000)
    dataset = dataset[["text", "business_id", "stars"]]
    unique_clients = dataset["business_id"].value_counts().index.tolist()
    unique_clients = unique_clients[:min(len(unique_clients), args.max_clients)]

    processed_dataset = []
    for cid in unique_clients:
        client_data = dataset[dataset["business_id"] == cid]
        if len(client_data) < args.min_samples_per_client:
            print(f"⚠️ Client {cid} skipped (too few examples: {len(client_data)})")
            continue
        processed_dataset.append(client_data)
    if not processed_dataset:
        raise RuntimeError("No clients with enough data.")
    processed_dataset = pd.concat(processed_dataset, axis=0)
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    processed_dataset.to_parquet(args.out, index=False)

    print("=" * 60, flush=True)
    print(f"[prep] wrote {args.out}", flush=True)
    print(f"[prep] shape          : {processed_dataset.shape}", flush=True)
    print(f"[prep] clients (site) : {processed_dataset['business_id'].nunique()}", flush=True)
    print(f"[prep] classes        : {processed_dataset['stars'].nunique()} "
          f"(min={processed_dataset['stars'].min()}, max={processed_dataset['stars'].max()})", flush=True)
    print(f"[prep] dtypes:\n{processed_dataset.dtypes}", flush=True)
    print("=" * 60, flush=True)

if __name__ == "__main__":
    main()
