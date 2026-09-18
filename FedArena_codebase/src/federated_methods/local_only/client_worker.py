"""Multiprocessing entry for LocalOnly clients."""

from __future__ import annotations

import os


def run_multiprocess_client(*client_args, **client_kwargs):
    for key, val in (
        ("OMP_NUM_THREADS", "1"),
        ("MKL_NUM_THREADS", "1"),
        ("OPENBLAS_NUM_THREADS", "1"),
        ("VECLIB_MAXIMUM_THREADS", "1"),
        ("NUMEXPR_NUM_THREADS", "1"),
    ):
        os.environ[key] = val

    cache_snapshot = client_kwargs.pop("_preprocessing_cache_snapshot", None)
    if cache_snapshot:
        from utils.preprocessing_cache import preprocessing_cache

        preprocessing_cache.clear()
        preprocessing_cache.update(cache_snapshot)

    import torch

    torch.set_num_threads(1)

    from federated_methods.fedavg.client import multiprocess_client

    multiprocess_client(*client_args, **client_kwargs)
