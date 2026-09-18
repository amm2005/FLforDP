"""Write a ``meta.json`` manifest for a finished Optuna tuning run."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf

# Map train-dataset module stems to canonical dataset keys used by the results/ tree.
_DATASET_NAME_ALIASES: dict[str, str] = {
    "home_credit": "homecredit",
    "home_credit_bureau_b": "bureau_b",
}


def _repo_root() -> Path:
    """Repo root = two levels up from ``src/utils/meta_writer.py``."""
    return Path(__file__).resolve().parents[2]


def _select(cfg, dotted: str, default=None):
    """``OmegaConf.select`` that degrades to *default* on any failure / None."""
    try:
        val = OmegaConf.select(cfg, dotted, default=default)
    except Exception:
        return default
    return default if val is None else val


def canonical_dataset_name(cfg) -> str:
    """Derive the canonical dataset key from ``cfg.train_dataset._target_``."""
    target = _select(cfg, "train_dataset._target_")
    if not isinstance(target, str) or not target:
        raise ValueError("cfg.train_dataset._target_ is missing or not a string")
    parts = target.split(".")
    module_stem = parts[-2] if len(parts) >= 2 else parts[0]
    if module_stem.endswith("_dataset"):
        module_stem = module_stem[: -len("_dataset")]
    name = module_stem.lower()
    return _DATASET_NAME_ALIASES.get(name, name)


def federated_method_name(cfg) -> str:
    """Method name from ``cfg.federated_method._target_`` (e.g. ``fedavg``)."""
    target = _select(cfg, "federated_method._target_")
    if not isinstance(target, str) or not target:
        raise ValueError("cfg.federated_method._target_ is missing or not a string")
    parts = target.split(".")
    return (parts[1] if len(parts) >= 2 else parts[0]).lower()


def optimizer_name(cfg) -> str:
    """Optimizer name from ``cfg.optimizer._target_`` (e.g. ``adamw``)."""
    target = _select(cfg, "optimizer._target_")
    if not isinstance(target, str) or not target:
        return "unknown"
    return target.split(".")[-1].lower()


def run_results_dir(cfg, *, dataset=None) -> Path:
    """``<repo>/results/<dataset>/<federated_method>/<study_name>`` for this run."""
    repo_root = _repo_root()
    ds = dataset or canonical_dataset_name(cfg)
    method = federated_method_name(cfg)
    study = _select(cfg, "tuning.study_name")
    if not isinstance(study, str) or not study:
        raise ValueError("cfg.tuning.study_name is missing or not a string")
    return repo_root / "results" / ds / method / study


def results_dir_for(cfg, method: str, study_name: str, *, dataset=None) -> Path:
    """``<repo>/results/<dataset>/<method>/<study_name>`` for an arbitrary run."""
    ds = dataset or canonical_dataset_name(cfg)
    return _repo_root() / "results" / str(ds) / str(method) / str(study_name)


def _jsonable(value):
    """Coerce numpy / exotic scalars into JSON-serialisable Python scalars."""
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def _best_trial_info(study):
    """``{number, value, params}`` for the best trial, or ``None`` if unavailable."""
    if study is None:
        return None
    try:
        best = study.best_trial
    except Exception:
        return None
    if best is None:
        return None
    try:
        params = {k: _jsonable(v) for k, v in dict(best.params).items()}
    except Exception:
        params = {}
    return {
        "number": getattr(best, "number", None),
        "value": _jsonable(getattr(best, "value", None)),
        "params": params,
    }


def _study_timing(study):
    """Return ``(started_iso, completed_iso, duration_seconds)`` defensively."""
    if study is None:
        return None, None, None
    try:
        trials = list(study.trials)
    except Exception:
        trials = []

    started = None
    completed = None
    if trials:
        started = getattr(trials[0], "datetime_start", None)
        if not isinstance(started, datetime):
            starts = [
                getattr(t, "datetime_start", None) for t in trials
            ]
            starts = [s for s in starts if isinstance(s, datetime)]
            started = min(starts) if starts else None

        completed = getattr(trials[-1], "datetime_complete", None)
        if not isinstance(completed, datetime):
            comps = [
                getattr(t, "datetime_complete", None) for t in trials
            ]
            comps = [c for c in comps if isinstance(c, datetime)]
            completed = max(comps) if comps else None

    started_iso = started.isoformat() if isinstance(started, datetime) else None
    completed_iso = (
        completed.isoformat() if isinstance(completed, datetime) else None
    )
    duration = None
    if isinstance(started, datetime) and isinstance(completed, datetime):
        try:
            duration = int((completed - started).total_seconds())
        except Exception:
            duration = None
    return started_iso, completed_iso, duration


def _db_rel_path(storage, repo_root: Path):
    """Turn an Optuna storage URL into a repo-root-relative DB path."""
    if not isinstance(storage, str) or not storage:
        return None
    prefix = "sqlite:///"
    if not storage.startswith(prefix):
        return storage
    raw = storage[len(prefix):]
    p = Path(raw)
    if not p.is_absolute():
        p = repo_root / p
    p = p.resolve()
    try:
        return p.relative_to(repo_root).as_posix()
    except ValueError:
        return Path(os.path.relpath(p, repo_root)).as_posix()


def write_meta_json(
    cfg, study, source_run_dir, *, aliases=None, dataset=None, is_local_only=None,
    storage=None, best_trial_override=None,
) -> Path:
    """Write the ``meta.json`` manifest for a run and return its path."""
    if cfg is None:
        raise ValueError("cfg must not be None")
    if source_run_dir is None:
        raise ValueError("source_run_dir must not be None")

    repo_root = _repo_root()

    if dataset is None:
        dataset = canonical_dataset_name(cfg)
    elif not isinstance(dataset, str) or not dataset:
        raise ValueError("dataset override must be a non-empty string")

    fed_method = federated_method_name(cfg)

    study_name = _select(cfg, "tuning.study_name")
    if not isinstance(study_name, str) or not study_name:
        raise ValueError("cfg.tuning.study_name is missing or not a string")

    metric_name = _select(cfg, "tuning.metric")
    optimizer = optimizer_name(cfg)
    storage_used = storage if storage is not None else _select(cfg, "tuning.storage")
    db_path = _db_rel_path(storage_used, repo_root)
    if is_local_only is None:
        is_local_only = fed_method == "local_only"
    else:
        is_local_only = bool(is_local_only)

    if aliases is None:
        aliases = [fed_method]
    else:
        aliases = [str(a) for a in aliases]

    src_path = Path(source_run_dir)
    if not src_path.is_absolute():
        src_path = repo_root / src_path
    src_path = src_path.resolve()
    try:
        source_rel = src_path.relative_to(repo_root).as_posix()
    except ValueError:
        # Outside the repo: keep a relative ".." path instead of an absolute one.
        source_rel = Path(os.path.relpath(src_path, repo_root)).as_posix()

    started_at, completed_at, duration_seconds = _study_timing(study)
    if study is None:
        n_trials = None
    else:
        try:
            n_trials = len(study.trials)
        except Exception:
            n_trials = None

    meta = {
        "dataset": dataset,
        "study_name": study_name,
        "federated_method": fed_method,
        "source_run_dir": source_rel,
        "db_path": db_path,
        "best_trial": (
            best_trial_override
            if best_trial_override is not None
            else _best_trial_info(study)
        ),
        "started_at": started_at,
        "completed_at": completed_at,
        "duration_seconds": duration_seconds,
        "n_trials": n_trials,
        "metric_name": metric_name,
        "optimizer": optimizer,
        "is_local_only": is_local_only,
        "aliases": aliases,
    }

    out_dir = run_results_dir(cfg, dataset=dataset)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "meta.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True, ensure_ascii=False)
        fh.write("\n")
    return out_path


def init_meta_json(
    cfg, source_run_dir, *, aliases=None, dataset=None, is_local_only=None, storage=None
) -> Path:
    """Write the manifest at the start of a run, before any trials exist."""
    return write_meta_json(
        cfg, None, source_run_dir,
        aliases=aliases, dataset=dataset, is_local_only=is_local_only, storage=storage,
    )
