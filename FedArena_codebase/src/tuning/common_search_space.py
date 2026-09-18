PARAM_MAP = {
    "lr": "optimizer.lr",
    "weight_decay": "optimizer.weight_decay",
}


_DEFAULT_LR_RANGE = (1e-5, 1e-1)

_DATASET_LR_RANGES = {
    "flair": (1e-5, 1e-2),
    "peta": (1e-5, 1e-2),
}


def _resolve_lr_range(cfg):
    """Pick LR range based on cfg.train_dataset._target_; fall back to _DEFAULT_LR_RANGE."""
    if cfg is None:
        return _DEFAULT_LR_RANGE
    try:
        target = str(cfg.train_dataset._target_).lower()
    except Exception:
        return _DEFAULT_LR_RANGE
    for keyword, lr_range in _DATASET_LR_RANGES.items():
        if keyword in target:
            return lr_range
    return _DEFAULT_LR_RANGE


def suggest_common_params(trial, cfg=None, skip=None):
    """Suggest common hyperparams (optimizer.lr, optimizer.weight_decay)."""
    skip = set(skip or [])
    params = {}

    lr_low, lr_high = _resolve_lr_range(cfg)

    if "lr" not in skip:
        params["optimizer.lr"] = trial.suggest_float("lr", lr_low, lr_high, log=True)
    if "weight_decay" not in skip:
        params["optimizer.weight_decay"] = trial.suggest_float(
            "weight_decay", 1e-6, 1e-2, log=True
        )
    return params
