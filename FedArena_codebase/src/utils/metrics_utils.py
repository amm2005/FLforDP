import math
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    f1_score,
    confusion_matrix,
    average_precision_score,
    fbeta_score,
    r2_score,
)

from .cifar_utils import calculate_cifar_metrics


__all__ = [
    "stopping_criterion",
]


def stopping_criterion(
    val_loss,
    metrics,
    best_metrics,
    epochs_no_improve,
):
    """Update best_metrics only if every saving metric improves."""
    metrics = dict(metrics.mean(axis=1))
    metrics_mask = all(
        metrics[key] >= best_metrics[key] for key in best_metrics.keys() - {"loss"}
    )
    if not metrics_mask:
        epochs_no_improve += 1
        return epochs_no_improve, best_metrics
    if "loss" in list(best_metrics.keys()):
        if val_loss >= best_metrics["loss"]:
            epochs_no_improve += 1
            return epochs_no_improve, best_metrics
    for key in list(best_metrics.keys()):
        if key == "loss":
            best_metrics[key] = val_loss
        else:
            best_metrics[key] = metrics[key]
    epochs_no_improve = 0
    return epochs_no_improve, best_metrics


def calculate_regression_metrics(fin_targets, fin_outputs, verbose=False):
    """R2 and RMSE for regression, un-standardized to original units if stats exist."""
    y_true = np.asarray(fin_targets, dtype=np.float64).ravel()
    y_pred = np.asarray(fin_outputs, dtype=np.float64).ravel()

    try:
        from utils.preprocessing_cache import preprocessing_cache
        cache = preprocessing_cache
        if cache and "target_std" in cache and "target_mean" in cache:
            t_mean = float(cache["target_mean"])
            t_std = float(cache["target_std"])
            if t_std != 0.0 and np.isfinite(t_std):
                y_true = y_true * t_std + t_mean
                y_pred = y_pred * t_std + t_mean
    except Exception:
        pass

    r2 = r2_score(y_true, y_pred)
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    df = pd.DataFrame(
        columns=["regression"],
        index=["R2", "RMSE"],
    )
    df.loc["R2", "regression"] = r2
    df.loc["RMSE", "regression"] = rmse
    if verbose:
        print(df)
    return df


def calculate_multilabel_metrics(fin_targets, fin_outputs, verbose=False):
    """macro AP and micro AP for multi-label classification from raw logits."""
    y_true = np.asarray(fin_targets, dtype=np.float32)
    y_logits = np.asarray(fin_outputs, dtype=np.float32)

    y_score = 1.0 / (1.0 + np.exp(-np.clip(y_logits, -50, 50)))

    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="No positive class found in y_true",
            category=UserWarning,
        )
        try:
            macro_ap = float(average_precision_score(y_true, y_score, average="macro"))
        except ValueError:
            macro_ap = float("nan")
        try:
            micro_ap = float(average_precision_score(y_true, y_score, average="micro"))
        except ValueError:
            micro_ap = float("nan")

    df = pd.DataFrame(
        columns=["multilabel"],
        index=["macro_AP", "micro_AP"],
    )
    df.loc["macro_AP", "multilabel"] = macro_ap
    df.loc["micro_AP", "multilabel"] = micro_ap
    if verbose:
        print(df)
    return df


def calculate_metrics(
    fin_targets,
    fin_outputs,
    verbose=False,
):
    fin_targets = torch.as_tensor(fin_targets)
    fin_outputs = torch.as_tensor(fin_outputs)

    if (
        fin_targets.dim() == 2
        and fin_targets.shape[1] > 1
        and fin_targets.dtype.is_floating_point
    ):
        metrics = calculate_multilabel_metrics(
            fin_targets.numpy(), fin_outputs.numpy(), verbose
        )
        return metrics, None

    if fin_outputs.dim() == 1 or (fin_outputs.dim() == 2 and fin_outputs.shape[1] == 1):
        metrics = calculate_regression_metrics(
            fin_targets.numpy(), fin_outputs.numpy(), verbose
        )
        prediction_threshold = None
        return metrics, prediction_threshold

    softmax = torch.nn.Softmax(dim=1)
    probs = softmax(fin_outputs)
    results = probs.max(dim=1)[1]
    metrics = calculate_cifar_metrics(
        fin_targets, results, probs=probs.detach().cpu().numpy(), verbose=verbose
    )
    prediction_threshold = None
    return metrics, prediction_threshold


def check_metrics_names(metrics):
    allowed_metrics = [
        "loss",
        "Specificity",
        "Sensitivity",
        "G-mean",
        "f1-score",
        "fbeta2-score",
        "ROC-AUC",
        "AP",
        "Precision (PPV)",
        "NPV",
        "R2",
        "RMSE",
        "macro_AP",
        "micro_AP",
        "MAP@5",
        "balanced_accuracy",
    ]

    assert all(
        [k in allowed_metrics for k in metrics.keys()]
    ), f"federated_params.server_saving_metrics can be only {allowed_metrics}, but get {metrics.keys()}"
