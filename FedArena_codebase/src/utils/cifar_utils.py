import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_cifar_metrics(fin_targets, results, probs=None, verbose=False):
    """Classification metrics for multi-class (single-label) tasks."""
    df = pd.DataFrame(
        columns=["cifar"],
        index=[
            "Accuracy",
            "Precision",
            "Recall",
            "f1-score",
        ],
    )
    df.loc["Accuracy", "cifar"] = accuracy_score(fin_targets, results)
    df.loc["Precision", "cifar"] = precision_score(
        fin_targets, results, average="macro", zero_division=0
    )
    df.loc["Recall", "cifar"] = recall_score(
        fin_targets, results, average="macro", zero_division=0
    )
    df.loc["f1-score", "cifar"] = f1_score(
        fin_targets, results, average="macro", zero_division=0
    )
    df.loc["balanced_accuracy", "cifar"] = balanced_accuracy_score(
        fin_targets, results
    )

    if probs is not None:
        probs = np.asarray(probs)
        y_true = np.asarray(fin_targets).ravel()
        num_classes = probs.shape[1]

        try:
            if num_classes == 2:
                roc = roc_auc_score(y_true, probs[:, 1])
            else:
                roc = roc_auc_score(
                    y_true, probs, multi_class="ovr", average="macro"
                )
        except ValueError:
            roc = 0.0
        df.loc["ROC-AUC", "cifar"] = roc

        y_onehot = np.zeros((len(y_true), num_classes), dtype=np.float32)
        y_onehot[np.arange(len(y_true)), y_true.astype(int)] = 1.0

        per_class_ap = []
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="No positive class found in y_true",
                category=UserWarning,
            )
            for c in range(num_classes):
                if y_onehot[:, c].sum() > 0:
                    try:
                        per_class_ap.append(
                            average_precision_score(y_onehot[:, c], probs[:, c])
                        )
                    except ValueError:
                        pass
        macro_ap = float(np.mean(per_class_ap)) if per_class_ap else float("nan")
        df.loc["macro_AP", "cifar"] = macro_ap

        k = min(5, num_classes)
        top_k = np.argsort(-probs, axis=1)[:, :k]
        match = top_k == y_true.astype(int)[:, None]
        has = match.any(axis=1)
        rank0 = np.argmax(match, axis=1)
        score = np.where(has, 1.0 / (rank0 + 1.0), 0.0)
        df.loc["MAP@5", "cifar"] = float(score.mean())

    if verbose:
        print(df)
    return df
