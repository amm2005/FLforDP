import torch
import numpy as np


def get_loss(loss_cfg, df=None, device=None, init_pos_weight=None):
    loss_name = loss_cfg.loss_name
    loss = loss_cfg.config
    if loss_name == "ce":
        if init_pos_weight:
            pos_weight = calculate_class_weights_multi_class(df)
            pos_weight = torch.tensor(pos_weight).to(device)
        else:
            pos_weight = None
        return torch.nn.CrossEntropyLoss(
            weight=pos_weight,
            ignore_index=loss.ignore_index,
            reduction=loss.reduction,
            label_smoothing=loss.label_smoothing,
        )
    elif loss_name == "mse":
        return torch.nn.MSELoss(reduction=loss.reduction)
    elif loss_name == "bce":
        return torch.nn.BCEWithLogitsLoss(reduction=loss.reduction)
    elif loss_name == "focal_weighted":
        alpha = loss.alpha if "alpha" in loss else None
        if alpha is not None:
            alpha = torch.tensor(list(alpha), dtype=torch.float)
        elif init_pos_weight and df is not None:
            alpha = torch.tensor(
                calculate_class_weights_multi_class(df), dtype=torch.float
            )
        else:
            raise ValueError(
                "focal_weighted loss requires `config.alpha` in the loss yaml "
                "or init_pos_weight=True with a df to derive per-class weights."
            )
        return FocalLoss(alpha=alpha, gamma=loss.gamma, reduction=loss.reduction)
    else:
        raise ValueError("Unknown type of loss function")


class FocalLoss(torch.nn.Module):
    """Weighted focal loss (FLamby Fed-ISIC2019 BaselineLoss)."""

    def __init__(self, alpha, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = torch.as_tensor(alpha, dtype=torch.float)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        targets = targets.view(-1, 1).type_as(inputs)
        logpt = torch.nn.functional.log_softmax(inputs, dim=1)
        logpt = logpt.gather(1, targets.long())
        logpt = logpt.view(-1)
        pt = logpt.exp()
        self.alpha = self.alpha.to(targets.device)
        at = self.alpha.gather(0, targets.data.view(-1).long())
        logpt = logpt * at
        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


def calculate_class_weights_multi_class(df):
    df_copy = df.copy()
    target_array = np.array(df_copy["target"].tolist())

    class_weights = {}
    ordered_weights = []

    unique_classes = np.unique(target_array)
    total_count = len(target_array)

    for cls in unique_classes:
        class_count = np.sum(target_array == cls, axis=0)
        class_weight = float(total_count / (len(unique_classes) * class_count))
        class_weights[cls] = class_weight

    for cls in sorted(unique_classes):
        ordered_weights.append(class_weights[cls])

    return ordered_weights
