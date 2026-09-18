"""Optuna search space for local-only training.

On image datasets ``round_epochs`` is tuned like the federated methods so the
local baseline gets the same total-local-epoch budget (R*E) for a fair
comparison. On tabular datasets it stays fixed at 1 (historical behaviour).
"""

from omegaconf import OmegaConf

# round_epochs only enters best_params on image runs (tuned below); mapping it
# here lets build_overrides_from_params propagate it to the final multi-seed
# eval. Harmless on tabular runs (key absent from best_params).
PARAM_MAP = {"round_epochs": "federated_params.round_epochs"}

# fed_isic2019 (like flair/peta/cifar) is an image dataset -> tune round_epochs so
# the local baseline gets the same total-local-epoch budget (R*E) as the federated
# methods (which tune round_epochs unconditionally). NOTE: "cct" is another image
# dataset NOT listed here, so its local_only round_epochs currently stays fixed at 1.
_IMAGE_DATASET_KEYS = ("flair", "peta", "cifar", "isic")


def _is_image_dataset(cfg):
    if cfg is None:
        return False
    target = str(OmegaConf.select(cfg, "train_dataset._target_") or "").lower()
    return any(k in target for k in _IMAGE_DATASET_KEYS)


def suggest_federated_params(trial, cfg=None):
    if _is_image_dataset(cfg):
        return {
            "federated_params.round_epochs": trial.suggest_int("round_epochs", 1, 10),
        }
    return {"federated_params.round_epochs": 1}
