PARAM_MAP = {
    # Head (personal classifier) local head epochs (paper's tau).
    "fedrep_head_epochs": "federated_method.fedrep_head_epochs",
    # Body (shared feature extractor) local body epochs.
    "fedrep_body_epochs": "federated_method.fedrep_body_epochs",
}


SKIP_COMMON_PARAMS = {"lr", "weight_decay"}

FEDREP_EPOCH_GRID = [1, 2, 5, 10]
SEARCH_GRID = {
    "fedrep_head_epochs": FEDREP_EPOCH_GRID,
    "fedrep_body_epochs": FEDREP_EPOCH_GRID,
}

def suggest_federated_params(trial):
    return {
        "federated_method.fedrep_head_epochs": trial.suggest_categorical("fedrep_head_epochs", FEDREP_EPOCH_GRID),
        "federated_method.fedrep_body_epochs": trial.suggest_categorical("fedrep_body_epochs", FEDREP_EPOCH_GRID),
    }
