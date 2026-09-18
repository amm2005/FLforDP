PARAM_MAP = {
    "round_epochs": "federated_params.round_epochs",
    "fed_dyn_alpha": "federated_method.fed_dyn_alpha",
}


def suggest_federated_params(trial):
    return {
        "federated_params.round_epochs": trial.suggest_int("round_epochs", 1, 10),
        "federated_method.fed_dyn_alpha": trial.suggest_float("fed_dyn_alpha", 1e-3, 1e-1, log=True),
    }
