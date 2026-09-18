PARAM_MAP = {
    "round_epochs": "federated_params.round_epochs",
}


def suggest_federated_params(trial):
    return {
        "federated_params.round_epochs": trial.suggest_int("round_epochs", 1, 10),
    }
