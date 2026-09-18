PARAM_MAP = {
    "round_epochs": "federated_params.round_epochs",
    "server_lr": "federated_method.server_lr",
    "tau": "federated_method.tau",
}


def suggest_federated_params(trial):
    return {
        "federated_params.round_epochs": trial.suggest_int("round_epochs", 1, 10),
        "federated_method.server_lr": trial.suggest_float("server_lr", 1e-3, 10.0, log=True),
        "federated_method.tau": trial.suggest_float("tau", 1e-5, 1e-1, log=True),
    }
