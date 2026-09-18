PARAM_MAP = {
    "round_epochs": "federated_params.round_epochs",
    "fed_prox_lambda": "federated_method.fed_prox_lambda",
}


def suggest_federated_params(trial):
    return {
        "federated_params.round_epochs": trial.suggest_int("round_epochs", 1, 10),
        "federated_method.fed_prox_lambda": trial.suggest_float(
            "fed_prox_lambda", 1e-5, 1, log=True
        ),
    }
