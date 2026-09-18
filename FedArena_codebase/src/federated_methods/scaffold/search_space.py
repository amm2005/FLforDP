PARAM_MAP = {
    "round_epochs": "federated_params.round_epochs",
    "scaffold_local_lr": "federated_method.local_lr",
}


SKIP_COMMON_PARAMS = {'lr'}


def suggest_federated_params(trial):
    return {
        "federated_params.round_epochs": trial.suggest_int("round_epochs", 1, 10),
        "federated_method.local_lr": trial.suggest_float(
            "scaffold_local_lr", 1e-5, 1e-1, log=True
        ),
    }
