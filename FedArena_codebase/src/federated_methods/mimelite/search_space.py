PARAM_MAP = {
    "round_epochs": "federated_params.round_epochs",
}


def suggest_federated_params(trial):
    # Only round_epochs is method-specific. optimizer.lr / optimizer.weight_decay
    # are tuned via the COMMON search space (suggest_common_params), same as
    # fedavg / fedyogi / feddyn / mime. No MimeLite-specific tuned params.
    return {
        "federated_params.round_epochs": trial.suggest_int("round_epochs", 1, 10),
    }
