PARAM_MAP = {
    "round_epochs": "federated_params.round_epochs",
    "lambda_param": "federated_method.lambda_param",
    "alpha": "federated_method.alpha",
    "rho_0": "federated_method.rho_0",
}

def suggest_federated_params(trial): 
    return {
        "federated_params.round_epochs": trial.suggest_int("round_epochs", 1, 10), 
        "federated_method.lambda_param": trial.suggest_float("lambda_param", 1, 1000, log=True),
        "federated_method.alpha": trial.suggest_float("alpha", 0.0, 1.0),
        "federated_method.rho_0": trial.suggest_float("rho_0", 1e-3, 0.5, log=True),
    }