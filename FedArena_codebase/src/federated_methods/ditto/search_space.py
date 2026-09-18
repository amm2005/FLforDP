# λ-grid tuning. The ONLY tuned Ditto hyperparameter is `ditto_lambda`

PARAM_MAP = {
    "ditto_lambda": "federated_method.ditto_lambda",
}

# Common lr/weight_decay are NOT tuned (they belong to the FedAvg-fixed global model).
SKIP_COMMON_PARAMS = {"lr", "weight_decay"}

# 13-value λ grid over [1e-4, 1] (1-2-5 per decade). (13 trials, no pruning).
DITTO_LAMBDA_GRID = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
SEARCH_GRID = {"ditto_lambda": DITTO_LAMBDA_GRID}


def suggest_federated_params(trial):
    return {
        "federated_method.ditto_lambda": trial.suggest_categorical(
            "ditto_lambda", DITTO_LAMBDA_GRID
        ),
    }
