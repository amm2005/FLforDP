# pFedMe tuning. Per the PFedMe protocol (see pfedme.py docstring), Optuna tunes
# ONLY (lambda, beta); K, eta, R and the inner lr are taken from the tuned FedAvg
# run (fedavg_study_name) / config defaults, NOT from this search. lr and
# weight_decay are FedAvg-fixed too, so they are excluded from the common search.

PARAM_MAP = {
    "pfedme_lambda": "federated_method.pfedme_lambda",
    "pfedme_beta": "federated_method.pfedme_beta",
}

# Common lr/weight_decay are NOT tuned (they belong to the FedAvg-fixed global
# model, mirrored from the separately-tuned FedAvg study).
SKIP_COMMON_PARAMS = {"lr", "weight_decay"}

# lambda: the paper requires lambda > 2L (nonconvex) / lambda in (0, inf); pFedMe
# experiments use large lambda (~10-30) so theta_k is well-anchored to w_k. Ditto
# results showed the personal model often wants lambda at the top of the grid, so
# the range extends to 50 to avoid saturating at the boundary.
# 1-2-5 per decade over [1, 50].
PFEDME_LAMBDA_GRID = [1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0]
# beta: global relaxation (Line 12). beta=1 -> plain averaging of the w_k;
# beta<1 -> conservative (keeps a little of the old w^t). Grid {0.95, 1.0}:
# drop over-relaxation (beta=2 was unstable + the global leg now in objective).
PFEDME_BETA_GRID = [0.9, 0.95, 1.0, 1.10]
# 8 x 2 = 16 trials (no pruning, mirrors ditto search_space).
SEARCH_GRID = {
    "pfedme_lambda": PFEDME_LAMBDA_GRID,
    "pfedme_beta": PFEDME_BETA_GRID,
}


def suggest_federated_params(trial):
    return {
        "federated_method.pfedme_lambda": trial.suggest_categorical(
            "pfedme_lambda", PFEDME_LAMBDA_GRID
        ),
        "federated_method.pfedme_beta": trial.suggest_categorical(
            "pfedme_beta", PFEDME_BETA_GRID
        ),
    }
