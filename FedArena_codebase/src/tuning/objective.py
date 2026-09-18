import copy
import importlib
import inspect
import multiprocessing as _mp
import traceback

import optuna
from omegaconf import OmegaConf, open_dict
from hydra.utils import instantiate

from .common_search_space import suggest_common_params


SEARCH_SPACE_REGISTRY = {
    "federated_methods.fedavg": "federated_methods.fedavg.search_space",
    "federated_methods.local_only": "federated_methods.local_only.search_space",
    "federated_methods.fedprox": 'federated_methods.fedprox.search_space',
    "federated_methods.scaffold": "federated_methods.scaffold.search_space",
    "federated_methods.feddyn": "federated_methods.feddyn.search_space",
    "federated_methods.fednova": "federated_methods.fednova.search_space",
    "federated_methods.fedspeed": "federated_methods.fedspeed.search_space",
    "federated_methods.fedyogi": "federated_methods.fedyogi.search_space",
    "federated_methods.mime": "federated_methods.mime.search_space",
    "federated_methods.mimelite": "federated_methods.mimelite.search_space",
    "federated_methods.ditto": "federated_methods.ditto.search_space",
    "federated_methods.fedrep": "federated_methods.fedrep.search_space",
    "federated_methods.pfedme": "federated_methods.pfedme.search_space",
}


def apply_overrides(cfg, overrides):
    """Deep-copy cfg and apply {dotted_key: value} overrides via OmegaConf.update."""
    cfg = copy.deepcopy(cfg)
    with open_dict(cfg):
        for key, value in overrides.items():
            OmegaConf.update(cfg, key, value)
    return cfg


def _resolve_method_key(cfg):
    target = cfg.federated_method._target_
    parts = target.split(".")
    return f"{parts[0]}.{parts[1]}"


def get_method_search_space(cfg):
    key = _resolve_method_key(cfg)
    if key not in SEARCH_SPACE_REGISTRY:
        raise ValueError(
            f"No search space registered for '{key}'. "
            f"Available: {list(SEARCH_SPACE_REGISTRY.keys())}"
        )
    return importlib.import_module(SEARCH_SPACE_REGISTRY[key])


def build_overrides_from_params(base_cfg, params):
    """Map Optuna best_params dict back to Hydra config overrides."""
    from .common_search_space import PARAM_MAP as common_map

    method_space = get_method_search_space(base_cfg)
    method_map = getattr(method_space, "PARAM_MAP", {})

    full_map = {**common_map, **method_map}
    overrides = {}
    for optuna_name, value in params.items():
        if optuna_name in full_map:
            overrides[full_map[optuna_name]] = value
        else:
            print(f"Warning: unknown Optuna param '{optuna_name}', skipping")
    return overrides


class _PruningCallback:
    """Reports intermediate val metric to Optuna; prunes low-performing trials."""

    def __init__(self, trial, metric_name="ROC-AUC"):
        self.trial = trial
        self.metric_name = metric_name

    def __call__(self, round_idx, current_val_metrics):
        value = current_val_metrics.get(self.metric_name, 0.0)
        self.trial.report(value, step=round_idx)
        if self.trial.should_prune():
            raise optuna.TrialPruned()


def _run_trial(conn, trial_cfg, df, trial, metric_name):
    """Execute one FL trial in a forked subprocess (isolated CUDA context)."""
    try:
        trainer = instantiate(trial_cfg.federated_method, _recursive_=False)
        trainer._init_federated(trial_cfg, df)

        callback = _PruningCallback(trial, metric_name=metric_name)
        results = trainer.begin_train(round_callback=callback)

        conn.send(("ok", results["best_val_tune_metric"]))
    except optuna.TrialPruned:
        conn.send(("pruned", None))
    except Exception as e:
        traceback.print_exc()
        conn.send(("error", str(e)))
    finally:
        conn.close()


def create_objective(base_cfg, df, metric_name="ROC-AUC"):
    method_space = get_method_search_space(base_cfg)

    def objective(trial):
        overrides = {}
        skip = getattr(method_space, "SKIP_COMMON_PARAMS", ())
        overrides.update(suggest_common_params(trial, cfg=base_cfg, skip=skip))
        fed_fn = method_space.suggest_federated_params
        if "cfg" in inspect.signature(fed_fn).parameters:
            overrides.update(fed_fn(trial, cfg=base_cfg))
        else:
            overrides.update(fed_fn(trial))
        overrides["federated_params.print_client_metrics"] = False
        
        print(f"\n{'=' * 60}", flush=True)
        print(f"TRIAL #{trial.number} starting", flush=True)
        print(f"{'=' * 60}", flush=True)
        for name, value in sorted(trial.params.items()):
            if isinstance(value, float):
                print(f"  {name}: {value:.6g}", flush=True)
            else:
                print(f"  {name}: {value}", flush=True)
        print(f"{'=' * 60}\n", flush=True)

        trial_cfg = apply_overrides(base_cfg, overrides)

        reader, writer = _mp.Pipe(duplex=False)
        p = _mp.Process(target=_run_trial, args=(writer, trial_cfg, df, trial, metric_name))
        p.start()
        writer.close()  # parent must close its write-end copy, else recv() hangs on child death

        try:
            status, value = reader.recv()
        except EOFError:
            p.join()
            raise RuntimeError(
                f"Trial subprocess crashed (exit code {p.exitcode})"
            )
        p.join()

        if status == "pruned":
            print(
                f"\n>>> TRIAL #{trial.number} PRUNED "
                f"(intermediate {metric_name} below median across surviving trials)\n",
                flush=True,
            )
            raise optuna.TrialPruned()
        elif status == "error":
            print(f"\n>>> TRIAL #{trial.number} ERROR: {value}\n", flush=True)
            raise RuntimeError(value)

        print(
            f"\n>>> TRIAL #{trial.number} COMPLETED with {metric_name}={value:.6f}\n",
            flush=True,
        )
        return value

    return objective
