from ..fedavg.fedavg import FedAvg
from .fedrep_client import FedRepClient
from .fedrep_server import FedRepServer


class FedRep(FedAvg):
    """FedRep (Collins et al., ICML 2021): shared federated body with a personal local head."""

    emit_two_track_report = True

    def __init__(
        self,
        fedrep_head_epochs=10,
        fedrep_body_epochs=1,
        head_param_keys=("fc", "linear", "classifier"),
        fedavg_study_name="adamw_nopruner",
        fedrep_save_state=True,
    ):
        super().__init__()
        self.fedrep_head_epochs = int(fedrep_head_epochs)
        self.fedrep_body_epochs = int(fedrep_body_epochs)
        self.head_param_keys = list(head_param_keys)
        self.fedavg_study_name = fedavg_study_name
        self.fedrep_save_state = bool(fedrep_save_state)

    def _init_federated(self, cfg, df):
        if self.fedavg_study_name is not None:
            self._load_fedavg_global_hypers(cfg)
        super()._init_federated(cfg, df)

    def _load_fedavg_global_hypers(self, cfg):
        """Inject best FedAvg lr/weight_decay into cfg from a tuned study's meta.json."""
        import json

        from omegaconf import OmegaConf, open_dict

        from utils.meta_writer import results_dir_for

        meta_path = results_dir_for(cfg, "fedavg", self.fedavg_study_name) / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"FedRep: FedAvg meta.json not found at {meta_path}. Point "
                f"federated_method.fedavg_study_name at a tuned FedAvg study, or "
                f"set it to null to use the config defaults."
            )
        params = (
            json.loads(meta_path.read_text()).get("best_trial", {}).get("params", {})
        )
        with open_dict(cfg):
            if "lr" in params:
                OmegaConf.update(cfg, "optimizer.lr", float(params["lr"]))
            if "weight_decay" in params:
                OmegaConf.update(
                    cfg, "optimizer.weight_decay", float(params["weight_decay"])
                )
        print(
            f"[FedRep] From FedAvg '{self.fedavg_study_name}' ({meta_path}): "
            f"lr={cfg.optimizer.lr}, wd={cfg.optimizer.weight_decay}",
            flush=True,
        )

    def _init_server(self, cfg):
        self.server = FedRepServer(
            cfg,
            train_df=getattr(self, "df", None),
            save_state=self.fedrep_save_state,
        )

    def _init_client_cls(self):
        super()._init_client_cls()
        self.client_cls = FedRepClient
        self.client_kwargs["client_cls"] = self.client_cls
        self.client_args.extend(
            [self.fedrep_head_epochs, self.fedrep_body_epochs, self.head_param_keys]
        )

    def get_communication_content(self, rank):
        content = super().get_communication_content(rank)
        # Head must be applied before the global body: dict insertion order == pipe order.
        prefix = {
            "fedrep_head": self.server.head_states[rank],
        }
        return {**prefix, **content}

    def _two_track_final_metric(self, best_val_after, best_val_max3):
        """Optuna objective: median over clients of the best-val max track."""
        import math

        import numpy as np

        vals = [v for v in best_val_max3 if v is not None and not math.isinf(v)]
        return float(np.median(vals)) if vals else 0.0
