from ..fedavg.fedavg import FedAvg
from .ditto_client import DittoClient
from .ditto_server import DittoServer


class Ditto(FedAvg):
    """Ditto (Li et al., ICML 2021): personalized FL with a prox-regularized personal model."""

    emit_two_track_report = True

    def __init__(
        self,
        ditto_lambda=0.1,
        ditto_personal_lr=0.0003,
        ditto_personal_epochs=1,
        fedavg_study_name="adamw_nopruner",
    ):
        super().__init__()
        self.ditto_lambda = float(ditto_lambda)
        self.ditto_personal_lr = float(ditto_personal_lr)
        self.ditto_personal_epochs = int(ditto_personal_epochs)
        self.fedavg_study_name = fedavg_study_name

    def _init_federated(self, cfg, df):
        if self.fedavg_study_name is not None:
            self._load_fedavg_global_hypers(cfg)
        super()._init_federated(cfg, df)

    def _load_fedavg_global_hypers(self, cfg):
        """Inject best FedAvg lr/weight_decay/round_epochs into cfg from a tuned study's meta.json."""
        import json

        from omegaconf import OmegaConf, open_dict

        from utils.meta_writer import results_dir_for

        meta_path = results_dir_for(cfg, "fedavg", self.fedavg_study_name) / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Ditto: FedAvg meta.json not found at {meta_path}. Point "
                f"federated_method.fedavg_study_name at a tuned FedAvg study, or "
                f"set it to null to use the config defaults for the global leg."
            )
        params = (
            json.loads(meta_path.read_text()).get("best_trial", {}).get("params", {})
        )
        with open_dict(cfg):
            if "lr" in params:
                OmegaConf.update(cfg, "optimizer.lr", float(params["lr"]))
                self.ditto_personal_lr = float(params["lr"])
            if "weight_decay" in params:
                OmegaConf.update(
                    cfg, "optimizer.weight_decay", float(params["weight_decay"])
                )
            if "round_epochs" in params:
                OmegaConf.update(
                    cfg, "federated_params.round_epochs", int(params["round_epochs"])
                )
                self.ditto_personal_epochs = int(params["round_epochs"])
        print(
            f"[Ditto] From FedAvg '{self.fedavg_study_name}' ({meta_path}): "
            f"global lr={cfg.optimizer.lr}, wd={cfg.optimizer.weight_decay}, "
            f"round_epochs={cfg.federated_params.round_epochs}; "
            f"personal_lr={self.ditto_personal_lr}, "
            f"personal_epochs={self.ditto_personal_epochs}",
            flush=True,
        )

    def _init_server(self, cfg):
        self.server = DittoServer(cfg, train_df=getattr(self, "df", None))

    def _init_client_cls(self):
        super()._init_client_cls()
        self.client_cls = DittoClient
        self.client_kwargs["client_cls"] = self.client_cls
        self.client_args.extend(
            [self.ditto_lambda, self.ditto_personal_lr, self.ditto_personal_epochs]
        )

    def get_communication_content(self, rank):
        content = super().get_communication_content(rank)
        content["ditto_v"] = self.server.v_states[rank]
        return content

    def _two_track_final_metric(self, best_val_after, best_val_max3):
        """Optuna objective: median over clients of the best-val max track."""
        import math

        import numpy as np

        vals = [v for v in best_val_max3 if v is not None and not math.isinf(v)]
        return float(np.median(vals)) if vals else 0.0
