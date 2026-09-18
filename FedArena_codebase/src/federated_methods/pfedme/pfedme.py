from ..fedavg.fedavg import FedAvg
from .pfedme_client import PFedMeClient
from .pfedme_server import PFedMeServer


class PFedMe(FedAvg):
    """pFedMe (Dinh et al., NeurIPS 2020). Moreau-envelope personalized FL."""

    emit_two_track_report = True

    def __init__(
        self,
        pfedme_lambda=15.0,
        pfedme_k=5,
        pfedme_eta=0.01,
        pfedme_beta=1.0,
        pfedme_R=None,
        pfedme_uniform_agg=True,
        pfedme_personal_lr=0.0003,
        fedavg_study_name="adamw_nopruner",
    ):
        super().__init__()
        self.pfedme_lambda = float(pfedme_lambda)
        self.pfedme_k = int(pfedme_k)
        self.pfedme_eta = float(pfedme_eta)
        self.pfedme_beta = float(pfedme_beta)
        self.pfedme_R = None if pfedme_R is None else int(pfedme_R)
        self.pfedme_uniform_agg = bool(pfedme_uniform_agg)
        self.pfedme_personal_lr = float(pfedme_personal_lr)
        self.fedavg_study_name = fedavg_study_name

    def _init_federated(self, cfg, df):
        if self.fedavg_study_name is not None:
            self._load_fedavg_global_hypers(cfg)
        super()._init_federated(cfg, df)

    def _load_fedavg_global_hypers(self, cfg):
        """Load lr/wd/round_epochs from a tuned FedAvg run."""
        import json

        from omegaconf import OmegaConf, open_dict

        from utils.meta_writer import results_dir_for

        meta_path = results_dir_for(cfg, "fedavg", self.fedavg_study_name) / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"FedAvg meta.json not found: {meta_path}")
        params = (
            json.loads(meta_path.read_text()).get("best_trial", {}).get("params", {})
        )
        with open_dict(cfg):
            if "lr" in params:
                OmegaConf.update(cfg, "optimizer.lr", float(params["lr"]))
                self.pfedme_personal_lr = float(params["lr"])
            if "weight_decay" in params:
                OmegaConf.update(cfg, "optimizer.weight_decay", float(params["weight_decay"]))
            if "round_epochs" in params:
                OmegaConf.update(cfg, "federated_params.round_epochs", int(params["round_epochs"]))

    def _init_server(self, cfg):
        self.server = PFedMeServer(cfg, train_df=getattr(self, "df", None))

    def _init_client_cls(self):
        super()._init_client_cls()
        self.client_cls = PFedMeClient
        self.client_kwargs["client_cls"] = self.client_cls
        self.client_args.extend(
            [self.pfedme_lambda, self.pfedme_k, self.pfedme_eta,
             self.pfedme_personal_lr, self.pfedme_R]
        )

    def aggregate(self):
        """Beta-relaxed uniform aggregation (Algorithm 1, Line 12)."""
        aggregated_weights = self.server.global_model.state_dict()
        device = self.server.device

        reporting = [
            (i, g) for i, g in enumerate(self.server.client_gradients) if len(g) > 0
        ]
        if not reporting:
            return aggregated_weights

        S = len(reporting)
        if self.pfedme_uniform_agg:
            scales = [self.pfedme_beta / S] * S
        else:
            total = sum(self.client_sizes[i] for i, _ in reporting)
            scales = [
                self.pfedme_beta * (self.client_sizes[i] / total)
                for i, _ in reporting
            ]

        for (i, grad), scale in zip(reporting, scales):
            for key, g in grad.items():
                aggregated_weights[key] = aggregated_weights[key] + g.to(device) * scale
        return aggregated_weights

    def _two_track_final_metric(self, best_val_after, best_val_max3):
        """Optuna objective: median over clients of the max-of-3 val track."""
        import math

        import numpy as np

        vals = [v for v in best_val_max3 if v is not None and not math.isinf(v)]
        return float(np.median(vals)) if vals else 0.0
