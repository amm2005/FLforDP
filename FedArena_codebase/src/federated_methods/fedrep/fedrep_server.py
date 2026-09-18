import numpy as np
import pandas as pd
import torch

from ..fedavg.server import Server


class FedRepServer(Server):
    """FedRep server (Collins et al., ICML 2021): persists per-client heads and tunes on the personalized model."""

    def __init__(self, cfg, train_df=None, save_state=True):
        super().__init__(cfg, train_df=train_df)
        n = cfg.federated_params.amount_of_clients
        self.head_states = [None] * n
        self.pers_val_metrics = [None] * n
        self.save_state = bool(save_state)

    def set_client_result(self, client_result):
        super().set_client_result(client_result)
        rank = client_result["rank"]
        if "fedrep_head" in client_result:
            self.head_states[rank] = client_result["fedrep_head"]
        self.pers_val_metrics[rank] = client_result.get("client_val_after")

    def _mean_head_state(self):
        """Uniform mean of trained client heads (None until the first round)."""
        heads = [h for h in self.head_states if h is not None]
        if not heads:
            return None
        return {
            k: torch.stack([h[k].float() for h in heads]).mean(dim=0)
            for k in heads[0]
        }

    def _eval_with_mean_head(self, eval_call):
        """Evaluate the global model with the mean head swapped in, then restore the original."""
        mean_head = self._mean_head_state()
        if mean_head is None:
            print("[FedRep] No trained client heads yet; global eval uses the init head.")
            eval_call()
            return
        state = self.global_model.state_dict()
        orig_head = {k: state[k].clone() for k in mean_head}
        self.global_model.load_state_dict(mean_head, strict=False)
        n_heads = sum(h is not None for h in self.head_states)
        print(f"[FedRep] Global eval uses body + uniform mean of {n_heads} client heads.")
        try:
            eval_call()
        finally:
            self.global_model.load_state_dict(orig_head, strict=False)

    def _save_fedrep_state(self):
        if not self.save_state:
            return
        path = f"{self.model_path}_fedrep_state.pt"
        torch.save(
            {
                "global_model": {
                    k: v.detach().cpu()
                    for k, v in self.global_model.state_dict().items()
                },
                "head_states": self.head_states,
            },
            path,
        )

    def test_global_model(self):
        self._eval_with_mean_head(super().test_global_model)
        self._save_fedrep_state()

    def validate_global_model(self):
        self._eval_with_mean_head(super().validate_global_model)
        # Clear server-val so tuning falls back to the personalized model metrics.
        self.current_server_val_metrics = {}

    def save_best_model(self, round):
        super().save_best_model(round)

        valid = [m for m in self.pers_val_metrics if m is not None]
        if not valid:
            return

        client_metrics_dfs = [m[0] for m in valid]
        val_losses = [m[1] for m in valid]
        val_len_dfs = [m[2] for m in valid]
        total_len = sum(val_len_dfs)
        weights = [
            (v / total_len if total_len > 0 else 1.0 / len(valid))
            for v in val_len_dfs
        ]
        metrics_names = client_metrics_dfs[0].index

        if self.metric_aggregation == "uniform":
            agg_val_loss = np.mean(val_losses)
            agg_metrics = pd.concat(client_metrics_dfs).groupby(level=0).mean()
        else:  # "weighted"
            agg_val_loss = np.sum([loss * w for loss, w in zip(val_losses, weights)])
            agg_metrics = sum(w * m for w, m in zip(weights, client_metrics_dfs))
        agg_metrics = agg_metrics.reindex(metrics_names)

        self.current_val_metrics = {"loss": agg_val_loss}
        for idx in agg_metrics.index:
            self.current_val_metrics[idx] = agg_metrics.loc[idx].mean()
