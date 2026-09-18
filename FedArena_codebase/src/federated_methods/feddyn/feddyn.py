import copy

import torch

from ..fedavg.fedavg import FedAvg
from .feddyn_client import FedDynClient

class FedDyn(FedAvg):
    """FedDyn (Acar et al., ICLR 2021) — dynamic-regularization federated learning."""

    def __init__(self, fed_dyn_alpha=0.01):
        super().__init__()
        self.fed_dyn_alpha = float(fed_dyn_alpha)

        self.num_clients = None
        self.client_weights = None
        self.alpha_k = None
        self.h_scaled = None

    def _init_federated(self, cfg, df):
        self.num_clients = cfg.federated_params.amount_of_clients
        self._compute_client_weights(df)
        self.alpha_k = [self.fed_dyn_alpha / w for w in self.client_weights]

        super()._init_federated(cfg, df)

    def _compute_client_weights(self, df):
        sizes = []
        data = df.orig_data
        has_pool = "server_pool" in data.columns
        has_phase = "phase" in data.columns
        for rank in range(self.num_clients):
            mask = data["client"] == rank
            if has_pool:
                mask &= ~data["server_pool"]
            if has_phase:
                mask &= data["phase"] == "train"
            sizes.append(int(mask.sum()))
        mean_n = sum(sizes) / len(sizes)
        self.client_weights = [n / mean_n for n in sizes]

    def _zero_like_named_params(self):
        return {name: torch.zeros_like(param.detach()).cpu() for name, param in self.server.global_model.named_parameters()}

    def begin_train(self, round_callback=None):
        self.h_scaled = [self._zero_like_named_params() for _ in range(self.num_clients)]
        return super().begin_train(round_callback=round_callback)

    def _init_client_cls(self):
        super()._init_client_cls()
        self.client_cls = FedDynClient
        self.client_kwargs['client_cls'] = self.client_cls

        self.client_args.append(self.alpha_k)

    def get_communication_content(self, rank):
        content = super().get_communication_content(rank)
        content['h_k_scaled'] = {k: v.detach().cpu() for k, v in self.h_scaled[rank].items()}
        content['alpha_coef_k'] = float(self.alpha_k[rank])
        return content

    def parse_communication_content(self, client_result):
        super().parse_communication_content(client_result)
        rank = client_result["rank"]
        delta = client_result['grad']

        for key in self.h_scaled[rank].keys():
            if key in delta:
                self.h_scaled[rank][key] = (self.h_scaled[rank][key] + delta[key].detach().cpu())

    def aggregate(self):
        aggregated = self.server.global_model.state_dict()
        num_client_grads = len(self.server.client_gradients)
        device = self.server.device

        for client_grad in self.server.client_gradients:
            for key, grad in client_grad.items():
                aggregated[key] = aggregated[key] + grad.to(device) / num_client_grads

        for name in self.h_scaled[0].keys():
            mean_h = torch.zeros_like(self.h_scaled[0][name])
            for rank in range(self.num_clients):
                mean_h = mean_h + self.h_scaled[rank][name]
            mean_h = mean_h / self.num_clients
            aggregated[name] = aggregated[name] + mean_h.to(device)

        return aggregated
