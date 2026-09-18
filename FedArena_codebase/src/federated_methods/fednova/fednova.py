import copy

import torch

from ..fedavg.fedavg import FedAvg
from .fednova_client import FedNovaClient
class FedNova(FedAvg):
    """FedNova (Wang et al., ICLR 2021) — normalized averaging with effective step weighting."""

    def __init__(self):
        super().__init__()

        self.num_clients = None
        self.client_tau = None

    def _init_federated(self, cfg, df):
        super()._init_federated(cfg, df)
        self.num_clients = cfg.federated_params.amount_of_clients
        self.client_tau = [0] * self.num_clients

    def _init_client_cls(self):
        super()._init_client_cls()
        self.client_cls = FedNovaClient
        self.client_kwargs['client_cls'] = self.client_cls

    def parse_communication_content(self, client_result):
        super().parse_communication_content(client_result)
        rank = client_result["rank"]
        self.client_tau[rank] = int(client_result['tau'])

    def aggregate(self):
        aggregated_weights = self.server.global_model.state_dict()
        m = self.num_clients

        total_n = sum(self.client_sizes)
        p = [n / total_n for n in self.client_sizes]

        tau_eff = sum(p[i] * self.client_tau[i] for i in range(m))

        for i in range(m):
            coef = tau_eff * p[i] / self.client_tau[i]
            for key, delta in self.server.client_gradients[i].items():
                aggregated_weights[key] = aggregated_weights[key] + delta.to(self.server.device) * coef
        return aggregated_weights
