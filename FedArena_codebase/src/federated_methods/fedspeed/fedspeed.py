import copy

import torch

from ..fedavg.fedavg import FedAvg
from .fedspeed_client import FedSpeedClient

class FedSpeed(FedAvg):
    """FedSpeed (Sun et al., ICLR 2024) — gradient-perturbation-based federated optimization."""

    def __init__(self, lambda_param=0.01, alpha=0.9, rho_0=0.01):
        super().__init__()
        self.lambda_param = float(lambda_param)
        self.alpha = float(alpha)
        self.rho_0 = float(rho_0)

        self.num_clients = None
        self.g_hat = None

    def _init_federated(self, cfg, df):
        self.num_clients = cfg.federated_params.amount_of_clients
        super()._init_federated(cfg, df)

    def _zero_like_named_params(self):
        return {name: torch.zeros_like(param.detach()).cpu() for name, param in self.server.global_model.named_parameters()}

    def begin_train(self, round_callback=None):
        self.g_hat = [self._zero_like_named_params() for _ in range(self.num_clients)]
        return super().begin_train(round_callback=round_callback)

    def _init_client_cls(self):
        super()._init_client_cls()
        self.client_cls = FedSpeedClient
        self.client_kwargs['client_cls'] = self.client_cls
        self.client_args.append(self.lambda_param)
        self.client_args.append(self.alpha)
        self.client_args.append(self.rho_0)

    def get_communication_content(self, rank):
        content = super().get_communication_content(rank)
        content['g_hat'] = {k: v.detach().cpu() for k, v in self.g_hat[rank].items()}
        return content

    def parse_communication_content(self, client_result):
        content = super().parse_communication_content(client_result)

        rank = client_result["rank"]
        delta = client_result['grad']

        for key in self.g_hat[rank].keys():
            if key in delta:
                self.g_hat[rank][key] = (self.g_hat[rank][key] - delta[key].detach().cpu() / self.lambda_param)

    def aggregate(self):
        aggregated = self.server.global_model.state_dict()
        num_client_grads = len(self.server.client_gradients)
        device = self.server.device

        for client_grad in self.server.client_gradients:
            for key, grad in client_grad.items():
                aggregated[key] = aggregated[key] + grad.to(device) / num_client_grads

        for name in self.g_hat[0].keys():
            mean_g_hat = torch.zeros_like(self.g_hat[0][name])
            for rank in range(self.num_clients):
                mean_g_hat = mean_g_hat + self.g_hat[rank][name]
            mean_g_hat = mean_g_hat / self.num_clients
            aggregated[name] = aggregated[name] - self.lambda_param * mean_g_hat.to(device)
        return aggregated
