import copy

import torch

from ..fedavg.fedavg import FedAvg
from .scaffold_client import ScaffoldClient

import math


class Scaffold(FedAvg):
    """SCAFFOLD (Karimireddy et al., ICML 2020) — stochastic controlled averaging."""

    def __init__(self, local_lr=3e-4):
        super().__init__()
        self.global_lr = None
        self.local_lr = local_lr
        self.num_clients = None
        self.global_control = None
        self.clients_control = None
        self.clients_delta_control = None

    def _init_federated(self, cfg, df):
        super()._init_federated(cfg, df)
        self.num_clients = cfg.federated_params.amount_of_clients
        self.global_lr = 1.0

    def _init_client_cls(self):
        super()._init_client_cls()
        self.client_cls = ScaffoldClient
        self.client_kwargs["client_cls"] = self.client_cls
        self.client_args.append(self.local_lr)

    def _zero_control_like_model(self):
        control = {}
        for name, param in self.server.global_model.named_parameters():
            control[name] = torch.zeros_like(param.detach()).cpu()
        return control

    def _init_controls(self):
        self.global_control = self._zero_control_like_model()
        self.clients_control = [
            copy.deepcopy(self.global_control) for _ in range(self.num_clients)
        ]
        self.clients_delta_control = [
            copy.deepcopy(self.global_control) for _ in range(self.num_clients)
        ]

    def _get_local_steps(self, rank):
        n_k = self.client_sizes[rank]
        batch_size = self.cfg.training_params.batch_size
        round_epochs = self.cfg.federated_params.round_epochs
        num_batches = math.ceil(n_k / batch_size)
        return num_batches * round_epochs

    def begin_train(self, round_callback=None):
        self._init_controls()
        return super().begin_train(round_callback=round_callback)

    def _get_client_grad_control(self, rank):
        grad_control = {}
        for key in self.clients_control[rank].keys():
            grad_control[key] = self.local_lr * (
                self.clients_control[rank][key] - self.global_control[key]
            )
        return grad_control

    def _update_client_control(self, rank, state):
        updated_control = {}
        K = self._get_local_steps(rank)
        coef = -1 / (K * self.local_lr)

        for key in self.clients_control[rank].keys():
            updated_control[key] = (
                self.clients_control[rank][key]
                - self.global_control[key]
                + coef * state[key].detach().cpu()
            )

        self.clients_delta_control[rank] = {
            key: updated_control[key] - self.clients_control[rank][key]
            for key in updated_control.keys()
        }
        self.clients_control[rank] = updated_control

    def get_communication_content(self, rank):
        content = super().get_communication_content(rank)
        content["grad_control"] = {
            k: v.detach().cpu()
            for k, v in self._get_client_grad_control(rank).items()
        }
        return content

    def parse_communication_content(self, client_result):
        super().parse_communication_content(client_result)
        rank = client_result["rank"]
        self._update_client_control(rank, client_result["grad"])

    def _update_global_control(self):
        for key in self.global_control.keys():
            delta_sum = torch.zeros_like(self.global_control[key])
            for delta_control in self.clients_delta_control:
                delta_sum += delta_control[key].to(delta_sum.device)
            self.global_control[key] = self.global_control[key] + (
                delta_sum / self.num_clients
            )

        zero_control = {
            k: torch.zeros_like(v) for k, v in self.global_control.items()
        }
        self.clients_delta_control = [
            copy.deepcopy(zero_control) for _ in range(self.num_clients)
        ]

    def aggregate(self):
        aggregated_weights = self.server.global_model.state_dict()
        num_client_grads = len(self.server.client_gradients)

        for client_grad in self.server.client_gradients:
            for key, grad in client_grad.items():
                aggregated_weights[key] = aggregated_weights[key] + (
                    self.global_lr * grad.to(self.server.device) / num_client_grads
                )

        self._update_global_control()
        return aggregated_weights
