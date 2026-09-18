import torch

from ..fedavg.client import Client
from hydra.utils import instantiate

import copy
import time

class FedDynClient(Client):
    """FedDyn client-side regularizer (Acar et al., ICLR 2021)."""

    def __init__(self, *client_args, **client_kwargs):
        base_client_args = client_args[:2]
        self.alpha_list = client_args[2]
        rank = client_kwargs['rank']

        self.alpha_coef_k = self.alpha_list[rank]
        super().__init__(*base_client_args, **client_kwargs)

        self.client_args = client_args
        self.client_kwargs = client_kwargs

        self.h_k_scaled = None
        self._server_theta_flat = None
        self._h_k_scaled_flat = None

    def _init_optimizer(self):
        self.optimizer = instantiate(
            self.cfg.optimizer,
            params=self.model.parameters(),
        )

    def create_pipe_commands(self):
        pipe_commands_map = super().create_pipe_commands()
        pipe_commands_map['h_k_scaled'] = self.set_h_k_scaled
        pipe_commands_map['alpha_coef_k'] = self.set_alpha_coef_k
        return pipe_commands_map

    def set_h_k_scaled(self, h_k_scaled):
        self.h_k_scaled = {k: v.to(self.device) for k, v in h_k_scaled.items()}
        flats = [
            self.h_k_scaled[name].reshape(-1)
            for name, _ in self.model.named_parameters()
        ]
        self._h_k_scaled_flat = torch.cat(flats, dim=0).detach()

    def set_alpha_coef_k(self, alpha_coef_k):
        self.alpha_coef_k = float(alpha_coef_k)

    def _refresh_server_theta_flat(self):
        server_model = self.server_model_state
        flats = [server_model[name].reshape(-1).to(self.device) for name, _ in self.model.named_parameters()]
        self._server_theta_flat = torch.cat(flats, dim=0).detach()

    def train(self):
        self.server_model_state = copy.deepcopy(self.model).state_dict()
        self._refresh_server_theta_flat()

        start = time.time()
        self.val_loss_before, self.val_metrics_before = self.eval_fn()
        self.test_loss_before, self.test_metrics_before = self.eval_fn(
            self.test_loader
        )
        self.train_fn()
        self.val_loss_after, self.val_metrics_after = self.eval_fn()
        self.test_loss_after, self.test_metrics_after = self.eval_fn(
            self.test_loader
        )
        self.get_grad()
        self.result_time = time.time() - start

    def get_loss_value(self, outputs, targets):
        base_loss = super().get_loss_value(outputs, targets)

        theta_flat = torch.cat(
            [p.reshape(-1) for p in self.model.parameters()], dim=0
        )
        linear = self.alpha_coef_k * torch.dot(
            theta_flat, -self._server_theta_flat + self._h_k_scaled_flat
        )
        quadratic = 0.5 * self.alpha_coef_k * torch.dot(theta_flat, theta_flat)
        return base_loss + linear + quadratic
