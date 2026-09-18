import torch

from ..fedavg.client import Client
from hydra.utils import instantiate
from omegaconf import OmegaConf

import copy
import math
import time

class FedSpeedClient(Client):
    """FedSpeed client (Sun et al., ICLR 2024) — extrapolated-gradient local updates."""

    def __init__(self, *client_args, **client_kwargs):
        base_client_args = client_args[:2]
        self.lambda_param = client_args[2]
        self.alpha = client_args[3]
        self.rho_0 = client_args[4]

        super().__init__(*client_args, **client_kwargs)

        self.client_args = client_args
        self.client_kwargs = client_kwargs

        self.g_hat = None
        self._server_theta = None

    def _init_optimizer(self):
        lr = float(self.cfg.optimizer.lr)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=0.0,
            weight_decay=0.0,
        )

    def create_pipe_commands(self):
        pipe_commands_map = super().create_pipe_commands()
        pipe_commands_map['g_hat'] = self.set_g_hat
        return pipe_commands_map

    def set_g_hat(self, g_hat):
        self.g_hat = {k: v.to(self.device) for k, v in g_hat.items()}

    def train(self):
        self.server_model_state = copy.deepcopy(self.model).state_dict()
        self._snapshot_server_params()

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

    def _snapshot_server_params(self):
        sd = self.server_model_state
        self._server_theta = {
            name: sd[name].detach().to(self.device)
            for name, _ in self.model.named_parameters()
        }

    def _forward_backward(self, batch):
        _, (inputs, targets) = batch
        targets = targets.to(self.device)

        outputs = self._model_forward(inputs)
        if outputs.dim() == 2 and outputs.shape[1] == 1:
            outputs = outputs.squeeze(-1)
            targets = targets.to(outputs.dtype)

        loss = self.criterion(outputs, targets)
        self.optimizer.zero_grad()
        loss.backward()

    def _global_grad_norm(self, named_params):
        sq = 0.0
        for _, p in named_params:
            if p.grad is not None:
                sq += p.grad.detach().pow(2).sum().item()
        return math.sqrt(sq) + 1e-12

    def _maybe_clip(self, named_params):
        if self._max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                [p for _, p in named_params], max_norm=self._max_grad_norm
            )

    def train_fn(self):
        wd = float(self.cfg.optimizer.weight_decay)
        lr = float(self.cfg.optimizer.lr)
        alpha = self.alpha
        lam = self.lambda_param
        rho_0 = self.rho_0

        _mgn = OmegaConf.select(self.cfg, "federated_params.max_grad_norm", default=None)
        self._max_grad_norm = None if _mgn is None else float(_mgn)

        named_params = [
            (name, p) for name, p in self.model.named_parameters()
            if p.requires_grad
        ]

        self.model.train()
        for _ in range(self.cfg.federated_params.round_epochs):
            for batch in self.train_loader:

                self._forward_backward(batch)
                self._maybe_clip(named_params)
                g1 = {
                    name: p.grad.detach().clone()
                    for name, p in named_params
                    if p.grad is not None
                }

                g1_norm = self._global_grad_norm(named_params)
                rho = rho_0 / g1_norm

                with torch.no_grad():
                    for name, p in named_params:
                        if name in g1:
                            p.add_(g1[name], alpha=rho)

                self._forward_backward(batch)
                self._maybe_clip(named_params)
                g2 = {
                    name: p.grad.detach().clone()
                    for name, p in named_params
                    if p.grad is not None
                }

                with torch.no_grad():
                    for name, p in named_params:
                        if name in g1:
                            p.sub_(g1[name], alpha=rho)


                with torch.no_grad():
                    for name, p in named_params:
                        if name not in g1:
                            continue
                        g_tilde = (1.0 - alpha) * g1[name] + alpha * g2[name]
                        prox = (p.detach() - self._server_theta[name]) / lam
                        G = (
                            g_tilde
                            - self.g_hat[name]
                            + prox
                            + wd * p.detach()
                        )
                        p.sub_(G, alpha=lr)
