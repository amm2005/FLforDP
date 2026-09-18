import copy
import time

import torch
from omegaconf import OmegaConf

from ..fedavg.client import Client


class MimeLiteClient(Client):
    """MimeLite client (Karimireddy et al., NeurIPS 2021) — local Adam steps with frozen global state, no control variate."""

    def __init__(self, *client_args, **client_kwargs):
        super().__init__(*client_args, **client_kwargs)
        self.mimelite_m = None
        self.mimelite_v = None
        self._server_grad = None

    def create_pipe_commands(self):
        pipe_commands_map = super().create_pipe_commands()
        pipe_commands_map["mimelite_m"] = self.set_mimelite_m
        pipe_commands_map["mimelite_v"] = self.set_mimelite_v
        return pipe_commands_map

    def set_mimelite_m(self, m):
        self.mimelite_m = {k: v.to(self.device) for k, v in m.items()}

    def set_mimelite_v(self, v):
        self.mimelite_v = {k: v.to(self.device) for k, v in v.items()}

    def _unpack_batch(self, batch):
        _, (inputs, targets) = batch
        return inputs, targets.to(self.device)

    def _grad_at(self, batch, named_params):
        """One forward+backward at the current params; returns cloned {name: grad}."""
        inputs, targets = self._unpack_batch(batch)
        outputs = self._model_forward(inputs)
        if outputs.dim() == 2 and outputs.shape[1] == 1:
            outputs = outputs.squeeze(-1)
            targets = targets.to(outputs.dtype)
        loss = self.criterion(outputs, targets)
        for _, p in named_params:
            p.grad = None
        loss.backward()
        return {
            name: p.grad.detach().clone()
            for name, p in named_params
            if p.grad is not None
        }

    def _full_batch_server_grad(self, named_params):
        """Dataset-mean gradient over the whole local train set at the server point."""
        was_training = self.model.training
        self.model.eval()
        accum = {name: torch.zeros_like(p) for name, p in named_params}
        n_total = 0
        for batch in self.train_loader:
            inputs, targets = self._unpack_batch(batch)
            outputs = self._model_forward(inputs)
            if outputs.dim() == 2 and outputs.shape[1] == 1:
                outputs = outputs.squeeze(-1)
                targets = targets.to(outputs.dtype)
            bs = targets.shape[0]
            loss = self.criterion(outputs, targets)
            for _, p in named_params:
                p.grad = None
            loss.backward()
            for name, p in named_params:
                if p.grad is not None:
                    accum[name] += p.grad.detach() * bs
            n_total += bs
        if was_training:
            self.model.train()
        return {name: accum[name] / max(n_total, 1) for name in accum}

    def _maybe_clip(self, g_i):
        """Clip g_i by global L2 norm to max_grad_norm if configured."""
        if self._max_grad_norm is not None:
            total = torch.norm(
                torch.stack([g.detach().norm(2) for g in g_i.values()]), 2
            )
            clip_coef = self._max_grad_norm / (total + 1e-6)
            if clip_coef < 1.0:
                for name in g_i:
                    g_i[name] = g_i[name] * clip_coef
        return g_i

    def train(self):
        self.server_model_state = copy.deepcopy(self.model).state_dict()

        start = time.time()
        self.val_loss_before, self.val_metrics_before = self.eval_fn()
        self.test_loss_before, self.test_metrics_before = self.eval_fn(self.test_loader)
        self.train_fn()
        self.val_loss_after, self.val_metrics_after = self.eval_fn()
        self.test_loss_after, self.test_metrics_after = self.eval_fn(self.test_loader)
        self.get_grad()
        self.result_time = time.time() - start

    def train_fn(self):
        eta = float(self.cfg.optimizer.lr)
        wd = float(self.cfg.optimizer.weight_decay)
        betas = self.cfg.optimizer.betas
        b1, _ = float(betas[0]), float(betas[1])
        eps = float(self.cfg.optimizer.eps)

        _mgn = OmegaConf.select(
            self.cfg, "federated_params.max_grad_norm", default=None
        )
        self._max_grad_norm = None if _mgn is None else float(_mgn)

        named_params = [
            (name, p) for name, p in self.model.named_parameters() if p.requires_grad
        ]

        self.model.train()
        self._server_grad = self._full_batch_server_grad(named_params)

        for _ in range(self.cfg.federated_params.round_epochs):
            for batch in self.train_loader:
                g_i = self._grad_at(batch, named_params)

                with torch.no_grad():
                    g_i = self._maybe_clip(g_i)

                    for name, p in named_params:
                        step_dir = (
                            (1.0 - b1) * g_i[name] + b1 * self.mimelite_m[name]
                        ) / (self.mimelite_v[name].sqrt() + eps)
                        if wd != 0.0:
                            p.mul_(1.0 - eta * wd)
                        p.sub_(step_dir, alpha=eta)

    def get_communication_content(self):
        result = super().get_communication_content()
        result["mimelite_server_grad"] = {
            k: v.detach().cpu() for k, v in self._server_grad.items()
        }
        return result
