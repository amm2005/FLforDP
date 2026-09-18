import copy
import time

import torch
from omegaconf import OmegaConf

from ..fedavg.client import Client


class MimeClient(Client):
    """Mime client (Karimireddy et al., NeurIPS 2021) — local SVRG-corrected Adam steps with frozen global state."""

    def __init__(self, *client_args, **client_kwargs):
        super().__init__(*client_args, **client_kwargs)
        self.mime_m = None
        self.mime_v = None
        self.mime_c = None
        self._server_grad = None

    def create_pipe_commands(self):
        pipe_commands_map = super().create_pipe_commands()
        pipe_commands_map["mime_m"] = self.set_mime_m
        pipe_commands_map["mime_v"] = self.set_mime_v
        pipe_commands_map["mime_c"] = self.set_mime_c
        return pipe_commands_map

    def set_mime_m(self, m):
        self.mime_m = {k: v.to(self.device) for k, v in m.items()}

    def set_mime_v(self, v):
        self.mime_v = {k: v.to(self.device) for k, v in v.items()}

    def set_mime_c(self, c):
        self.mime_c = None if c is None else {k: v.to(self.device) for k, v in c.items()}

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

    def _load_named_params(self, source):
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if name in source:
                    p.copy_(source[name])

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
        self._server_theta = {
            name: self.server_model_state[name].detach().to(self.device)
            for name, _ in self.model.named_parameters()
        }

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
        b1, b2 = float(betas[0]), float(betas[1])
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
                # Replay RNG so g_y and g_x share identical dropout masks (required for SVRG correctness).
                rng_state = torch.get_rng_state()
                cuda_rng_state = (
                    torch.cuda.get_rng_state(self.device)
                    if str(self.device).startswith("cuda")
                    else None
                )
                g_y = self._grad_at(batch, named_params)
                y_cache = {name: p.detach().clone() for name, p in named_params}
                self._load_named_params(self._server_theta)
                torch.set_rng_state(rng_state)
                if cuda_rng_state is not None:
                    torch.cuda.set_rng_state(cuda_rng_state, self.device)
                g_x = self._grad_at(batch, named_params)
                self._load_named_params(y_cache)

                with torch.no_grad():
                    g_i = {}
                    for name, _ in named_params:
                        gi = g_y[name] - g_x[name]
                        if self.mime_c is not None and name in self.mime_c:
                            gi = gi + self.mime_c[name]
                        g_i[name] = gi
                    g_i = self._maybe_clip(g_i)

                    for name, p in named_params:
                        step_dir = (
                            (1.0 - b1) * g_i[name] + b1 * self.mime_m[name]
                        ) / (self.mime_v[name].sqrt() + eps)
                        if wd != 0.0:
                            p.mul_(1.0 - eta * wd)
                        p.sub_(step_dir, alpha=eta)

    def get_communication_content(self):
        result = super().get_communication_content()
        result["mime_server_grad"] = {
            k: v.detach().cpu() for k, v in self._server_grad.items()
        }
        return result
