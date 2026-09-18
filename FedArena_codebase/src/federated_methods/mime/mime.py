import torch

from ..fedavg.fedavg import FedAvg
from .mime_client import MimeClient


class MimeMethod(FedAvg):
    """Mime (Karimireddy et al., NeurIPS 2021) — frozen global Adam state with SVRG control variate for cross-device FL."""

    def __init__(self):
        super().__init__()
        self.m = None
        self.v = None
        self.c = None
        self._server_grads = None

    def _adam_hparams(self):
        betas = self.cfg.optimizer.betas
        b1, b2 = float(betas[0]), float(betas[1])
        eps = float(self.cfg.optimizer.eps)
        return b1, b2, eps

    def begin_train(self, round_callback=None):
        self.m = None
        self.v = None
        self.c = None
        self._server_grads = [None] * self.cfg.federated_params.amount_of_clients
        return super().begin_train(round_callback=round_callback)

    def _init_client_cls(self):
        super()._init_client_cls()
        self.client_cls = MimeClient
        self.client_kwargs["client_cls"] = self.client_cls

    def _frozen_state_cpu(self):
        """Serialize s=(m, v) as {name: cpu_tensor} dicts; round 0 sends m=0, v=1."""
        if self.m is None:
            params = list(self.server.global_model.named_parameters())
            m_zero = {name: torch.zeros_like(p.detach()).cpu() for name, p in params}
            v_one = {name: torch.ones_like(p.detach()).cpu() for name, p in params}
            return m_zero, v_one
        m_cpu = {k: v.detach().cpu() for k, v in self.m.items()}
        v_cpu = {k: v.detach().cpu() for k, v in self.v.items()}
        return m_cpu, v_cpu

    def get_communication_content(self, rank):
        content = super().get_communication_content(rank)
        m_cpu, v_cpu = self._frozen_state_cpu()
        content["mime_m"] = m_cpu
        content["mime_v"] = v_cpu
        content["mime_c"] = (
            None if self.c is None else {k: v.detach().cpu() for k, v in self.c.items()}
        )
        return content

    def parse_communication_content(self, client_result):
        super().parse_communication_content(client_result)
        rank = client_result["rank"]
        self._server_grads[rank] = {
            k: v.detach().cpu() for k, v in client_result["mime_server_grad"].items()
        }

    def aggregate(self):
        device = self.server.device
        aggregated = self.server.global_model.state_dict()

        total_size = sum(self.client_sizes)
        weights = [size / total_size for size in self.client_sizes]
        for i in range(len(self.server.client_gradients)):
            for key, grad in self.server.client_gradients[i].items():
                aggregated[key] = aggregated[key] + grad.to(device) * weights[i]

        gbar = {}
        for i, server_grad in enumerate(self._server_grads):
            if server_grad is None:
                continue
            for name, g in server_grad.items():
                contrib = g.to(device) * weights[i]
                gbar[name] = contrib if name not in gbar else gbar[name] + contrib

        if self.m is None:
            self.m = {name: torch.zeros_like(g) for name, g in gbar.items()}
            self.v = {name: torch.zeros_like(g) for name, g in gbar.items()}

        b1, b2, _ = self._adam_hparams()
        for name, g in gbar.items():
            self.m[name] = (1.0 - b1) * g + b1 * self.m[name]
            self.v[name] = (1.0 - b2) * (g * g) + b2 * self.v[name]

        self.c = {name: g.clone() for name, g in gbar.items()}

        return aggregated
