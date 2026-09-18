import torch

from ..fedavg.fedavg import FedAvg


class FedYogi(FedAvg):
    """FedYogi (Reddi et al., 2020) — adaptive server step over aggregated client deltas."""

    def __init__(self, server_lr=0.01, beta1=0.9, beta2=0.99, tau=1e-3):
        super().__init__()
        self.server_lr = float(server_lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.tau = float(tau)

        self.m = None
        self.v = None

    def begin_train(self, round_callback=None):
        self.m = None
        self.v = None
        return super().begin_train(round_callback=round_callback)

    def aggregate(self):
        device = self.server.device
        aggregated = self.server.global_model.state_dict()

        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in aggregated.items():
                if torch.is_floating_point(val):
                    self.m[key] = torch.zeros_like(val, device=device)
                    self.v[key] = torch.full_like(val, self.tau ** 2, device=device)

        total_size = sum(self.client_sizes)
        weights = [size / total_size for size in self.client_sizes]
        delta = {}
        for i, client_grad in enumerate(self.server.client_gradients):
            for key, grad in client_grad.items():
                contrib = grad.to(device) * weights[i]
                delta[key] = contrib if key not in delta else delta[key] + contrib

        b1, b2, tau, lr = self.beta1, self.beta2, self.tau, self.server_lr
        for key, val in aggregated.items():
            d = delta.get(key)
            if d is None:
                continue
            base = val.to(device)
            if key in self.v:
                self.m[key] = b1 * self.m[key] + (1.0 - b1) * d
                d2 = d * d
                self.v[key] = self.v[key] - (1.0 - b2) * d2 * torch.sign(self.v[key] - d2)
                aggregated[key] = base + lr * self.m[key] / (self.v[key].sqrt() + tau)
            else:
                aggregated[key] = base + d
        return aggregated
