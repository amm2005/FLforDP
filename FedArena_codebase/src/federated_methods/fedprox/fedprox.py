from ..fedavg.fedavg import FedAvg
from .fedprox_client import FedProxClient


class FedProx(FedAvg):
    """FedProx (Li et al., MLSys 2020): FedAvg with a proximal term for heterogeneous clients."""

    def __init__(self, fed_prox_lambda):
        super().__init__()
        self.fed_prox_lambda = fed_prox_lambda

    def _init_client_cls(self):
        super()._init_client_cls()
        self.client_cls = FedProxClient
        self.client_kwargs["client_cls"] = self.client_cls
        self.client_args.append(self.fed_prox_lambda)
