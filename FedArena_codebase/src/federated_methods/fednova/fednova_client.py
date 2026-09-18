import torch

from ..fedavg.client import Client
from hydra.utils import instantiate

import copy
import time

class FedNovaClient(Client):
    """FedNova client (Wang et al., ICLR 2021) — counts local optimizer steps."""

    def __init__(self, *client_args, **client_kwargs):
        super().__init__(*client_args, **client_kwargs)
        self.tau_i = 0

    def _increment_tau(self):
        self.tau_i += 1

    def train_fn(self):
        from omegaconf import OmegaConf
        max_grad_norm = OmegaConf.select(
            self.cfg, "federated_params.max_grad_norm", default=None
        )
        self.tau_i = 0
        self.trainer.train(
            train_loader=self.train_loader,
            num_epochs=self.cfg.federated_params.round_epochs,
            after_optimizer_step=self._increment_tau,
            max_grad_norm=max_grad_norm,
        )

    def get_communication_content(self):
        result_dict = super().get_communication_content()
        result_dict["tau"] = int(self.tau_i)
        return result_dict
