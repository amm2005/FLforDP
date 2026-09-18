import torch

from ..fedavg.client import Client
from hydra.utils import instantiate

class ScaffoldClient(Client):
    """SCAFFOLD client-side control-variate correction (Karimireddy et al., ICML 2020)."""

    def __init__(self, *client_args, **client_kwargs):
        base_client_args = client_args[:2]
        self.lr = client_args[2]

        super().__init__(*base_client_args, **client_kwargs)

        self.client_args = client_args
        self.client_kwargs = client_kwargs

        self.grad_control = None

    def _init_optimizer(self):
        self.optimizer = instantiate(
            self.cfg.optimizer, params=self.model.parameters(), lr=self.lr
        )

    def create_pipe_commands(self):
        pipe_commands_map = super().create_pipe_commands()
        pipe_commands_map["grad_control"] = self.set_grad_control
        return pipe_commands_map

    def set_grad_control(self, grad_control):
        self.grad_control = {k: v.cpu() for k, v in grad_control.items()}

    def _add_grad_control(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.add_(self.grad_control[name].to(param.device))

    def train_fn(self):
        from omegaconf import OmegaConf
        max_grad_norm = OmegaConf.select(
            self.cfg, "federated_params.max_grad_norm", default=None
        )
        self.trainer.train(
            train_loader=self.train_loader,
            num_epochs=self.cfg.federated_params.round_epochs,
            after_optimizer_step=self._add_grad_control,
            max_grad_norm=max_grad_norm,
        )
