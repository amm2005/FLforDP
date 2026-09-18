import copy
import time

import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import OmegaConf

from ..fedavg.client import Client

# Norm layers with running stats that still drift on forward when frozen.
_NORM_WITH_STATS = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.SyncBatchNorm,
)


class FedRepClient(Client):
    """FedRep client (Collins et al., ICML 2021): alternating head/body local updates."""

    def __init__(self, *client_args, **client_kwargs):
        base_client_args = client_args[:2]
        super().__init__(*base_client_args, **client_kwargs)
        self.client_args = client_args
        self.fedrep_head_epochs = int(client_args[2])
        self.fedrep_body_epochs = int(client_args[3])
        self.head_param_substrings = list(client_args[4])
        param_names = [n for n, _ in self.model.named_parameters()]
        self.head_keys = [
            n for n in param_names if any(s in n for s in self.head_param_substrings)
        ]
        if not self.head_keys:
            self.head_keys = self._last_linear_param_names()
        if not self.head_keys:
            raise ValueError(
                f"[FedRep] Client {self.rank}: could not identify the head. No "
                f"param matched {self.head_param_substrings} and no nn.Linear was "
                f"found. Set federated_method.head_param_keys to your head's name."
            )
        self.head_keys_set = set(self.head_keys)
        self.head_state = None

    def create_pipe_commands(self):
        pipe_commands_map = super().create_pipe_commands()
        pipe_commands_map["update_model"] = self.load_global_and_head
        pipe_commands_map["fedrep_head"] = self.set_head
        return pipe_commands_map

    def _last_linear_param_names(self):
        last_name = None
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                last_name = name
        if last_name is None:
            return []
        prefix = f"{last_name}." if last_name else ""
        return [n for n, _ in self.model.named_parameters() if n.startswith(prefix)]

    def set_head(self, state):
        if state is None:
            self.head_state = None
        else:
            self.head_state = {k: v.cpu() for k, v in state.items()}

    def load_global_and_head(self, server_state_dict):
        """Load the received global body but overwrite head keys with this client's personal head."""
        new_state = {k: v.to(self.device) for k, v in server_state_dict.items()}
        if self.head_state is not None:
            for k in self.head_keys:
                new_state[k] = self.head_state[k].to(self.device)
        self.model.load_state_dict(new_state)

    def _lock_frozen_norm(self):
        """Freeze frozen BN-family running stats by forcing eval mode and a no-op train()."""
        self._locked_norm = []
        for module in self.model.modules():
            frozen = not any(
                p.requires_grad for p in module.parameters(recurse=False)
            )
            if isinstance(module, _NORM_WITH_STATS) and frozen:
                module.eval()
                module.train = lambda mode=True, _m=module: _m
                self._locked_norm.append(module)

    def _unlock_frozen_norm(self):
        for module in getattr(self, "_locked_norm", []):
            del module.train
        self._locked_norm = []

    def _set_leg(self, train_head):
        for name, param in self.model.named_parameters():
            is_head = name in self.head_keys_set
            param.requires_grad = is_head if train_head else (not is_head)
        self._unlock_frozen_norm()
        if train_head:
            self._lock_frozen_norm()
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = instantiate(self.cfg.optimizer, params=trainable)
        self.trainer.optimizer = self.optimizer

    def train_fn(self, num_epochs):
        max_grad_norm = OmegaConf.select(
            self.cfg, "federated_params.max_grad_norm", default=None
        )
        self.trainer.train(
            train_loader=self.train_loader,
            num_epochs=int(num_epochs),
            max_grad_norm=max_grad_norm,
        )

    def zero_head_grad(self):
        for k in self.head_keys:
            if k in self.grad:
                self.grad[k] = torch.zeros_like(self.grad[k])

    def train(self):
        self.server_model_state = copy.deepcopy(self.model).state_dict()
        start = time.time()

        if self.head_state is None:
            self.head_state = {
                k: self.server_model_state[k].detach().cpu() for k in self.head_keys
            }

        self.val_loss_before, self.val_metrics_before = self.eval_fn()
        self.test_loss_before, self.test_metrics_before = self.eval_fn(
            self.test_loader
        )

        self._set_leg(train_head=True)
        self.train_fn(self.fedrep_head_epochs)

        self._set_leg(train_head=False)
        self.train_fn(self.fedrep_body_epochs)

        self.head_state = {
            k: self.model.state_dict()[k].detach().cpu() for k in self.head_keys
        }

        self.val_loss_after, self.val_metrics_after = self.eval_fn()
        self.test_loss_after, self.test_metrics_after = self.eval_fn(self.test_loader)

        self.get_grad()
        # Zero head keys so only the body reaches aggregation.
        self.zero_head_grad()

        self.result_time = time.time() - start

    def get_communication_content(self):
        result_dict = super().get_communication_content()
        result_dict["fedrep_head"] = {
            k: v.detach().cpu() for k, v in self.head_state.items()
        }
        return result_dict
