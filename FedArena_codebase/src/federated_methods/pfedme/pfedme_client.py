import copy
import time

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from ..fedavg.client import Client


class PFedMeClient(Client):
    """pFedMe client (Dinh et al., NeurIPS 2020): Moreau-envelope personalization."""

    def __init__(self, *client_args, **client_kwargs):
        base_client_args = client_args[:2]  # cfg, df
        super().__init__(*base_client_args, **client_kwargs)
        self.client_args = client_args
        self.pfedme_lambda = float(client_args[2])
        self.pfedme_k = int(client_args[3])
        self.pfedme_eta = float(client_args[4])
        self.pfedme_personal_lr = float(client_args[5])
        self.pfedme_R = client_args[6]
        # Prox term on during inner steps, off during eval.
        self.do_proximity = False
        self.w_local_state = None
        self.theta_state = None
        self._batch_iter = None

    def _num_outer_rounds(self):
        """Number of [K inner steps + one w_k update] blocks per round."""
        if self.pfedme_R is not None:
            return int(self.pfedme_R)
        return int(self.cfg.federated_params.round_epochs) * max(
            1, len(self.train_loader)
        )

    def _next_batch(self):
        if self._batch_iter is None:
            self._batch_iter = iter(self.train_loader)
        try:
            return next(self._batch_iter)
        except StopIteration:
            self._batch_iter = iter(self.train_loader)
            return next(self._batch_iter)

    def _set_optimizer(self, lr, weight_decay=None):
        kwargs = {"params": self.model.parameters(), "lr": lr}
        if weight_decay is not None:
            kwargs["weight_decay"] = weight_decay
        self.optimizer = instantiate(self.cfg.optimizer, **kwargs)
        self.trainer.optimizer = self.optimizer

    def get_loss_value(self, outputs, targets):
        loss = super().get_loss_value(outputs, targets)
        if self.do_proximity:
            # Moreau prox: 0.5*lambda*||theta - w_k||^2 on trainable params.
            proximity = (
                0.5
                * self.pfedme_lambda
                * sum(
                    (p.float() - self.w_local_state[name].float()).norm() ** 2
                    for name, p in self.model.named_parameters()
                )
            )
            loss += proximity
        return loss

    def _outer_step(self):
        """w_k <- w_k - eta*lambda*(w_k - theta_k)."""
        theta = self.model.state_dict()
        step = self.pfedme_eta * self.pfedme_lambda
        with torch.no_grad():
            for key, w in self.w_local_state.items():
                t = theta[key]
                if w.is_floating_point():
                    w.sub_((w - t.detach()) * step)
                else:
                    w.copy_(t)

    def train_fn(self, num_epochs=None):
        """Run R blocks of K inner steps on one minibatch plus one outer step."""
        max_grad_norm = OmegaConf.select(
            self.cfg, "federated_params.max_grad_norm", default=None
        )
        for _ in range(self._num_outer_rounds()):
            batch = self._next_batch()
            self.trainer.train(
                train_loader=[batch] * self.pfedme_k,
                num_epochs=1,
                max_grad_norm=max_grad_norm,
            )
            self._outer_step()

    def train(self):
        self.server_model_state = copy.deepcopy(self.model).state_dict()
        start = time.time()

        self.val_loss_before, self.val_metrics_before = self.eval_fn()
        self.test_loss_before, self.test_metrics_before = self.eval_fn(
            self.test_loader
        )

        # Seed the local copies and the model at the received global weights.
        self.w_local_state = {
            k: v.detach().clone() for k, v in self.server_model_state.items()
        }
        self._batch_iter = None
        self.do_proximity = True
        self._set_optimizer(self.pfedme_personal_lr, weight_decay=0.0)
        self.train_fn()
        self.do_proximity = False

        # Evaluate the personalized model (theta_k) before loading w_k.
        self.val_loss_after, self.val_metrics_after = self.eval_fn()
        self.test_loss_after, self.test_metrics_after = self.eval_fn(
            self.test_loader
        )
        self.theta_state = {
            k: v.detach().cpu() for k, v in self.model.state_dict().items()
        }

        # Load w_k for aggregation (delta = w_k - w^t).
        self.model.load_state_dict(
            {k: v.to(self.device) for k, v in self.w_local_state.items()}
        )
        self.val_loss_wtk, self.val_metrics_wtk = self.eval_fn()
        self.test_loss_wtk, self.test_metrics_wtk = self.eval_fn(self.test_loader)
        self.get_grad()
        self.result_time = time.time() - start

    def get_communication_content(self):
        result_dict = super().get_communication_content()
        # Send theta_k (CPU) up so the server can persist personalized models.
        result_dict["pfedme_theta"] = self.theta_state
        result_dict["client_val_wtk"] = (
            self.val_metrics_wtk,
            self.val_loss_wtk,
            len(self.valid_df),
        )
        result_dict["client_test_wtk"] = (
            self.test_metrics_wtk,
            self.test_loss_wtk,
            len(self.test_df),
        )
        return result_dict
