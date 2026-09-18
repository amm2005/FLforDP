import copy
import time

from hydra.utils import instantiate

from ..fedavg.client import Client


class DittoClient(Client):
    """Ditto client (Li et al., ICML 2021): global leg then prox-regularized personal leg."""

    def __init__(self, *client_args, **client_kwargs):
        base_client_args = client_args[:2]
        super().__init__(*base_client_args, **client_kwargs)
        self.client_args = client_args
        self.ditto_lambda = float(client_args[2])
        self.ditto_personal_lr = float(client_args[3])
        self.ditto_personal_epochs = int(client_args[4])
        self.do_proximity = False
        self.v_k_state = None

    def create_pipe_commands(self):
        pipe_commands_map = super().create_pipe_commands()
        pipe_commands_map["ditto_v"] = self.set_v_k
        return pipe_commands_map

    def set_v_k(self, state):
        if state is None:
            self.v_k_state = None
        else:
            self.v_k_state = {k: v.cpu() for k, v in state.items()}

    def _set_optimizer(self, lr, weight_decay=None):
        kwargs = {"params": self.model.parameters(), "lr": lr}
        if weight_decay is not None:
            kwargs["weight_decay"] = weight_decay
        self.optimizer = instantiate(self.cfg.optimizer, **kwargs)
        self.trainer.optimizer = self.optimizer

    def get_loss_value(self, outputs, targets):
        loss = super().get_loss_value(outputs, targets)
        if self.do_proximity:
            proximity = (
                0.5
                * self.ditto_lambda
                * sum(
                    (p.float() - self.server_model_state[name].float()).norm() ** 2
                    for name, p in self.model.named_parameters()
                )
            )
            loss += proximity
        return loss

    def train_fn(self, num_epochs=None):
        from omegaconf import OmegaConf

        max_grad_norm = OmegaConf.select(
            self.cfg, "federated_params.max_grad_norm", default=None
        )
        epochs = (
            int(num_epochs)
            if num_epochs is not None
            else self.cfg.federated_params.round_epochs
        )
        self.trainer.train(
            train_loader=self.train_loader,
            num_epochs=epochs,
            max_grad_norm=max_grad_norm,
        )

    def train(self):
        self.server_model_state = copy.deepcopy(self.model).state_dict()
        start = time.time()

        self.do_proximity = False
        self._set_optimizer(float(self.cfg.optimizer.lr))
        self.val_loss_before, self.val_metrics_before = self.eval_fn()
        self.test_loss_before, self.test_metrics_before = self.eval_fn(
            self.test_loader
        )
        self.train_fn()
        self.val_loss_wtk, self.val_metrics_wtk = self.eval_fn()
        self.test_loss_wtk, self.test_metrics_wtk = self.eval_fn(self.test_loader)
        # Capture the global-leg delta while the model still equals w_k^t.
        self.get_grad()

        if self.v_k_state is None:
            self.v_k_state = {
                k: v.detach().clone() for k, v in self.server_model_state.items()
            }
        self.model.load_state_dict(
            {k: v.to(self.device) for k, v in self.v_k_state.items()}
        )
        self.do_proximity = True
        self._set_optimizer(self.ditto_personal_lr, weight_decay=0.0)
        self.train_fn(num_epochs=self.ditto_personal_epochs)
        self.v_k_state = {
            k: v.detach().cpu() for k, v in self.model.state_dict().items()
        }
        self.val_loss_after, self.val_metrics_after = self.eval_fn()
        self.test_loss_after, self.test_metrics_after = self.eval_fn(
            self.test_loader
        )
        self.result_time = time.time() - start

    def get_communication_content(self):
        result_dict = super().get_communication_content()
        result_dict["ditto_v"] = {
            k: v.detach().cpu() for k, v in self.v_k_state.items()
        }
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
