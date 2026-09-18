import sys
import copy
import time
import signal
import torch
import pandas as pd 
from collections import OrderedDict
from hydra.utils import instantiate
from omegaconf import OmegaConf

from utils.losses import get_loss
from utils.data_utils import get_dataset_loader
from utils.utils import handle_client_process_sigterm
from utils.attack_utils import add_attack_functionality


class Client:
    """FedAvg client (McMahan et al., AISTATS 2017): local SGD on client data."""

    def __init__(self, *client_args, **client_kwargs):
        self.client_args = client_args
        self.client_kwargs = client_kwargs
        cfg = self.client_args[0]
        df = self.client_args[1]
        self.cfg = cfg
        self.df = df
        self.train_dataset = df
        self.rank = client_kwargs["rank"]
        self.pipe = client_kwargs["pipe"]
        self.valid_df = None
        self.test_df = None
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None
        self.criterion = None
        self.server_model_state = None
        self.print_metrics = cfg.federated_params.print_client_metrics
        device_idx = (self.rank + 1) % len(self.cfg.training_params.device_ids)
        self.device = (
            "{}:{}".format(
                cfg.training_params.device, cfg.training_params.device_ids[device_idx]
            )
            if cfg.training_params.device == "cuda"
            else "cpu"
        )
        # Head size from the dataset's num_classes (cfg default is not authoritative).
        df_num_classes = getattr(df, "num_classes", None)
        self.model = (
            instantiate(cfg.models[0], num_classes=df_num_classes)
            if df_num_classes is not None
            else instantiate(cfg.models[0])
        )
        self.model.to(self.device)
        self._set_train_df()
        self._init_loaders()
        self._init_optimizer()
        self._init_criterion()
        self.pipe_commands_map = self.create_pipe_commands()
        self._init_trainer()

        self.grad = OrderedDict()

    def _init_compressor(self):
        return instantiate(self.cfg.compressor, model=self.model, device=self.device)

    def _init_trainer(self):
        self.trainer = instantiate(self.cfg.trainer, model=self.model, loss=self.get_loss_value, optimizer=self.optimizer, device=self.device)

    def _init_optimizer(self):
        self.optimizer = instantiate(self.cfg.optimizer, params=self.model.parameters())

    def _init_criterion(self):
        self.criterion = get_loss(
            loss_cfg=self.cfg.loss,
            device=self.device,
            df=self.train_dataset,
            init_pos_weight=self.init_pos_weight,
        )

    def _set_train_df(self):
        self.train_dataset = self.df.to_client_side(self.rank)
        self.init_pos_weight = False

    def _init_loaders(self):
        full_data = self.train_dataset.data

        if "phase" in full_data.columns:
            self.valid_df = copy.deepcopy(self.train_dataset)
            self.test_df = copy.deepcopy(self.train_dataset)

            self.train_dataset.data = (
                full_data[full_data["phase"] == "train"].reset_index(drop=True)
            )
            self.valid_df.data = (
                full_data[full_data["phase"] == "val"].reset_index(drop=True)
            )
            self.test_df.data = (
                full_data[full_data["phase"] == "test"].reset_index(drop=True)
            )
            if len(self.valid_df.data) == 0:
                print(
                    f"[Client {self.rank}] Warning: empty local val split.",
                    flush=True,
                )
            if len(self.test_df.data) == 0:
                print(
                    f"[Client {self.rank}] Warning: empty local test split.",
                    flush=True,
                )
        else:
            # Legacy path: split locally when no precomputed phases exist.
            fp = self.cfg.federated_params
            train_frac = OmegaConf.select(fp, "client_train_frac", default=0.7)
            val_frac = OmegaConf.select(fp, "client_val_frac", default=0.15)
            test_frac = OmegaConf.select(fp, "client_test_frac", default=0.15)

            self.valid_df = copy.deepcopy(self.train_dataset)
            self.test_df = copy.deepcopy(self.train_dataset)
            (
                self.train_dataset.data,
                self.valid_df.data,
                self.test_df.data,
            ) = self.df.train_val_test_split(
                df=full_data,
                random_state=self.cfg.random_state,
                train_frac=float(train_frac),
                val_frac=float(val_frac),
                test_frac=float(test_frac),
            )

        self.train_loader = get_dataset_loader(
            self.train_dataset, self.cfg, drop_last=False
        )
        self.valid_loader = get_dataset_loader(self.valid_df, self.cfg, drop_last=False)
        self.test_loader = get_dataset_loader(self.test_df, self.cfg, drop_last=False)

    def _set_attack_type(self, attack_content):
        self.attack_type = attack_content[0]
        self.attack_config = attack_content[1]

    def reinit_self(self, new_rank):
        self.client_kwargs["rank"] = new_rank
        self.__init__(*self.client_args, **self.client_kwargs)

        content = self.pipe.recv()
        self.parse_communication_content(content)

    def shutdown_self(self):
        print(f"Exit child {self.rank} process")
        sys.exit(0)

    def create_pipe_commands(self):
        pipe_commands_map = {
            "update_model": lambda state_dict: self.model.load_state_dict(
                {k: v.to(self.device) for k, v in state_dict.items()}
            ),
            "attack_type": self._set_attack_type,
            "shutdown": lambda _: self.shutdown_self(),
            "reinit": lambda new_rank: self.reinit_self(new_rank),
        }

        return pipe_commands_map

    def train_fn(self):
        from omegaconf import OmegaConf
        max_grad_norm = OmegaConf.select(
            self.cfg, "federated_params.max_grad_norm", default=None
        )
        self.trainer.train(
            train_loader=self.train_loader,
            num_epochs=self.cfg.federated_params.round_epochs,
            max_grad_norm=max_grad_norm,
        )
        
        
    def get_loss_value(self, outputs, targets):
        return self.criterion(outputs, targets)

    def _model_forward(self, inputs):
        """Dispatch model call by batch shape (image vs tabular)."""
        if len(inputs) == 1:
            return self.model(inputs[0].to(self.device))
        return self.model(
            x_num=inputs[0].to(self.device),
            x_cat=inputs[1].to(self.device),
        )

    def eval_fn(self, loader=None):
        if loader is None:
            loader = self.valid_loader
        return self.trainer.validate(valid_loader=loader)

    def get_grad(self):
        self.model.eval()
        for key, _ in self.model.state_dict().items():
            self.grad[key] = (
                self.model.state_dict()[key].to(self.device) - self.server_model_state[key]
            )
        compressor = self._init_compressor()
        grads, _ = compressor.compress([self.grad])
        self.grad = grads[0]
        for key, val in self.grad.items():
            self.grad[key] = val.to('cpu')

    def train(self):
        self.server_model_state = copy.deepcopy(self.model).state_dict()
        start = time.time()
        self.val_loss_before, self.val_metrics_before = self.eval_fn()
        self.test_loss_before, self.test_metrics_before = self.eval_fn(
            self.test_loader
        )
        self.train_fn()
        self.val_loss_after, self.val_metrics_after = self.eval_fn()
        self.test_loss_after, self.test_metrics_after = self.eval_fn(
            self.test_loader
        )
        self.get_grad()
        self.result_time = time.time() - start

    def get_communication_content(self):
        result_dict = {
            "grad": self.grad,
            "rank": self.rank,
            "time": self.result_time,
            # val/test_before: global-model eval (backward-compatible keys).
            "server_metrics": (
                self.val_metrics_before,
                self.val_loss_before,
                len(self.valid_df),
            ),
            "client_test_metrics": (
                self.test_metrics_before,
                self.test_loss_before,
                len(self.test_df),
            ),
            "client_val_after": (
                self.val_metrics_after,
                self.val_loss_after,
                len(self.valid_df),
            ),
            "client_test_after": (
                self.test_metrics_after,
                self.test_loss_after,
                len(self.test_df),
            ),
        }
        if self.print_metrics:
            result_dict["client_metrics"] = (
                self.val_loss_after,
                self.val_metrics_after,
            )

        return result_dict

    def parse_communication_content(self, content):
        for key, value in content.items():
            if key in self.pipe_commands_map.keys():
                self.pipe_commands_map[key](value)
            else:
                raise ValueError(
                    f"Recieved content in client {self.rank} from server, with unknown key={key}"
                )


def multiprocess_client(*client_args, client_cls, pipe, rank, attack_type):
    torch.set_num_threads(1)
    torch.multiprocessing.set_sharing_strategy("file_system")

    client_kwargs = {"pipe": pipe, "rank": rank}
    client = client_cls(*client_args, **client_kwargs)
    signal.signal(
        signal.SIGTERM,
        lambda signum, frame: handle_client_process_sigterm(signum, frame, rank),
    )

    while True:
        content = client.pipe.recv()
        client.parse_communication_content(content)

        if client.attack_type != "no_attack":
            client = add_attack_functionality(
                client, client.attack_type, client.attack_config
            )

        client.train()

        content = client.get_communication_content()
        client.pipe.send(content)
