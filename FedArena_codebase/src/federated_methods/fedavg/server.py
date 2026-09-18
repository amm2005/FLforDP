import os
import torch
import numpy as np
import pandas as pd
from collections import OrderedDict
from hydra.utils import instantiate

from utils.utils import create_model_info
from utils.losses import get_loss
from utils.data_utils import get_dataset_loader
from utils.metrics_utils import (
    calculate_metrics,
    stopping_criterion,
    check_metrics_names,
)


class Server:
    """FedAvg server (McMahan et al., AISTATS 2017): holds and aggregates the global model."""

    def __init__(self, cfg, train_df=None):
        self.cfg = cfg
        # Head size override from the dataset's num_classes.
        train_num_classes = getattr(train_df, "num_classes", None)
        self.global_model = (
            instantiate(cfg.models[0], num_classes=train_num_classes)
            if train_num_classes is not None
            else instantiate(cfg.models[0])
        )
        self.client_gradients = [
            OrderedDict() for _ in range(cfg.federated_params.amount_of_clients)
        ]
        self.server_metrics = [
            pd.DataFrame() for _ in range(cfg.federated_params.amount_of_clients)
        ]
        self.client_test_results = [None] * cfg.federated_params.amount_of_clients
        self.client_all_metrics = [None] * cfg.federated_params.amount_of_clients

        self.server_val_loss = None
        self.server_val_metrics_df = None
        self.current_server_val_metrics = {}

        pool_enabled = (
            train_df is not None
            and hasattr(train_df, "get_server_test_dataset")
            and hasattr(getattr(train_df, "orig_data", None), "columns")
            and "server_pool" in train_df.orig_data.columns
        )
        if pool_enabled:
            self.test_df = train_df.get_server_test_dataset()
            self.server_val_df = train_df.get_server_val_dataset()
            print(
                f"[Server] Using pooled test ({len(self.test_df)} rows) "
                f"and pooled val ({len(self.server_val_df)} rows) from train split.",
                flush=True,
            )
        else:
            self.test_df = instantiate(
                cfg.test_dataset, cfg=cfg, mode="test", _recursive_=False
            )
            self.server_val_df = None

        self.test_loader = get_dataset_loader(self.test_df, cfg, drop_last=False)
        if self.server_val_df is not None and len(self.server_val_df) > 0:
            self.server_val_loader = get_dataset_loader(
                self.server_val_df, cfg, drop_last=False
            )
        else:
            self.server_val_loader = None
        self.device = (
            "{}:{}".format(
                cfg.training_params.device, cfg.training_params.device_ids[0]
            )
            if cfg.training_params.device == "cuda"
            else "cpu"
        )
        self.model_path = self.create_model_path()
        self.best_metrics = {
            metric: 1000 * (metric == "loss")
            for metric in cfg.federated_params.server_saving_metrics
        }
        check_metrics_names(self.best_metrics)
        self.metric_aggregation = cfg.federated_params.server_saving_agg
        assert self.metric_aggregation in [
            "uniform",
            "weighted",
        ], f"federated_params.server_saving_agg can be only ['uniform', 'weighted'], you provide: {self.best_metrics}"
        self.best_round = 0
        self.last_metrics = None
        self.current_val_metrics = {}
    
    def _init_trainer(self):
        trainer = instantiate(self.cfg.trainer, model=self.global_model, loss = self.criterion, device=self.device)
        return trainer

    def eval_fn(self):
        self.criterion = get_loss(
            loss_cfg=self.cfg.loss,
            device=self.device,
            df=self.test_df,
        )
        print('criterion: ', self.criterion)
        trainer = self._init_trainer()
        print('trainer:')
        return trainer.test(test_loader=self.test_loader)

    def test_global_model(self):
        print(f"\nServer Test Results:")
        self.test_loss, fin_targets, fin_outputs = self.eval_fn()
        self.last_metrics = calculate_metrics(
            fin_targets,
            fin_outputs,
            verbose=True,
        )
        print(f"Server Test Loss: {self.test_loss}")

    def validate_global_model(self):
        """Evaluate the global model on the server validation set."""
        if self.server_val_loader is None:
            return

        print(f"\nServer Validation Results:")
        self.criterion = get_loss(
            loss_cfg=self.cfg.loss,
            device=self.device,
            df=self.server_val_df,
        )
        trainer = self._init_trainer()
        self.server_val_loss, fin_targets, fin_outputs = trainer.test(
            test_loader=self.server_val_loader,
        )
        metrics_tuple = calculate_metrics(
            fin_targets, fin_outputs, verbose=True,
        )
        self.server_val_metrics_df = metrics_tuple[0]
        print(f"Server Validation Loss: {self.server_val_loss}")

        self.current_server_val_metrics = {"loss": self.server_val_loss}
        for idx in self.server_val_metrics_df.index:
            self.current_server_val_metrics[idx] = (
                self.server_val_metrics_df.loc[idx].mean()
            )

    def set_client_result(self, client_result):
        rank = client_result["rank"]
        self.client_gradients[rank] = client_result["grad"]
        self.server_metrics[rank] = client_result["server_metrics"]
        self.client_test_results[rank] = client_result.get("client_test_metrics")
        self.client_all_metrics[rank] = {
            "val_before": client_result["server_metrics"],
            "test_before": client_result.get("client_test_metrics"),
            "val_after": client_result.get("client_val_after"),
            "test_after": client_result.get("client_test_after"),
        }

    def save_best_model(self, round):
        client_metrics_dfs = [m[0] for m in self.server_metrics]
        val_losses = [m[1] for m in self.server_metrics]
        val_len_dfs = [m[2] for m in self.server_metrics]
        weights = [v / sum(val_len_dfs) for v in val_len_dfs]
        metrics_names = client_metrics_dfs[0].index

        if self.metric_aggregation == "uniform":
            agg_val_loss = np.mean(val_losses)
            agg_metrics = pd.concat(client_metrics_dfs).groupby(level=0).mean()
        elif self.metric_aggregation == "weighted":
            agg_val_loss = np.sum(
                [loss * w for loss, w in zip(val_losses, weights)]
            )
            agg_metrics = sum(
                w * m for w, m in zip(weights, client_metrics_dfs)
            )
        agg_metrics = agg_metrics.reindex(metrics_names)

        self.current_val_metrics = {"loss": agg_val_loss}
        for idx in agg_metrics.index:
            self.current_val_metrics[idx] = agg_metrics.loc[idx].mean()

        print(f"\nAggregated Client Valid Results:\n{agg_metrics}")
        print(f"Aggregated Client Valid Loss: {agg_val_loss}")

        # Prefer server-val metrics for checkpointing when available.
        if self.server_val_metrics_df is not None:
            print('server_val_metrics is not None')
            ckpt_val_loss = self.server_val_loss
            ckpt_metrics = self.server_val_metrics_df.copy()
        else:
            ckpt_val_loss = agg_val_loss
            ckpt_metrics = agg_metrics

        epochs_no_improve, best_metrics = stopping_criterion(
            ckpt_val_loss, ckpt_metrics, self.best_metrics, epochs_no_improve=0
        )
        if epochs_no_improve == 0 and ckpt_val_loss is not np.nan:
            print("\nServer best model updated.")
            self.best_metrics = best_metrics
            self.best_round = round

        ckpt_metrics.loc["loss"] = ckpt_val_loss
        print(f"\nCriterion metrics:")
        for k, v in self.best_metrics.items():
            print(
                f"Current {k}: {ckpt_metrics.loc[k].mean()}\nBest {k}: {v}\nBest round: {self.best_round}\n",
            )

    def create_model_path(self):
        self.target_label_names = [self.test_df.name]

        return f"{self.cfg.single_run_dir}/{type(instantiate(self.cfg.federated_method, _recursive_=False)).__name__}_{'_'.join(self.target_label_names)}"

    def send_content_to_client(self, pipe_num, content):
        self.pipes[pipe_num].send(content)

    def rcv_content_from_client(self, pipe_num):
        # Poll periodically: a killed client never sends EOF, so bare recv() would hang.
        pipe = self.pipes[pipe_num]
        procs = getattr(self, "client_processes", None)
        while not pipe.poll(30):
            if procs is None:
                continue
            proc = procs[pipe_num]
            if not proc.is_alive():
                raise RuntimeError(
                    f"Client process on pipe {pipe_num} (pid {proc.pid}) died "
                    f"mid-round with exitcode {proc.exitcode} — see its "
                    f"traceback above in the log."
                )
        client_content = pipe.recv()

        return client_content
