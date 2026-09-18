import copy
from collections import OrderedDict

from federated_methods.fedavg.fedavg import FedAvg

from .client_worker import run_multiprocess_client
from .local_server import LocalServer

from .local_only_client import LocalOnlyClient


class LocalOnly(FedAvg):
    """Local-only baseline — each client trains independently with no aggregation."""

    def _init_server(self, cfg):
        self.server = LocalServer(cfg, train_df=getattr(self, "df", None))
        
    def _init_client_cls(self):
        self.client_cls = LocalOnlyClient
        self.client_args = [self.cfg, self.df]
        self.client_kwargs = {
            "client_cls": self.client_cls,
            "pipe": None,
            "rank": None,
            "attack_type": None,
        }

    def _init_manager(self):
        from utils.manager_utils import Manager
        self.manager = _LocalManager(self.cfg, self.server)

    def get_communication_content(self, rank):
        opt_state = self.server.client_optimizer_state_dicts[rank]
        return {
            "update_model": {
                k: v.cpu()
                for k, v in self.server.client_models[rank].items()
            },
            "update_optimizer": opt_state,
            "attack_type": (
                self.client_map_round[rank],
                self.attack_configs[self.client_map_round[rank]],
            ),
        }

    def aggregate(self):
        n_clients = self.cfg.federated_params.amount_of_clients

        for rank in range(n_clients):
            if self.server.client_gradients[rank]:
                for key, delta in self.server.client_gradients[rank].items():
                    self.server.client_models[rank][key] = (
                        self.server.client_models[rank][key] + delta.cpu()
                    )

        avg_weights = OrderedDict()
        for key in self.server.client_models[0]:
            avg_weights[key] = (
                sum(self.server.client_models[r][key].float() for r in range(n_clients))
                / n_clients
            )
        return avg_weights


class _LocalManager:
    """Spawns client processes with the spawn start method and a thread-safe worker entry."""

    def __init__(self, cfg, server):
        from utils.manager_utils import Manager
        self._inner = Manager(cfg, server)

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def create_clients(self, client_args, client_kwargs, attack_map):
        import torch.multiprocessing as tmp

        from utils.preprocessing_cache import preprocessing_cache

        ctx = tmp.get_context("spawn")
        self._inner.processes = []
        self._inner.pipes = [ctx.Pipe() for _ in range(self._inner.batches.batch_size)]
        self._inner.server.pipes = [pipe[0] for pipe in self._inner.pipes]

        cache_snapshot = copy.deepcopy(preprocessing_cache)
        client_kwargs = {
            **client_kwargs,
            "_preprocessing_cache_snapshot": cache_snapshot,
        }

        for rank in range(self._inner.batches.batch_size):
            client_kwargs["pipe"] = self._inner.pipes[rank][1]
            client_kwargs["rank"] = rank
            client_kwargs["attack_type"] = attack_map[rank]
            p = ctx.Process(
                target=run_multiprocess_client,
                args=client_args,
                kwargs=client_kwargs,
            )
            p.start()
            self._inner.processes.append(p)
