import time

from .server import Server
from .client import Client
from utils.manager_utils import Manager
from utils.attack_utils import (
    map_attack_clients,
    set_attack_rounds,
    set_client_map_round,
    load_attack_configs,
)
from hydra.utils import instantiate
from omegaconf import OmegaConf


class FedAvg:
    """FedAvg (McMahan et al., AISTATS 2017): federated averaging of client updates."""

    def __init__(self):
        self.server = None
        self.client = None
        self.rounds = 0
        self.terminated_event = None
        self.client_map_round = None
        self.client_sizes = None

    def _init_federated(self, cfg, df):
        self.cfg = cfg
        self.df = df
        self._compute_client_sizes(df)
        self.attack_setup(cfg)
        self._init_server(cfg)
        self._init_client_cls()
        self._init_manager()
        
    def _compute_client_sizes(self, df): 
        num_clients = self.cfg.federated_params.amount_of_clients
        data = df.orig_data 
        has_pool = 'server_pool' in data.columns 
        has_phase = 'phase' in data.columns
        sizes = [] 
        for rank in range(num_clients): 
            mask = data['client'] == rank
            if has_pool: 
                mask &= ~data['server_pool']
            if has_phase: 
                mask &= data['phase'] == 'train'
            sizes.append(int(mask.sum()))
        self.client_sizes = sizes 

    def _init_server(self, cfg):
        self.server = Server(cfg, train_df=getattr(self, "df", None))

    def _init_client_cls(self):
        self.client_cls = Client
        self.client_args = [self.cfg, self.df]
        self.client_kwargs = {
            "client_cls": self.client_cls,
            "pipe": None,
            "rank": None,
            "attack_type": None,
        }

    def _init_manager(self):
        self.manager = Manager(self.cfg, self.server)

    def attack_setup(self, cfg):
        self.max_rounds = cfg.federated_params.communication_rounds
        self.rounds = self.max_rounds
        self.client_attack_map = map_attack_clients(
            cfg.federated_params.clients_attack_types,
            cfg.federated_params.prop_attack_clients,
            cfg.federated_params.amount_of_clients,
        )
        self.attack_scheme = cfg.federated_params.attack_scheme
        self.attack_rounds = set_attack_rounds(
            cfg.federated_params.prop_attack_rounds, self.rounds, self.attack_scheme
        )
        self.attack_configs = load_attack_configs(
            cfg, cfg.federated_params.clients_attack_types
        )

    def aggregate(self):
        aggregated_weights = self.server.global_model.state_dict()
        total_size = sum(self.client_sizes) 
        weights = [size / total_size for size in self.client_sizes]
        for i in range(len(self.server.client_gradients)): 
            for key, grad in self.server.client_gradients[i].items(): 
                aggregated_weights[key] = aggregated_weights[key] + grad.to(self.server.device) * weights[i]
        return aggregated_weights

    def get_communication_content(self, rank):
        return {
            "update_model": {
                k: v.cpu() for k, v in self.server.global_model.state_dict().items()
            },
            "attack_type": (
                self.client_map_round[rank],
                self.attack_configs[self.client_map_round[rank]],
            ),
        }

    def parse_communication_content(self, client_result):
        self.server.set_client_result(client_result)
        print(f"Client {client_result['rank']} finished in {client_result['time']}")
        if self.cfg.federated_params.print_client_metrics:
            cm = client_result.get("client_metrics")
            if cm is not None:
                print(cm[1])
                print(f"Val loss (after): {cm[0]}\n")
            ct_before = client_result.get("client_test_metrics")
            if ct_before is not None:
                print(ct_before[0])
                print(f"Test loss (before): {ct_before[1]}\n")
            ct_after = client_result.get("client_test_after")
            if ct_after is not None:
                print(ct_after[0])
                print(f"Test loss (after): {ct_after[1]}\n")

    def train_round(self):
        for batch_idx, clients_batch in enumerate(self.clients_loader):
            print(f"Current batch of clients is {clients_batch}", flush=True)

            for pipe_num, rank in enumerate(clients_batch):
                content = self.get_communication_content(rank)
                self.server.send_content_to_client(pipe_num, content)

            for pipe_num, rank in enumerate(clients_batch):
                content = self.server.rcv_content_from_client(pipe_num)
                self.parse_communication_content(content)

            self.manager.step(batch_idx)

    @staticmethod
    def _extract_metric(metric_tuple, metric_name):
        """Extract a scalar metric from a (metrics_df, loss, size) tuple."""
        return dict(metric_tuple[0].mean(axis=1)).get(metric_name, 0.0)

    @staticmethod
    def _metric_snapshot(metric_tuple):
        """Convert a (metrics_df, loss, size) tuple to a serialisable dict."""
        return {
            "loss": float(metric_tuple[1]),
            "metrics": dict(metric_tuple[0].mean(axis=1)),
            "size": int(metric_tuple[2]),
        }

    def _two_track_final_metric(self, best_val_after, best_val_max3):
        """Per-client final-metric hook for two-track methods; base returns None."""
        return None

    def begin_train(self, round_callback=None):
        import torch.multiprocessing as _tmp
        _tmp.set_sharing_strategy("file_system")

        self.compressor = instantiate(self.cfg.compressor, model=self.server.global_model, device=self.server.device)
        self.server.client_gradients = self.compressor.decompress(self.server.client_gradients, shapes=None)
        self.manager.create_clients(
            self.client_args, self.client_kwargs, self.client_attack_map
        )
        self.clients_loader = self.manager.batches

        saving_metrics = list(self.cfg.federated_params.server_saving_metrics)
        best_metric_name = next(
            (m for m in saving_metrics if m != "loss"),
            saving_metrics[0] if saving_metrics else "ROC-AUC",
        )
        direction = OmegaConf.select(self.cfg, "tuning.direction", default="maximize")
        maximize = direction == "maximize"
        _inf = float("-inf") if maximize else float("inf")
        best_val_tune_metric = _inf
        best_tracked = _inf

        patience = OmegaConf.select(
            self.cfg, "federated_params.early_stopping_patience", default=16
        )
        rounds_without_improve = 0
        best_snapshot = {}

        n_clients = self.cfg.federated_params.amount_of_clients
        client_best_val = [_inf] * n_clients
        client_best_info = [None] * n_clients
        # Two-track state, only consumed when emit_two_track_report is set.
        emit_two_track = getattr(self, "emit_two_track_report", False)
        client_best_val_before = [_inf] * n_clients
        client_best_info_before = [None] * n_clients
        client_best_val_after = [_inf] * n_clients
        client_best_info_after = [None] * n_clients
        client_best_val_wtk = [_inf] * n_clients
        client_best_info_wtk = [None] * n_clients
        client_best_val_max3 = [_inf] * n_clients
        client_best_info_max3 = [None] * n_clients

        try:
            for round in range(self.max_rounds):
                print(f"\nRound number: {round}")
                begin_round_time = time.time()
                self.cur_round = round

                self.server.test_global_model()
                self.server.validate_global_model()

                print("\nTraining started\n")

                self.client_map_round = set_client_map_round(
                    self.client_attack_map, self.attack_rounds, self.attack_scheme, round
                )

                self.train_round()

                self.server.save_best_model(round)

                # Per-client best-val tracking across before/after phases.
                for rank in range(n_clients):
                    cm = self.server.client_all_metrics[rank]
                    if (
                        cm is None
                        or cm.get("val_before") is None
                        or cm.get("val_after") is None
                    ):
                        continue
                    vb = self._extract_metric(cm["val_before"], best_metric_name)
                    va = self._extract_metric(cm["val_after"], best_metric_name)
                    if maximize:
                        if va > vb:
                            phase = "after"
                            val_t = cm["val_after"]
                            test_t = cm["test_after"]
                            score_here = va
                        else:
                            phase = "before"
                            val_t = cm["val_before"]
                            test_t = cm["test_before"]
                            score_here = vb
                    else:
                        if va < vb:
                            phase = "after"
                            val_t = cm["val_after"]
                            test_t = cm["test_after"]
                            score_here = va
                        else:
                            phase = "before"
                            val_t = cm["val_before"]
                            test_t = cm["test_before"]
                            score_here = vb
                    if test_t is None:
                        continue
                    better = (
                        score_here > client_best_val[rank]
                        if maximize
                        else score_here < client_best_val[rank]
                    )
                    if better:
                        client_best_val[rank] = score_here
                        client_best_info[rank] = {
                            "best_round": round,
                            "phase": phase,
                            "client_val": self._metric_snapshot(val_t),
                            "client_test": self._metric_snapshot(test_t),
                        }

                    # Opt-in only: keep independent per-phase bests.
                    if emit_two_track:
                        if cm.get("test_before") is not None:
                            better_before = (
                                vb > client_best_val_before[rank]
                                if maximize
                                else vb < client_best_val_before[rank]
                            )
                            if better_before:
                                client_best_val_before[rank] = vb
                                client_best_info_before[rank] = {
                                    "best_round": round,
                                    "phase": "before",
                                    "client_val": self._metric_snapshot(
                                        cm["val_before"]
                                    ),
                                    "client_test": self._metric_snapshot(
                                        cm["test_before"]
                                    ),
                                }
                        if cm.get("test_after") is not None:
                            better_after = (
                                va > client_best_val_after[rank]
                                if maximize
                                else va < client_best_val_after[rank]
                            )
                            if better_after:
                                client_best_val_after[rank] = va
                                client_best_info_after[rank] = {
                                    "best_round": round,
                                    "phase": "after",
                                    "client_val": self._metric_snapshot(
                                        cm["val_after"]
                                    ),
                                    "client_test": self._metric_snapshot(
                                        cm["test_after"]
                                    ),
                                }
                        vw = None
                        if (
                            cm.get("val_wtk") is not None
                            and cm.get("test_wtk") is not None
                        ):
                            vw = self._extract_metric(cm["val_wtk"], best_metric_name)
                            better_wtk = (
                                vw > client_best_val_wtk[rank]
                                if maximize
                                else vw < client_best_val_wtk[rank]
                            )
                            if better_wtk:
                                client_best_val_wtk[rank] = vw
                                client_best_info_wtk[rank] = {
                                    "best_round": round,
                                    "phase": "wtk",
                                    "client_val": self._metric_snapshot(cm["val_wtk"]),
                                    "client_test": self._metric_snapshot(cm["test_wtk"]),
                                }
                        # max-of-3: best-by-val across before/after/wtk this round.
                        trio = [
                            ("before", vb, cm["val_before"], cm["test_before"]),
                            ("after", va, cm["val_after"], cm["test_after"]),
                        ]
                        if vw is not None:
                            trio.append(("wtk", vw, cm["val_wtk"], cm["test_wtk"]))
                        trio = [t for t in trio if t[3] is not None]
                        if trio:
                            w_phase, w_score, w_val, w_test = (
                                max(trio, key=lambda t: t[1])
                                if maximize
                                else min(trio, key=lambda t: t[1])
                            )
                            better_m3 = (
                                w_score > client_best_val_max3[rank]
                                if maximize
                                else w_score < client_best_val_max3[rank]
                            )
                            if better_m3:
                                client_best_val_max3[rank] = w_score
                                client_best_info_max3[rank] = {
                                    "best_round": round,
                                    "phase": w_phase,
                                    "client_val": self._metric_snapshot(w_val),
                                    "client_test": self._metric_snapshot(w_test),
                                }

                if self.server.current_server_val_metrics:
                    tune_metrics = self.server.current_server_val_metrics
                else:
                    tune_metrics = self.server.current_val_metrics

                val_metric = tune_metrics.get(best_metric_name, 0.0)
                if maximize:
                    if val_metric > best_val_tune_metric:
                        best_val_tune_metric = val_metric
                    improved = val_metric > best_tracked
                else:
                    if val_metric < best_val_tune_metric:
                        best_val_tune_metric = val_metric
                    improved = val_metric < best_tracked

                if improved:
                    best_tracked = val_metric
                    rounds_without_improve = 0
                    test_metrics_df = self.server.last_metrics[0]
                    best_snapshot = {
                        "server_test_loss": float(self.server.test_loss),
                        "server_test_metrics": test_metrics_df.mean(axis=1).to_dict(),
                        "server_val_loss": (
                            float(self.server.server_val_loss)
                            if self.server.server_val_loss is not None
                            else None
                        ),
                        "server_val_metrics": dict(tune_metrics),
                        "aggregated_val_metrics": dict(
                            self.server.current_val_metrics
                        ),
                        "best_round": round,
                    }
                else:
                    rounds_without_improve += 1

                if round_callback is not None:
                    round_callback(round, tune_metrics)

                aggregated_weights = self.aggregate()
                self.server.global_model.load_state_dict(aggregated_weights)

                print(f"Round time: {time.time() - begin_round_time}", flush=True)

                if rounds_without_improve >= patience:
                    print(
                        f"\nEarly stopping: {best_metric_name} did not improve for "
                        f"{patience} rounds (direction={direction}).\n",
                        flush=True,
                    )
                    break
        finally:
            print("Shutdown clients, federated learning end", flush=True)
            self.manager.stop_train()

        results = {
            "best_val_tune_metric": best_val_tune_metric,
            **best_snapshot,
            "per_client_best": list(client_best_info),
        }
        # Opt-in only: expose all per-client model tracks.
        if emit_two_track:
            results["per_client_best"] = list(client_best_info_max3)
            results["per_client_best_vk"] = list(client_best_info_after)
            results["per_client_best_global"] = list(client_best_info_before)
            results["per_client_best_wtk"] = list(client_best_info_wtk)
            results["per_client_best_max3"] = list(client_best_info_max3)
            final = self._two_track_final_metric(
                client_best_val_after, client_best_val_max3
            )
            if final is not None:
                results["best_val_tune_metric"] = final
        return results
