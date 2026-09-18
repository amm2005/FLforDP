from ..fedavg.server import Server


class DittoServer(Server):
    """Ditto server (Li et al., ICML 2021): persists per-client personal models and the global-leg metrics."""

    def __init__(self, cfg, train_df=None):
        super().__init__(cfg, train_df=train_df)
        n = cfg.federated_params.amount_of_clients
        self.v_states = [None] * n

    def set_client_result(self, client_result):
        super().set_client_result(client_result)
        rank = client_result["rank"]
        if "ditto_v" in client_result:
            self.v_states[rank] = client_result["ditto_v"]
        cm = self.client_all_metrics[rank]
        if cm is not None:
            cm["val_wtk"] = client_result.get("client_val_wtk")
            cm["test_wtk"] = client_result.get("client_test_wtk")
