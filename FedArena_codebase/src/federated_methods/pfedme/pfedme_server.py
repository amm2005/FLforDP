from ..fedavg.server import Server


class PFedMeServer(Server):
    """pFedMe server (Dinh et al., NeurIPS 2020): stores per-client personalized models theta_k."""

    def __init__(self, cfg, train_df=None):
        super().__init__(cfg, train_df=train_df)
        n = cfg.federated_params.amount_of_clients
        # CPU copies of theta_k, one per client.
        self.theta_states = [None] * n

    def set_client_result(self, client_result):
        super().set_client_result(client_result)
        rank = client_result["rank"]
        if "pfedme_theta" in client_result:
            self.theta_states[rank] = client_result["pfedme_theta"]
        cm = self.client_all_metrics[rank]
        if cm is not None:
            cm["val_wtk"] = client_result.get("client_val_wtk")
            cm["test_wtk"] = client_result.get("client_test_wtk")
