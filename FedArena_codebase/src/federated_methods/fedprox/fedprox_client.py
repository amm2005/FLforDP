from ..fedavg.client import Client


class FedProxClient(Client):
    """FedProx client (Li et al., MLSys 2020): local update with a proximal term toward the global model."""

    def __init__(self, *client_args, **client_kwargs):
        base_client_args = client_args[:2]
        super().__init__(*base_client_args, **client_kwargs)
        self.client_args = client_args
        self.fed_prox_lambda = self.client_args[2]
        self.server_model_state = None

    def get_loss_value(self, outputs, targets):
        loss = super().get_loss_value(outputs, targets)
        proximity = (
            0.5
            * self.fed_prox_lambda
            * sum((p.float() - self.server_model_state[name].float()).float().norm() ** 2 for name, p in self.model.named_parameters())
        )
        loss += proximity
        return loss
