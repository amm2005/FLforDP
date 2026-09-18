from collections import OrderedDict

import torch
from omegaconf import OmegaConf

from federated_methods.fedavg.server import Server
from utils.losses import get_loss
from utils.metrics_utils import calculate_metrics

import transformers


class LocalServer(Server):
    """Server that maintains independent model weights for each client."""

    def __init__(self, cfg, train_df=None):
        super().__init__(cfg, train_df=train_df)
        n_clients = cfg.federated_params.amount_of_clients
        initial_state = self.global_model.state_dict()
        self.client_models = {
            rank: OrderedDict(
                {k: v.clone().cpu() for k, v in initial_state.items()}
            )
            for rank in range(n_clients)
        }
        
        self.client_optimizer_state_dicts = {
            rank: None
            for rank in range(n_clients)
        }
        
        self.n_clients = n_clients
        ensemble_mode = OmegaConf.select(
            cfg, "federated_params.local_only_ensemble", default="uniform"
        ).lower()
        if ensemble_mode not in ("uniform", "weighted"):
            raise ValueError(
                "federated_params.local_only_ensemble must be 'uniform' or 'weighted', "
                f"got {ensemble_mode!r}"
            )
        self._ensemble_mode = ensemble_mode
        self._ensemble_client_weights = self._compute_ensemble_weights(
            train_df, n_clients, ensemble_mode
        )
        if ensemble_mode == "weighted":
            print(
                f"[LocalServer] Ensemble weights (by train size): "
                f"{[f'{w:.1f}' for w in self._ensemble_client_weights]}",
                flush=True,
            )
            
    def set_client_result(self, client_result):
        super().set_client_result(client_result)
        self.client_optimizer_state_dicts[client_result["rank"]] = client_result["update_optimizer_state_dict"]
        

    @staticmethod
    def _compute_ensemble_weights(train_df, n_clients, mode):
        """Normalized ensemble weights per client; uniform is 1/n."""
        if mode != "weighted":
            return [1.0 / n_clients] * n_clients
        if train_df is None or not hasattr(train_df, "orig_data"):
            print(
                "[LocalServer] weighted ensemble requested but train_df missing; "
                "using uniform.",
                flush=True,
            )
            return [1.0 / n_clients] * n_clients
        df = train_df.orig_data
        if "phase" not in df.columns:
            print(
                "[LocalServer] no 'phase' column; cannot infer train sizes; "
                "using uniform.",
                flush=True,
            )
            return [1.0 / n_clients] * n_clients
        sizes = []
        for rank in range(n_clients):
            m = (df["client"] == rank) & (df["phase"] == "train")
            if "server_pool" in df.columns:
                m = m & (~df["server_pool"])
            sizes.append(int(m.sum()))
        total = sum(sizes)
        if total == 0:
            print(
                "[LocalServer] zero total train size for weighting; using uniform.",
                flush=True,
            )
            return [1.0 / n_clients] * n_clients
        return [s / total for s in sizes]

    @staticmethod
    def _prepare_outputs_targets(outputs, targets):
        """Reshape single-column outputs/targets for the loss."""
        if outputs.dim() == 2 and outputs.shape[1] == 1:
            outputs = outputs.squeeze(-1)
            targets = targets.to(outputs.dtype)
        return outputs, targets

    def _model_forward(self, inputs):
        """Dispatch a single forward pass based on the input structure."""
        if len(inputs) == 1:
            return self.global_model(inputs[0].to(self.device))
        elif isinstance(inputs, transformers.tokenization_utils_base.BatchEncoding):
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            return self.global_model(**inputs)
        else:
            return self.global_model(
                x_num=inputs[0].to(self.device),
                x_cat=inputs[1].to(self.device),
            )

    @staticmethod
    def _is_multilabel(targets):
        """Detect multi-label task from target tensor shape and dtype."""
        return (
            targets.dim() == 2
            and targets.shape[1] > 1
            and targets.dtype.is_floating_point
        )

    def _ensemble_run_loader(self, loader, df_for_loss):
        """Aggregate weighted client model predictions over a loader."""
        self.criterion = get_loss(
            loss_cfg=self.cfg.loss,
            device=self.device,
            df=df_for_loss,
        )
        fin_targets = []
        fin_outputs = []
        total_loss = 0.0
        n_batches = len(loader)
        self.global_model.to(self.device)
        self.global_model.eval()

        for batch in loader:
            if isinstance(batch, transformers.tokenization_utils_base.BatchEncoding):
                inputs = batch
                targets = batch["labels"]
            else:
                _, (inputs, targets) = batch
            targets = targets.to(self.device)

            is_ml = self._is_multilabel(targets)

            if is_ml:
                ensemble_probs = None
                for rank in range(self.n_clients):
                    w = self._ensemble_client_weights[rank]
                    self.global_model.load_state_dict(
                        self.client_models[rank], strict=True
                    )
                    with torch.no_grad():
                        out = self._model_forward(inputs)
                        probs = torch.sigmoid(out)
                    ensemble_probs = (
                        w * probs
                        if ensemble_probs is None
                        else ensemble_probs + w * probs
                    )

                eps = 1e-6
                p_clamped = ensemble_probs.clamp(min=eps, max=1.0 - eps)
                ensemble_logits = torch.log(p_clamped / (1.0 - p_clamped))

                total_loss += float(
                    self.criterion(ensemble_logits, targets).detach().item()
                )
                fin_targets.extend(targets.detach().cpu().tolist())
                fin_outputs.extend(ensemble_logits.detach().cpu().tolist())
            else:
                ensemble_outputs = None
                for rank in range(self.n_clients):
                    w = self._ensemble_client_weights[rank]
                    self.global_model.load_state_dict(
                        self.client_models[rank], strict=True
                    )
                    with torch.no_grad():
                        out = self._model_forward(inputs)
                    if isinstance(batch, transformers.tokenization_utils_base.BatchEncoding):
                        out = out.logits
                    ensemble_outputs = (
                        w * out
                        if ensemble_outputs is None
                        else ensemble_outputs + w * out
                    )

                ensemble_outputs, targets_use = self._prepare_outputs_targets(
                    ensemble_outputs, targets
                )
                total_loss += float(
                    self.criterion(ensemble_outputs, targets_use).detach().item()
                )
                fin_targets.extend(targets_use.detach().cpu().tolist())
                fin_outputs.extend(ensemble_outputs.detach().cpu().tolist())

        mean_loss = total_loss / max(n_batches, 1)
        return mean_loss, fin_targets, fin_outputs

    def validate_global_model(self):
        """Ensemble client models on server validation."""
        if self.server_val_loader is None:
            return

        print(
            f"\nServer Validation Results (ensemble, mode={self._ensemble_mode}):",
            flush=True,
        )
        val_loss, fin_targets, fin_outputs = self._ensemble_run_loader(
            self.server_val_loader,
            self.server_val_df,
        )
        self.server_val_loss = val_loss

        metrics_tuple = calculate_metrics(
            fin_targets,
            fin_outputs,
            verbose=True,
        )
        self.server_val_metrics_df = metrics_tuple[0]
        print(f"Server Validation Loss: {val_loss}", flush=True)

        self.current_server_val_metrics = {"loss": self.server_val_loss}
        for idx in self.server_val_metrics_df.index:
            self.current_server_val_metrics[idx] = float(
                self.server_val_metrics_df.loc[idx].mean()
            )

    def test_global_model(self):
        """Ensemble client models on server test split."""
        print(
            f"\nServer Test Results (ensemble, mode={self._ensemble_mode}):",
            flush=True,
        )
        test_loss, fin_targets, fin_outputs = self._ensemble_run_loader(
            self.test_loader,
            self.test_df,
        )
        self.test_loss = test_loss
        self.last_metrics = calculate_metrics(
            fin_targets,
            fin_outputs,
            verbose=True,
        )
        print(f"Server Test Loss: {self.test_loss}", flush=True)
