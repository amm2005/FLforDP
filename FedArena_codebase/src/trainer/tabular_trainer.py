import torch
from .trainer import BaseTrainer
from utils.metrics_utils import calculate_metrics


class TabularTrainer(BaseTrainer):
    """Trainer for tabular data where each batch yields (index, ([x_num, x_cat], label))."""

    def _prepare_outputs_targets(self, outputs, targets):
        """For regression: squeeze (batch,1)->(batch), ensure float32. Classification: leave targets as long."""
        if outputs.dim() == 2 and outputs.shape[1] == 1:
            outputs = outputs.squeeze(-1)
            targets = targets.to(outputs.dtype)
        return outputs, targets

    def train_one_epoch(self, train_loader, *args, **kwargs):
        # Optional gradient clipping - pass via kwargs from the client.
        max_grad_norm = kwargs.get("max_grad_norm", None)
        loss = 0
        for batch in train_loader:
            _, (inputs, targets) = batch
            x_num = inputs[0].to(self.device)
            x_cat = inputs[1].to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(x_num=x_num, x_cat=x_cat)
            outputs, targets = self._prepare_outputs_targets(outputs, targets)

            self.optimizer.zero_grad()
            batch_loss = self.loss(outputs, targets)
            batch_loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=max_grad_norm
                )
            self.optimizer.step()

            after_step = kwargs.get("after_optimizer_step")
            if after_step is not None:
                after_step()

            loss += batch_loss.item()

            x_num = x_num.to("cpu")
            x_cat = x_cat.to("cpu")
            targets = targets.to("cpu")
        return loss / len(train_loader)

    def validate(self, valid_loader, *args, **kwargs):
        self.model.eval()
        val_loss = 0
        fin_targets = []
        fin_outputs = []

        with torch.no_grad():
            for _, batch in enumerate(valid_loader):
                _, (inputs, targets) = batch
                x_num = inputs[0].to(self.device)
                x_cat = inputs[1].to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(x_num=x_num, x_cat=x_cat)
                outputs, targets = self._prepare_outputs_targets(outputs, targets)

                val_loss += self.loss(outputs, targets).detach().item()

                fin_targets.extend(targets.tolist())
                fin_outputs.extend(outputs.tolist())

                x_num = x_num.to("cpu")
                x_cat = x_cat.to("cpu")
                targets = targets.to("cpu")

        metrics, _ = self.calculate_metrics(fin_targets, fin_outputs)
        return val_loss / len(valid_loader), metrics

    def test(self, test_loader, *args, **kwargs):
        self.model.to(self.device)
        self.model.eval()
        test_loss = 0
        fin_targets = []
        fin_outputs = []

        with torch.no_grad():
            for _, batch in enumerate(test_loader):
                _, (inputs, targets) = batch
                x_num = inputs[0].to(self.device)
                x_cat = inputs[1].to(self.device)
                
                targets = targets.to(self.device)

                outputs = self.model(x_num=x_num, x_cat=x_cat)
                outputs, targets = self._prepare_outputs_targets(outputs, targets)

                test_loss += self.loss(outputs, targets).detach().item()

                fin_targets.extend(targets.tolist())
                fin_outputs.extend(outputs.tolist())

                x_num = x_num.to("cpu")
                x_cat = x_cat.to("cpu")
                targets = targets.to("cpu")

        test_loss /= len(test_loader)
        return test_loss, fin_targets, fin_outputs

    def calculate_metrics(self, fin_targets, fin_outputs):
        return calculate_metrics(fin_targets, fin_outputs)
