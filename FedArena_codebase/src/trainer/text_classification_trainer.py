import torch
from .trainer import BaseTrainer
from utils.metrics_utils import calculate_metrics


class TextClassificationTrainer(BaseTrainer):
    """Trainer for tabular data where each batch yields (index, ([x_num, x_cat], label))."""

    def train_one_epoch(self, train_loader, *args, **kwargs):
            # Optional gradient clipping - pass via kwargs from the client.
            max_grad_norm = kwargs.get("max_grad_norm", None)
            loss = 0
            n_total = 0
            for batch in train_loader:
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**inputs)

                self.optimizer.zero_grad()
                batch_loss = outputs.loss
                batch_loss.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=max_grad_norm
                    )
                self.optimizer.step()

                after_step = kwargs.get("after_optimizer_step")
                if after_step is not None:
                    after_step()

                loss += batch_loss.item() * len(batch["input_ids"])
                n_total += len(batch["input_ids"])
            return loss / n_total

    def validate(self, valid_loader, *args, **kwargs):
        self.model.eval()
        self.model.to(self.device)
        val_loss = 0
        n_total = 0
        fin_targets = []
        fin_outputs = []

        with torch.no_grad():
            for batch in valid_loader:
                inputs = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(**inputs)

                logits = outputs.logits.detach().cpu()
                targets = inputs["labels"].cpu()
                val_loss += outputs.loss.item() * len(inputs["input_ids"])
                n_total += len(inputs["input_ids"])

                fin_targets.extend(targets.tolist())
                fin_outputs.extend(logits.tolist())

        metrics, _ = self.calculate_metrics(fin_targets, fin_outputs)
        return val_loss / n_total, metrics

    def test(self, test_loader, *args, **kwargs):
        self.model.eval()
        self.model.to(self.device)
        test_loss = 0
        n_total = 0
        fin_targets = []
        fin_outputs = []

        with torch.no_grad():
            for batch in test_loader:
                inputs = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(**inputs)

                logits = outputs.logits.detach().cpu()
                targets = inputs["labels"].cpu()
                test_loss += outputs.loss.item() * len(inputs["input_ids"])
                n_total += len(inputs["input_ids"])

                fin_targets.extend(targets.tolist())
                fin_outputs.extend(logits.tolist())

        test_loss /= len(test_loader)
        return test_loss, fin_targets, fin_outputs

    def calculate_metrics(self, fin_targets, fin_outputs):
        return calculate_metrics(fin_targets, fin_outputs)
