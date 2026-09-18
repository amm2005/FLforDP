from .trainer import BaseTrainer
import torch
from utils.metrics_utils import calculate_metrics

class CifarTrainer(BaseTrainer):
    def train_one_epoch(self, train_loader, *args, **kwargs):
        # Optional gradient clipping (e.g. SCAFFOLD).
        max_grad_norm = kwargs.get("max_grad_norm", None)
        loss = 0
        for batch in train_loader:
            _, (input, targets) = batch

            inp = input[0].to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(inp)

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
        return loss / len(train_loader)

    
    def validate(self, valid_loader, *args, **kwargs):
        self.model.eval()
        val_loss = 0
        # Accumulate batch tensors on-device; convert to Python lists ONCE at the
        # end (single GPU->CPU sync) instead of a .tolist() sync every batch.
        fin_targets = []
        fin_outputs = []

        with torch.no_grad():
            for _, batch in enumerate(valid_loader):
                _, (input, targets) = batch

                inp = input[0].to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inp)

                val_loss += self.loss(outputs, targets).detach().item()

                fin_targets.append(targets)
                fin_outputs.append(outputs)

        fin_targets = torch.cat(fin_targets).cpu().tolist() if fin_targets else []
        fin_outputs = torch.cat(fin_outputs).cpu().tolist() if fin_outputs else []

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
                _, (input, targets) = batch
                inp = input[0].to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inp)

                test_loss += self.loss(outputs, targets).detach().item()
                fin_targets.append(targets)
                fin_outputs.append(outputs)

        fin_targets = torch.cat(fin_targets).cpu().tolist() if fin_targets else []
        fin_outputs = torch.cat(fin_outputs).cpu().tolist() if fin_outputs else []
        test_loss /= len(test_loader)
        return test_loss, fin_targets, fin_outputs
 
    
    def calculate_metrics(self, fin_targets, fin_outputs):
        return calculate_metrics(fin_targets, fin_outputs)