import torch
import torch.nn as nn
from torch.optim import Optimizer
from abc import abstractmethod
from typing import Optional, Callable, List

class BaseTrainer():
    def __init__(self, model: nn.Module, loss: Callable[..., torch.Tensor], optimizer: Optional[Optimizer]=None, device: str='cpu'):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.device = device
        
        self.train_loss = None

    @abstractmethod
    def train_one_epoch(self, train_loader, *args, **kwargs):
        pass
    
    @abstractmethod
    def validate(self, val_loader, *args, **kwargs):
        pass
    
    @abstractmethod
    def test(self, test_loader, *args, **kwargs):
        pass
    
    @abstractmethod
    def calcuilate_metrics(self):
        pass
    
    def train(self, train_loader, num_epochs, *args, **kwargs):
        self.model.train()
        self.model.to(self.device)
        self.train_loss = []
        for epoch_idx in range(num_epochs):
            loss_epoch = self.train_one_epoch(
                train_loader,
                *args,
                epoch_idx=epoch_idx,
                num_epochs=num_epochs,
                **kwargs,
            )
            self.train_loss.append(loss_epoch)
        