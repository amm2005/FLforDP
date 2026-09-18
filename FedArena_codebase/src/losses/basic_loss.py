import torch.nn as nn
import torch

class BasicLoss(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.set_loss_fn()
        
    def set_loss_fn():
        raise NotImplementedError('You need to ')
    
    def forward(self, prediction, target, **kwargs):
        return self.loss_fn(prediction, target, **kwargs)
    