import torch 

from ..fedavg.client import Client 
from hydra.utils import instantiate 

import copy 
import time

def _optimizer_state_to(state_dict, device):
    """Move tensors in an optimizer state_dict to the given device."""
    new_state = {}
    for param_id, param_state in state_dict["state"].items():
        new_state[param_id] = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in param_state.items()
        }
    return {
        "state": new_state,
        "param_groups": state_dict["param_groups"],
    }

class LocalOnlyClient(Client): 
    
    
    def create_pipe_commands(self): 
        pipe_commands_map = super().create_pipe_commands() 
        
        def _load_optimizer_state(opt_state):
            if opt_state is None:
                return
            self.optimizer.load_state_dict(opt_state)
            for pg in self.optimizer.param_groups:
                pg["lr"] = float(self.cfg.optimizer.lr)
                pg["weight_decay"] = float(self.cfg.optimizer.weight_decay)
        
        pipe_commands_map['update_optimizer'] = _load_optimizer_state
        return pipe_commands_map 
    
    def get_communication_content(self): 
        result_dict = super().get_communication_content()
        result_dict['update_optimizer_state_dict'] = _optimizer_state_to(self.optimizer.state_dict(), device='cpu')
        return result_dict
