import torch
import torchrl.envs as envBase

class BalancingTreeRL(envBase):

    def __init__(self, device='cpu'):
        super.__init__(device = device)
        self.device = device
        self.reset()
    
    def reset(self, **kwargs):
        return