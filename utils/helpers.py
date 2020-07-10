import random

import torch
import torch.nn as nn
import numpy as np
        

def set_global_seed(seed, deterministic_execution=False):
    """
    Taken from: https://github.com/lmzintgraf/varibad

    Fix the random seeds of:
    1. random
    2. torch
    3. numpy
    
    Parameters:
    -----------
    seed : `int`
        Random seed (default: 0).
    deterministic_execution : `Boolean`
        Make code fully deterministic. 
        Expects 1 process and uses deterministic CUDNN.        
    """     
    print('Seeding random, torch, numpy.')
    random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    
    if deterministic_execution:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        print('Note that due to parallel processing results will be similar but not identical. '
              'Use only one process and set --deterministic_execution to True if you want identical results. '
              '(Not recommended; will slow code down and might not be possible with A2C). ')   


def sf01(arr):
    """
    swap axes 0 and 1 and then flatten 
    """
    return torch.flatten(arr.transpose(0, 1))

# --- Taken from: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr ---


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module           
        
        
# Necessary for KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """
    Decreases the learning rate linearly.
    """
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr        