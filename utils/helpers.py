import random

import numpy as np
import torch


def set_global_seed(seed, deterministic_execution=False):
    """
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
    np.random.seed(seed)
    torch.manual_seed(seed)

    if deterministic_execution:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def sf01(arr):
    """
    swap axes 0 and 1 and then flatten
    """
    return torch.flatten(arr.transpose(0, 1))


def reset_envs(envs, device):
    """
    Reset OpenAI ProcGen environments by feeding action value of -1
    to all the enviornments.
    """
    action = (torch.zeros(envs.num_envs, 1)-1).to(device)
    obs, _, _, _ = envs.step(action)
    return obs


# --- Taken from: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr ---


def init(module, weight_init, bias_init, gain=1):
    """
    Initialisation of the network weights and biases.
    """
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module
