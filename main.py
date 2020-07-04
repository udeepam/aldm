"""
Main script to start experiments.
"""
import argparse
import warnings

import torch

# get configs
from configs.coinrun import args_coinrun_ppo
from configs.maze import args_maze_ppo

from learner import Learner


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_label', type=str, default='coinrun_ppo',
                        help='label of experiment.')
    args, rest_args = parser.parse_known_args()
    exp_label = args.exp_label
    
    # --- Procgen Baseline ---
    
    if exp_label == 'coinrun_ppo':
        args = args_coinrun_ppo.get_args(rest_args)
    elif exp_label == 'maze_ppo':
        args = args_maze_ppo.get_args(rest_args)   
    else:
        raise NotImplementedError     
    
    # warning
    if args.deterministic_execution:
        print('Envoking deterministic code execution.')
        if torch.backends.cudnn.enabled:
            warnings.warn('Running with deterministic CUDNN.')
        if args.num_processes > 1:
            raise RuntimeError('If you want fully deterministic code, run it with num_processes=1.'
                               'Warning: This will slow things down and might break A2C if '
                               'policy_num_steps < env._max_episode_steps.')    
    

    # start training
    learner = Learner(args)
    print("Training beginning")
    learner.train()
    
    
if __name__ == '__main__':
    main()