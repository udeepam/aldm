"""
Main script to start experiments.
"""
import os
import argparse
import warnings

import torch

# get configs
from configs.procgen import args_procgen_ppo, args_procgen_ibac, args_procgen_ibac_sni, args_procgen_dist_match

from learner import Learner


# environment variable sets the number of threads to use for parallel regions
# https://github.com/ray-project/ray/issues/6962
os.environ["OMP_NUM_THREADS"] = "1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='procgen:procgen-coinrun-v0',
                        help='name of the environment to train on.')
    parser.add_argument('--model', type=str, default='ppo',
                        help='the model to use for training. {ppo, ibac, ibac-sni, ours}')
    args, rest_args = parser.parse_known_args()
    env_name = args.env_name
    model = args.model

    # --- OPENAI PROCGEN ---

    if env_name.startswith('procgen'):
        if model == 'ppo':
            args = args_procgen_ppo.get_args(rest_args)
        elif model == 'ibac':
            args = args_procgen_ibac.get_args(rest_args)
        elif model == 'ibac_sni':
            args = args_procgen_ibac_sni.get_args(rest_args)
        elif model == 'dist_match':
            args = args_procgen_dist_match.get_args(rest_args)
        else:
            raise NotImplementedError

    else:
        raise NotImplementedError

    # warnings
    if args.deterministic_execution:
        print('Envoking deterministic code execution.')
        if torch.backends.cudnn.enabled:
            warnings.warn('Running with deterministic CUDNN.')
        if args.num_processes > 1:
            raise RuntimeError('If you want fully deterministic code, run it with num_processes=1.'
                               'Warning: This will slow things down and might break A2C if '
                               'policy_num_steps < env._max_episode_steps.')
    elif args.use_distribution_matching and (not args.use_bottleneck or not args.percentage_levels_train<1.0 or not args.num_val_envs>0 or args.num_val_envs>args.num_processes):
        raise ValueError('If --use_distribution_matching=True then you must also have'
                         '--use_bottleneck=True and --percentage_levels_train<1.0 and --num_processes>--num_val_envs>0.')
    elif args.sni_type is not None and args.recurrent_policy:
        raise NotImplementedError

    # place other args back into argparse.Namespace
    args.env_name = env_name
    args.model = model
    args.num_train_envs = args.num_processes-args.num_val_envs if args.use_distribution_matching else args.num_processes

    # start training
    learner = Learner(args)
    print("Training beginning")
    learner.train()

    # start testing
    if args.test:
        print("Testing beginning")
        learner.test()

    # analyse the learnt representation
    if args.use_bottleneck:
        print("Analysing latent representation")
        learner.analyse_representation()


if __name__ == '__main__':
    main()
