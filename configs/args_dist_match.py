import argparse

import torch

from utils.cli import boolean_argument


from configs.args_ppo import get_args as get_base_args


def get_args(rest_args):

    base_parser = get_base_args()

    # --- BOTTLENECK ---
    base_parser.add_argument("--use_bottleneck", type=boolean_argument, default=True,
                             help='Whether to use the variational information bottleneck (default: True).')
    base_parser.add_argument("--vib_coef", type=float, default=1e-4,
                             help='DVIB coefficient in front of KL divergence term in loss. Operates as a trade-off \
                             parameter between the complexity rate of the representation I(s;z) and the \
                             amount of preserved relevant information I(a;z).')

    # --- DISTRIBUTION MATCHING ---
    base_parser.add_argument("--use_dist_matching", type=boolean_argument, default=True,
                             help='Whether to optimise the distribution matching loss.')

    # hyperparameters
    base_parser.add_argument("--dist_matching_loss", type=str, default="kl",
                             help='Which divergence to use for calculating the loss for distribution matching {kl, jsd}.')
    base_parser.add_argument("--dist_matching_coef", type=float, default=1e-3,
                             help='Coefficient in front distribution matching loss term.')

    # splitting train envs and levels
    base_parser.add_argument("--percentage_levels_train", type=float, default=0.8,
                             help='Proportion of the train levels to use for train and the rest is used for validation. Range is [0,1]')
    base_parser.add_argument("--num_val_envs", type=int, default=10,
                             help='Number of environments from --num_processes to use for validation.')

    args = base_parser.parse_args(rest_args)
    args.cuda = torch.cuda.is_available()

    return args
