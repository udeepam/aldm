import argparse

import torch

from utils.cli import boolean_argument, none_or_str

from configs.args_ppo import get_args as get_base_args


def get_args(rest_args):

    base_parser = get_base_args()

    # --- BOTTLENECK ---
    base_parser.add_argument("--use_bottleneck", type=boolean_argument, default=True,
                             help='Whether to use the variational information bottleneck (default: True).')
    base_parser.add_argument("--vib_coef", type=float, default=1e-4,
                             help='VIB coefficient in front of KL divergence term in loss. Operates as a trade-off \
                             parameter between the complexity rate of the representation I(s;z) and the \
                             amount of preserved relevant information I(a;z).')

    # --- SNI ---
    base_parser.add_argument("--sni_type", type=none_or_str, default='vib',
                             help='Type of selective noise injection {None, vib}.')
    base_parser.add_argument("--sni_coef", type=float, default=0.5,
                             help='Coefficient for SNI.')

    args = base_parser.parse_args(rest_args)
    args.cuda = torch.cuda.is_available()

    return args
