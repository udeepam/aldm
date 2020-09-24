import torch

from utils.cli import boolean_argument

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

    args = base_parser.parse_args(rest_args)
    args.cuda = torch.cuda.is_available()

    return args
