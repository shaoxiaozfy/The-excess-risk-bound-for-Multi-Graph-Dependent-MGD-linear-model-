import argparse
import os
from utils.get_path import get_project_path


def base_opt_parser():
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument('--opt', type=str, default='sgd')
    parser.add_argument('--lr', type=float, default=0.02)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    # sgd
    parser.add_argument('--momentum', type=float, default=0.9, help="Momentum for SGD.(None means the default in optm)")
    parser.add_argument('--nesterov', action="store_true")
    # adam
    parser.add_argument('--betas', type=float, default=None, nargs='+',
                        help="Betas for AdamW Optimizer.(None means the default in optm)")
    parser.add_argument('--eps', type=float, default=None,
                        help="Epsilon for AdamW Optimizer.(None means the default in optm)")
    return parser


def sam_opt_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--rho', type=float, default=0.05, help="Perturbation intensity of SAM type optims.")
    parser.add_argument('--sparsity', type=float, default=0.2,
                        help="The proportion of parameters that do not calculate perturbation.")
    parser.add_argument('--update_freq', type=int, default=5, help="Update frequency (epoch) of sparse SAM.")

    parser.add_argument('--num_samples', type=int, default=1024,
                        help="Number of samples to compute fisher information. Only for `ssam-f`.")
    parser.add_argument('--drop_rate', type=float, default=0.5, help="Death Rate in `ssam-d`. Only for `ssam-d`.")
    parser.add_argument('--drop_strategy', type=str, default='gradient', help="Strategy of Death. Only for `ssam-d`.")
    parser.add_argument('--growth_strategy', type=str, default='random', help="Only for `ssam-d`.")
    return parser


def my_opt_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--cuda", type=str, default="0", help="Select zero-indexed cuda device. -1 to use CPU.", )
    parser.add_argument("--do_initial", type=bool, default=False, )
    parser.add_argument("--seed", type=int, default=2222)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument('--epoch', type=int, default=40)
    parser.add_argument('--dataset_name', type=str, default='voc')
    parser.add_argument("--measure", type=str, default='micro-auc')
    parser.add_argument('--model_name', type=str, default='lm')
    parser.add_argument('--num_workers', type=int, default=2)
    return parser

def language_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--update_count', type=int, required=False, default=1)
    parser.add_argument('--bert', type=str, required=False, default='bert-base')
    parser.add_argument('--max_len', type=int, required=False, default=512)

    parser.add_argument('--valid', action='store_true')

    parser.add_argument('--swa', action='store_true')
    parser.add_argument('--swa_warmup', type=int, required=False, default=10)
    parser.add_argument('--swa_step', type=int, required=False, default=100)

    parser.add_argument('--group_y_group', type=int, default=0)
    parser.add_argument('--group_y_candidate_num', type=int, required=False, default=3000)
    parser.add_argument('--group_y_candidate_topk', type=int, required=False, default=10)

    parser.add_argument('--eval_step', type=int, required=False, default=20000)

    parser.add_argument('--hidden_dim', type=int, required=False, default=300)

    parser.add_argument('--eval_model', action='store_true')

    parser.add_argument('--loss_name', type=str, required=False, default='u3')
    return parser


def get_args(out_parsers=None):
    all_parser_funcs = [base_opt_parser, sam_opt_parser, my_opt_parser, language_parser]

    all_parsers = [parser_func() for parser_func in all_parser_funcs]
    if out_parsers:
        all_parsers.append(out_parsers)

    final_parser = argparse.ArgumentParser(parents=all_parsers)
    args = final_parser.parse_args()
    return args


if __name__ == '__main__':
    print(get_args())
