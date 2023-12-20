import os
from argparse import ArgumentParser

import torch
from utils import parse_config
from models import initialize_gcn_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Working on {device}")


def main():
    parser = ArgumentParser()
    parser.add_argument('dataset_dir', help='dataset directory path')
    parser.add_argument('config_file', help='config file')
    parser.add_argument('exp_dir', help='experiment directory')
    parser.add_argument('--resume', action='store_true', default=False, help='False: start from scratch, ')
    parser.add_argument('--test', action='store_true', default=False, help='False: no testing'
                                                                           'True: testing on the test set at the end')
    parser.add_argument('--log_test_results', action='store_true', default=False, help='False: no logging'
                                                                                       'True: logging test results')

    args = parser.parse_args()
    if not os.path.exists(args.config_file):
        raise FileNotFoundError(f"{args.config_file} could not be found!")
    if not os.path.exists(args.dataset_dir):
        raise FileNotFoundError(f"{args.dataset_dir} could not be found!")
    cfg = parse_config(args.config_file)
    model, optimizer, loss_fn = initialize_gcn_model(cfg, device)
    print(model)


if __name__ == '__main__':
    main()
