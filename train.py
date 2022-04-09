import argparse
import yaml

import torch


def train():
    return


def train_epoch():
    return


def validate():
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True, help="Path to yaml config file")
    parser.add_argument('--force-cpu', action='store_true', help='Force PyTorch to use cpu when CUDA is available.')
    args = parser.parse_args()

    # Load config file
    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)
        settings = cfg['settings']
        train_cfg = cfg['train_params']
        opt_cfg = cfg['optimizer_params']
        data_cfg = cfg['dataset_params']
        model_cfg = cfg['model_params']