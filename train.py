import argparse

import numpy as np
import torch

import datasets
from models import DenoisingDiffusion
from utils.config import DDIM_STEPS, add_common_args, load_config


def parse_args_and_config():
    parser = argparse.ArgumentParser(
        description='Training Patch-Based Denoising Diffusion Models')
    add_common_args(parser)
    parser.add_argument('--resume', default='', type=str,
                        help="Checkpoint to resume from, or 'auto' for the latest "
                             "one in training.out_dir")
    parser.add_argument("--sampling_timesteps", type=int, default=DDIM_STEPS,
                        help="Number of implicit sampling steps for validation image patches")
    parser.add_argument("--image_folder", default='results/images/', type=str,
                        help="Location to save restored validation image patches")
    parser.add_argument('--seed', default=None, type=int, metavar='N',
                        help='Random seed (default: config seed, else 0)')
    args = parser.parse_args()

    config = load_config(args.config, args.set, args.data_root)
    if args.seed is None:
        args.seed = getattr(config, "seed", 0)

    return args, config


def main():
    args, config = parse_args_and_config()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    DATASET = datasets.__dict__[config.data.dataset](config)

    # create model
    diffusion = DenoisingDiffusion(args, config)
    diffusion.train(DATASET)


if __name__ == "__main__":
    main()
