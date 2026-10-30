"""Evaluate a checkpoint on the LOL test split: PSNR / SSIM / LPIPS -> results.json."""
import argparse
import json
import os

import lpips
import numpy as np
import torch
from torchvision.transforms.functional import convert_image_dtype
from torchvision.utils import save_image

import datasets
from models import DenoisingDiffusion
from utils.config import DDIM_STEPS, add_common_args, load_config, resolve_resume
from utils.metrics import calculate_psnr, calculate_ssim


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--resume", default="auto", type=str,
                        help="Checkpoint path, or 'auto' for the latest in training.out_dir")
    parser.add_argument("--ema", action="store_true", help="Evaluate the EMA weights")
    parser.add_argument("--out", default=None, type=str,
                        help="Output dir (default: <training.out_dir>/eval)")
    parser.add_argument("-b", "--batch-size", type=int, default=512)
    parser.add_argument("-g", "--grid_r", type=int, default=16)
    parser.add_argument("-t", "--sampling_timesteps", type=int, default=DDIM_STEPS)
    parser.add_argument("--no-patch", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Only the first N test images")
    parser.add_argument("--seed", default=None, type=int)
    args = parser.parse_args()
    config = load_config(args.config, args.set, args.data_root)
    if args.seed is None:
        args.seed = getattr(config, "seed", 0)
    return args, config


def main():
    args, config = parse_args_and_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config.device = device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out = args.out or os.path.join(config.training.out_dir, "eval")
    img_dir = os.path.join(out, "images")
    os.makedirs(img_dir, exist_ok=True)

    test_loader = datasets.__dict__[config.data.dataset](config).test_loader()
    model = DenoisingDiffusion(args, config)
    resume = resolve_resume(args.resume, config)
    if resume == "":
        raise SystemExit("no checkpoint found; pass --resume <path>")
    model.load(resume, ema=args.ema)
    model.model.eval()

    lpips_fn = lpips.LPIPS(net="alex").to(device).eval()
    grid_r = None if args.no_patch else args.grid_r

    per_image = {}
    for (lows, highs), labels in test_loader:
        for low, high, label in zip(lows, highs, labels):
            if args.limit is not None and len(per_image) >= args.limit:
                break
            name = str(label.item())
            restored = model.restore_single(
                low.to(device), grid_r, high.shape[-2:], batch_size=args.batch_size)
            assert torch.isfinite(restored).all(), f"NaN/inf in output for image {name}"
            # score the 8-bit image that is actually saved
            restored = convert_image_dtype(restored.cpu(), torch.uint8)
            gt = convert_image_dtype(high.cpu(), torch.uint8)
            save_image(convert_image_dtype(restored, torch.float32),
                       os.path.join(img_dir, f"{name}.png"))
            with torch.no_grad():
                lp = lpips_fn(convert_image_dtype(restored, torch.float32)[None].to(device) * 2 - 1,
                              convert_image_dtype(gt, torch.float32)[None].to(device) * 2 - 1).item()
            per_image[name] = {
                "PSNR": float(calculate_psnr(restored, gt)),
                "SSIM": float(calculate_ssim(restored, gt)),
                "LPIPS": lp,
            }
            print(name, per_image[name])

    mean = {k: float(np.mean([v[k] for v in per_image.values()]))
            for k in ("PSNR", "SSIM", "LPIPS")}
    results = {
        "config": args.config,
        "checkpoint": resume,
        "ema": args.ema,
        "num_images": len(per_image),
        "sampling_timesteps": args.sampling_timesteps,
        "grid_r": grid_r,
        "mean": mean,
        "per_image": per_image,
    }
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("mean:", mean, "->", os.path.join(out, "results.json"))


if __name__ == "__main__":
    main()
