import argparse
import glob
import os

import yaml

DDIM_STEPS = 25


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            value = dict2namespace(value)
        setattr(namespace, key, value)
    return namespace


def apply_overrides(config, overrides):
    """Apply `a.b.c=value` strings to a plain dict; values are parsed as YAML."""
    for item in overrides or []:
        key, _, value = item.partition("=")
        node = config
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(value)
    return config


def load_config(path, overrides=None, data_root=None):
    with open(path) as f:
        config = yaml.safe_load(f)
    if data_root is not None:
        config.setdefault("data", {})["root"] = data_root
    apply_overrides(config, overrides)
    return dict2namespace(config)


def add_common_args(parser):
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data-root", default=None, type=str,
                        help="Overrides data.root from the config")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="Config overrides, e.g. training.max_iters=2")


def latest_checkpoint(out_dir):
    """Most recently written resumable checkpoint (ckpt-*.pth or final.pth)."""
    paths = glob.glob(os.path.join(out_dir, "ckpt-*.pth"))
    final = os.path.join(out_dir, "final.pth")
    if os.path.exists(final):
        paths.append(final)
    if not paths:
        return None
    return max(paths, key=os.path.getmtime)


def resolve_resume(resume, config):
    """`--resume auto` -> latest checkpoint in training.out_dir (or '' if none)."""
    if resume != "auto":
        return resume
    out_dir = getattr(getattr(config, "training", None), "out_dir", None)
    if out_dir is None:
        raise ValueError("--resume auto needs training.out_dir in the config")
    return latest_checkpoint(out_dir) or ""
