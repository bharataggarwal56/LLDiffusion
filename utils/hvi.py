"""RGB -> HVI colour space, ported from HVI-CIDNet (net/HVI_transform.py, RGB_HVI.HVIT).

The original has a learnable density_k (init 0.2); here it is a fixed constant.
Input is RGB in [0, 1]; returns (H, V, I) with H, V in [-1, 1] and I in [0, 1].
"""
import torch
from math import pi


def rgb_to_hvi(img, k=0.2):
    eps = 1e-8
    r, g, b = img[:, 0], img[:, 1], img[:, 2]
    value = img.max(1)[0]
    img_min = img.min(1)[0]
    delta = value - img_min + eps

    hue = torch.zeros_like(value)
    is_b = b == value
    is_g = g == value
    is_r = r == value
    hue = torch.where(is_b, 4.0 + (r - g) / delta, hue)
    hue = torch.where(is_g, 2.0 + (b - r) / delta, hue)
    hue = torch.where(is_r, (0.0 + (g - b) / delta) % 6, hue)
    hue = torch.where(img_min == value, torch.zeros_like(hue), hue)
    hue = hue / 6.0

    saturation = (value - img_min) / (value + eps)
    saturation = torch.where(value == 0, torch.zeros_like(saturation), saturation)

    hue = hue.unsqueeze(1)
    saturation = saturation.unsqueeze(1)
    value = value.unsqueeze(1)

    color_sensitive = ((value * 0.5 * pi).sin() + eps).pow(k)
    ch = (2.0 * pi * hue).cos()
    cv = (2.0 * pi * hue).sin()
    H = color_sensitive * saturation * ch
    V = color_sensitive * saturation * cv
    return torch.cat([H, V, value], dim=1)
