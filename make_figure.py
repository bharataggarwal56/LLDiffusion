"""Qualitative figure + results tables from eval_lol.py outputs.

Reads <run>/eval/results.json and <run>/eval/images/<name>.png for each run folder
(missing runs are skipped) and writes, into --out-dir:
  figure.png       Low-light input | run 1 | run 2 | ... | Ground truth, each output labelled with PSNR
  figure_zoom.png  same grid, each image cropped to the region where ours improves most on baseline
  results.md / results.tex   mean PSNR↑ SSIM↑ LPIPS↓, best value per column in bold

The first run is the baseline. Unless --images is given, the 4 images where the best
non-baseline run beats the baseline by the largest PSNR margin are shown.
"""
import argparse
import json
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

DEFAULT_RUNS = ["outputs/lol_baseline", "outputs/lol_hvi", "outputs/lol_hvi_freq"]
PRETTY = {"lol_baseline": "Baseline", "lol_hvi": "HVI", "lol_hvi_freq": "HVI+Freq"}
METRICS = [("PSNR", "PSNR↑", True), ("SSIM", "SSIM↑", True), ("LPIPS", "LPIPS↓", False)]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="*", default=DEFAULT_RUNS,
                   help="run folders, baseline first (missing ones are skipped)")
    p.add_argument("--low", default="data/sets/LOL/test/low", help="test low-light folder")
    p.add_argument("--high", default="data/sets/LOL/test/high", help="test ground-truth folder")
    p.add_argument("--names", default=None,
                   help="comma-separated column labels, one per run folder given (default: derived from folder name)")
    p.add_argument("--images", default=None,
                   help="comma-separated test image names to show, e.g. 1,22,23,55 (default: auto-pick)")
    p.add_argument("--num-images", type=int, default=4, help="how many images to auto-pick")
    p.add_argument("--crop-frac", type=float, default=0.3,
                   help="zoom window side as a fraction of the image side")
    p.add_argument("--cell-width", type=int, default=360, help="width in px of one grid cell")
    p.add_argument("--out-dir", default=".")
    return p.parse_args()


def load_runs(paths, names):
    if names is not None:
        names = [n.strip() for n in names.split(",")]
        if len(names) != len(paths):
            raise SystemExit(f"--names has {len(names)} entries for {len(paths)} run folders")
    else:
        names = [PRETTY.get(os.path.basename(os.path.normpath(p)), os.path.basename(os.path.normpath(p)))
                 for p in paths]
    runs = []
    for path, name in zip(paths, names):
        res_path = os.path.join(path, "eval", "results.json")
        if not os.path.isfile(res_path):
            print(f"skipping {path}: no {res_path}")
            continue
        with open(res_path) as f:
            res = json.load(f)
        runs.append({"name": name, "path": path, "res": res,
                     "img_dir": os.path.join(path, "eval", "images")})
    return runs


def pick_images(runs, n):
    """Images where the best non-baseline run beats the baseline PSNR by the most."""
    base, ours = runs[0], runs[1:]
    common = set(base["res"]["per_image"])
    for r in ours:
        common &= set(r["res"]["per_image"])
    gain = {k: max(r["res"]["per_image"][k]["PSNR"] for r in ours) - base["res"]["per_image"][k]["PSNR"]
            for k in common}
    ranked = sorted(gain, key=gain.get, reverse=True)[:n]
    print("auto-picked (PSNR gain over baseline): " + ", ".join(f"{k} (+{gain[k]:.2f})" for k in ranked))
    return ranked


def load_rgb(path, size=None):
    img = Image.open(path).convert("RGB")
    if size is not None and img.size != size:
        print(f"warning: resizing {path} {img.size} -> {size}")
        img = img.resize(size, Image.BICUBIC)
    return img


def best_crop(gt, base, ours, frac):
    """(left, top, right, bottom) of the window where the best output improves most over baseline."""
    g = np.asarray(gt, np.float32)
    err_base = np.abs(np.asarray(base, np.float32) - g).mean(-1)
    err_ours = np.min([np.abs(np.asarray(o, np.float32) - g).mean(-1) for o in ours], axis=0)
    diff = err_base - err_ours
    h, w = diff.shape
    ch, cw = max(1, int(h * frac)), max(1, int(w * frac))
    ii = np.pad(diff.cumsum(0).cumsum(1), ((1, 0), (1, 0)))  # integral image
    box = ii[ch:, cw:] - ii[:-ch, cw:] - ii[ch:, :-cw] + ii[:-ch, :-cw]
    top, left = np.unravel_index(np.argmax(box), box.shape)
    return int(left), int(top), int(left + cw), int(top + ch)


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def tag(img, text, f, pos="bottom"):
    """Overlay a label on a dark band at the bottom (or top) of img."""
    d = ImageDraw.Draw(img, "RGBA")
    l, t, r, b = d.textbbox((0, 0), text, font=f)
    pad = 4
    bh = b - t + 2 * pad
    y0 = img.height - bh if pos == "bottom" else 0
    d.rectangle([0, y0, r - l + 2 * pad, y0 + bh], fill=(0, 0, 0, 170))
    d.text((pad - l, y0 + pad - t), text, font=f, fill=(255, 255, 255, 255))


def make_grid(rows, col_titles, cell_w, out_path):
    """rows: list of (row_label, [(PIL image, overlay text or None), ...])."""
    f_head, f_tag = font(max(14, cell_w // 18)), font(max(12, cell_w // 22))
    gap = 4
    cells = [[im for im, _ in cols] for _, cols in rows]
    cell_h = max(round(im.height * cell_w / im.width) for r in cells for im in r)
    head_h = f_head.size + 14
    ncol = len(col_titles)
    canvas = Image.new("RGB", (ncol * cell_w + (ncol + 1) * gap,
                               head_h + len(rows) * (cell_h + gap) + gap), "white")
    d = ImageDraw.Draw(canvas)
    for c, title in enumerate(col_titles):
        x = gap + c * (cell_w + gap)
        l, t, r, b = d.textbbox((0, 0), title, font=f_head)
        d.text((x + (cell_w - (r - l)) / 2 - l, (head_h - (b - t)) / 2 - t), title, font=f_head, fill="black")
    for ri, (label, cols) in enumerate(rows):
        y = head_h + ri * (cell_h + gap)
        for c, (im, text) in enumerate(cols):
            cell = im.resize((cell_w, cell_h), Image.BICUBIC)
            if text:
                tag(cell, text, f_tag)
            if c == 0:
                tag(cell, f"#{label}", f_tag, pos="top")
            canvas.paste(cell, (gap + c * (cell_w + gap), y))
    canvas.save(out_path)
    print("wrote", out_path)


def fmt(v, key):
    return f"{v:.2f}" if key == "PSNR" else f"{v:.4f}"


def best_flags(runs):
    best = {}
    for key, _, higher in METRICS:
        vals = [r["res"]["mean"][key] for r in runs]
        target = max(vals) if higher else min(vals)
        best[key] = [v == target for v in vals]
    return best


def write_tables(runs, out_dir):
    best = best_flags(runs)
    show_n = len({r["res"]["num_images"] for r in runs}) > 1
    if show_n:
        print("warning: runs were evaluated on different numbers of images; adding an N column")

    def row(i, bold, esc):
        r = runs[i]
        cells = [esc(r["name"])] + [
            bold(fmt(r["res"]["mean"][k], k), best[k][i]) for k, _, _ in METRICS]
        if show_n:
            cells.append(str(r["res"]["num_images"]))
        return cells

    heads = ["Method"] + [h for _, h, _ in METRICS] + (["N"] if show_n else [])
    lines = ["| " + " | ".join(heads) + " |",
             "|" + "|".join([":--"] + ["--:"] * (len(heads) - 1)) + "|"]
    for i in range(len(runs)):
        lines.append("| " + " | ".join(row(i, lambda s, b: f"**{s}**" if b else s, str)) + " |")
    with open(os.path.join(out_dir, "results.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    def tex_esc(s):
        for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#")):
            s = s.replace(a, b)
        return s

    tex_heads = ["Method"] + [h.replace("↑", r"$\uparrow$").replace("↓", r"$\downarrow$")
                              for _, h, _ in METRICS] + (["N"] if show_n else [])
    tex = [r"\begin{tabular}{l" + "r" * (len(tex_heads) - 1) + "}",
           r"\toprule",
           " & ".join(tex_heads) + r" \\",
           r"\midrule"]
    for i in range(len(runs)):
        tex.append(" & ".join(row(i, lambda s, b: rf"\textbf{{{s}}}" if b else s, tex_esc)) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(out_dir, "results.tex"), "w") as f:
        f.write("\n".join(tex) + "\n")
    print("wrote", os.path.join(out_dir, "results.md"), "and results.tex")


def main():
    args = parse_args()
    runs = load_runs(args.runs, args.names)
    if not runs:
        raise SystemExit("no run folders with eval/results.json found")
    os.makedirs(args.out_dir, exist_ok=True)

    write_tables(runs, args.out_dir)

    if len(runs) < 2 and args.images is None:
        raise SystemExit("need a baseline plus at least one other run to auto-pick images; "
                         "pass --images to plot a single run")
    names = [s.strip() for s in args.images.split(",")] if args.images else pick_images(runs, args.num_images)
    for n in names:
        for r in runs:
            if n not in r["res"]["per_image"] or not os.path.isfile(os.path.join(r["img_dir"], f"{n}.png")):
                raise SystemExit(f"image {n} missing from run {r['path']}")

    full_rows, zoom_rows = [], []
    for n in names:
        low_p, gt_p = os.path.join(args.low, f"{n}.png"), os.path.join(args.high, f"{n}.png")
        for p in (low_p, gt_p):
            if not os.path.isfile(p):
                raise SystemExit(f"missing test image {p}")
        gt = load_rgb(gt_p)
        low = load_rgb(low_p, gt.size)
        outs = [load_rgb(os.path.join(r["img_dir"], f"{n}.png"), gt.size) for r in runs]
        box = best_crop(gt, outs[0], outs[1:], args.crop_frac) if len(outs) > 1 else \
            (0, 0, gt.width, gt.height)
        cols = [(low, None)] + [(o, f"PSNR {r['res']['per_image'][n]['PSNR']:.2f} dB")
                                for o, r in zip(outs, runs)] + [(gt, None)]
        full_rows.append((n, cols))
        zoom_rows.append((n, [(im.crop(box), t) for im, t in cols]))

    titles = ["Low-light input"] + [r["name"] for r in runs] + ["Ground truth"]
    make_grid(full_rows, titles, args.cell_width, os.path.join(args.out_dir, "figure.png"))
    make_grid(zoom_rows, titles, args.cell_width, os.path.join(args.out_dir, "figure_zoom.png"))


if __name__ == "__main__":
    main()
