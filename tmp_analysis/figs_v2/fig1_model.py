"""Figure 1 — the model: a two-stream network fed magno- and parvo-like inputs,
with a developmental diet on the luminance channel that matures over training.

  (a) pipeline schematic: RGB -> DKL -> luminance (+ the DVD acuity blur and
      contrast-sensitivity cut, both at one age that rises from birth to 300
      months over training) -> stream 1; L−M -> stream 2; affine readout over the
      weighted sum of the two streams' features. The networks compared.
  (b) what each stream sees, on one NSD image: the DKL channels, and the luminance
      channel through the diet at three ages of the trajectory (30, 150 and 300
      months: epochs 5, 25 and 50); the 300-month endpoint is every network's test
      input and the control's training input throughout. Rendered with the
      training front end (src/analysis/encoding/dkl_frontend.apply_dvd_diet).
  (c) validation accuracy over training: the model, its control, the BA diet under
      the model's recipe and a trajectory stopping at 48 months (50 epochs, left);
      the BA network under the thesis's recipe (150 epochs, right).

Usage: PYTHONPATH=.:tmp_analysis/figs_v2 python tmp_analysis/figs_v2/fig1_model.py [nsd_image_id]
"""
from __future__ import annotations

import json
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import style as S
import v3runs as R

CONDS = [("model", R.MODEL, S.MODEL, "-"),
         ("control", R.CONTROL, S.CONTROL, "-"),
         ("fixed BA diet, model recipe", R.FIXEDREC, S.FIXEDREC, (0, (3, 2))),
         ("diet maturing 0→48 mo", R.DVD48, S.DVD48, (0, (1, 1.5)))]
BA_CONDS = [("BA network (fixed diet)", R.BA, S.BA), ("BA network, no diet", R.BANODIET, S.BANODIET)]
# the CS cut keeps no energy above 1/8 of Nyquist until ~120 months, so detail returns
# only in the second half of training: epoch 5 (magno bias already in place), 25, 50
AGES = [(30, "epoch 5"), (150, "epoch 25"), (300, "epoch 50 = test")]
IMG_ID = int(sys.argv[1]) if len(sys.argv) > 1 else -1


# ----------------------------------------------------------------- panel (b)
def pick_image(F, cmin, cmax, n=160):
    """The NSD image with the most red–green variance among a fixed candidate
    grid, so the chromatic tile shows what the parvo stream actually receives."""
    ids = np.arange(0, 73000, 73000 // n)[:n]
    rgb = F.load_nsd_images(ids)
    x3 = F.rgb_to_dkl3(rgb, cmin, cmax)
    sd = x3[:, 1].reshape(len(ids), -1).std(1).numpy()
    best = int(ids[int(np.argmax(sd))])
    print(f"  most chromatic candidate: NSD id {best} (rg sd {sd.max():.3f})")
    return best


def stream_inputs(img_id: int):
    import torch
    from src.analysis.encoding import dkl_frontend as F
    mm = json.load(open("src/analysis/encoding/dkl_channel_minmax.json"))
    cmin, cmax = np.array(mm["min"]), np.array(mm["max"])
    if img_id < 0:
        img_id = pick_image(F, cmin, cmax)
    rgb = F.load_nsd_images(np.array([img_id]))          # (1, 425, 425, 3)
    x3 = F.rgb_to_dkl3(rgb, cmin, cmax)                   # (1, 3, 256, 256)
    lum, rg = x3[:, 0:1], x3[:, 1:2]
    with torch.no_grad():
        aged = [F.apply_dvd_diet(lum, float(a))[0, 0].cpu().numpy() for a, _ in AGES]
    return rgb[0], lum[0, 0].cpu().numpy(), rg[0, 0].cpu().numpy(), aged


def rg_to_rgb(rg01: np.ndarray) -> np.ndarray:
    """Render the L−M channel as a red–green opponent image around grey."""
    # contrast-stretched for display only: the stored channel spans the whole
    # dataset's range, so a single image occupies a narrow band around 0.5.
    dev = rg01 - 0.5
    scale = np.percentile(np.abs(dev), 99) + 1e-6
    v = np.clip(dev / scale, -1, 1)
    out = np.empty(rg01.shape + (3,))
    out[..., 0] = 0.55 + 0.45 * np.clip(v, 0, 1) - 0.35 * np.clip(-v, 0, 1)
    out[..., 1] = 0.55 + 0.45 * np.clip(-v, 0, 1) - 0.35 * np.clip(v, 0, 1)
    out[..., 2] = 0.55 - 0.30 * np.abs(v)
    return np.clip(out, 0, 1)


# ----------------------------------------------------------------- panel (a)
def box(ax, x, y, w, h, text, fc="white", ec=S.INK, lw=0.8, fs=6.8, color=S.INK,
        style="round,pad=0.02,rounding_size=0.06", weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=color, fontweight=weight, linespacing=1.15)


def arrow(ax, x0, y0, x1, y1, color=S.MUTED, lw=0.9):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=7, color=color, lw=lw,
                                 shrinkA=0, shrinkB=0))


def schematic(ax):
    ax.set_xlim(0, 10); ax.set_ylim(0, 6.4); ax.axis("off")
    FS = 5.9
    GREY, EDGE = "#F4F4F4", "#B0B0B0"
    # input and colour space
    box(ax, 0.10, 3.55, 1.05, 0.90, "RGB\nimage", fc=GREY, ec=EDGE, fs=FS)
    box(ax, 1.45, 3.55, 1.30, 0.90, "DKL\ncone-opponent\nspace", fc=GREY, ec=EDGE, fs=5.3)
    arrow(ax, 1.15, 4.00, 1.45, 4.00)
    # two channels
    arrow(ax, 2.75, 4.25, 3.15, 4.95)
    arrow(ax, 2.75, 3.75, 3.15, 3.05)
    box(ax, 3.15, 4.62, 1.20, 0.66, "luminance\nL + M", fc=GREY, ec=EDGE, fs=FS)
    box(ax, 3.15, 2.72, 1.20, 0.66, "chromatic\nL − M", fc=GREY, ec=EDGE, fs=FS)
    # the maturing diet on luminance
    arrow(ax, 4.35, 4.95, 4.60, 4.95)
    box(ax, 4.60, 4.36, 1.95, 1.18, "developmental diet\nacuity blur + CS cut\nat one age, rising\n0 → 300 mo in training", fc="#E3EFF8", ec=S.MODEL, fs=5.0)
    arrow(ax, 6.55, 4.95, 6.85, 4.95)
    # chromatic straight through
    arrow(ax, 4.35, 3.05, 6.85, 3.05)
    # streams
    box(ax, 6.85, 4.52, 1.45, 0.86, "stream 1\nResNet-18\n“magno”", fc="white", ec=S.INK, lw=1.2, fs=5.9, weight="bold")
    box(ax, 6.85, 2.62, 1.45, 0.86, "stream 2\nResNet-18\n“parvo”", fc="white", ec=S.INK, lw=1.2, fs=5.9, weight="bold")
    ax.text(7.57, 4.05, "trunks bit-identical at init.", ha="center", va="center", fontsize=4.9, color=S.MUTED, style="italic")
    ax.text(8.36, 4.95, "y₁", fontsize=7, va="center", ha="left", color=S.INK, fontweight="bold")
    ax.text(8.36, 3.05, "y₂", fontsize=7, va="center", ha="left", color=S.INK, fontweight="bold")
    # readout
    arrow(ax, 8.62, 4.80, 9.12, 4.25)
    arrow(ax, 8.62, 3.20, 9.12, 3.75)
    box(ax, 9.05, 3.62, 0.76, 0.76, "+", fc=GREY, ec=EDGE, fs=10)
    ax.text(9.43, 3.45, "w·y₁ + (1−w)·y₂\nw = 0.5\none cross-entropy", ha="center", va="top", fontsize=5.2, color=S.INK)
    arrow(ax, 9.43, 4.38, 9.43, 4.75)
    ax.text(9.43, 4.80, "linear classifier\n112 classes", ha="center", va="bottom", fontsize=5.2, color=S.INK)
    # the networks compared
    y0 = 2.05
    ax.text(0.10, y0 + 0.30, "networks compared", fontsize=6.4, fontweight="bold", va="top", color=S.INK)
    ax.text(2.35, y0, "stream 1 trains on", fontsize=5.8, color=S.MUTED, va="top")
    ax.text(7.35, y0, "recipe", fontsize=5.8, color=S.MUTED, va="top")
    rows = [("model", "luminance + diet maturing 0 → 300 mo", "50 epochs", S.MODEL),
            ("control", "luminance + 300-mo diet from step 0", "50 epochs", S.CONTROL),
            ("BA network", "luminance + blur σ 4 px + CS at 48 mo", "thesis, 150 ep.", S.BA),
            ("BA, no diet", "luminance", "thesis, 150 ep.", S.BANODIET)]
    ax.plot([0.10, 9.9], [y0 - 0.30, y0 - 0.30], "-", color="#D0D0D0", lw=0.6)
    for i, (name, s1, rec, c) in enumerate(rows):
        yy = y0 - 0.55 - 0.40 * i
        ax.plot([0.13, 0.40], [yy, yy], "-", color=c, lw=3, solid_capstyle="butt")
        ax.text(0.55, yy, name, fontsize=6.0, va="center", color=S.INK, fontweight="bold")
        ax.text(2.35, yy, s1, fontsize=6.0, va="center", color=S.INK)
        ax.text(7.35, yy, rec, fontsize=6.0, va="center", color=S.INK)
    ax.text(0.13, y0 - 0.55 - 0.40 * 4 + 0.05, "stream 2 trains on L − M in every network; the model and control are tested on the 300-month endpoint",
            fontsize=5.2, va="center", color=S.MUTED, style="italic")


# ----------------------------------------------------------------- panel (c)
def curves(conds):
    out = {}
    for label, tmpl, *rest in conds:
        runs = [R.history(r) for r in R.finished(tmpl)]
        if runs:
            out[label] = (np.array([[e["val_top1"] for e in h] for h in runs]), *rest)
    return out


def draw_curves(ax, H, ep):
    for label, (runs, c, *ls) in H.items():
        ls = ls[0] if ls else "-"
        for r in runs:
            ax.plot(ep, r, ls=ls, color=c, lw=0.6, alpha=0.35, zorder=2)
        ax.plot(ep, runs.mean(0), ls=ls, color=c, lw=1.8, zorder=3, label=f"{label} ({R.fmt_seeds(len(runs))})")
        fin = runs[:, -1]
        print(f"  {label:30s} final {fin.mean():.3f} [{fin.min():.3f}, {fin.max():.3f}]  "
              f"best {runs.max(1).mean():.3f}  n={len(runs)}")


def main():
    fig = plt.figure(figsize=(S.FULL, 5.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.08, 0.85], width_ratios=[1.45, 1.0],
                          hspace=0.42, wspace=0.18, left=0.045, right=0.985, top=0.94, bottom=0.085)

    # (a)
    axA = fig.add_subplot(gs[0, 0])
    schematic(axA)
    S.letter(axA, "a", dx=-0.01, dy=0.005, title="A two-stream network whose magno input matures")

    # (b)
    rgb, lum, rg, aged = stream_inputs(IMG_ID)
    gb = gs[0, 1].subgridspec(2, 3, wspace=0.06, hspace=0.34)
    tiles = [(rgb, "RGB image (NSD)"), (lum, "luminance  L + M\n→ stream 1"), (rg_to_rgb(rg), "chromatic  L − M\n→ stream 2")]
    tiles += [(a, f"stream 1 at {age} mo\n({when})") for a, (age, when) in zip(aged, AGES)]
    first_ax = None
    for i, (im, lab) in enumerate(tiles):
        ax = fig.add_subplot(gb[i // 3, i % 3])
        ax.axis("off")
        first_ax = first_ax or ax
        if im.ndim == 2:
            ax.imshow(im, cmap="gray", vmin=0, vmax=1, interpolation="bilinear")
        else:
            ax.imshow(im, interpolation="bilinear")
        ax.set_title(lab, fontsize=5.6, color=S.INK, pad=2.5, linespacing=1.1)
    S.letter(first_ax, "b", dx=-0.012, dy=0.035, title="What stream 1 sees as the diet matures")

    # (c) 50-epoch recipe | thesis recipe
    gc = gs[1, :].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.10)
    axC = fig.add_subplot(gc[0, 0])
    draw_curves(axC, curves(CONDS), np.arange(1, 51))
    axC.set_xlim(0, 51); axC.set_ylim(0, 0.80); axC.set_xticks([0, 10, 20, 30, 40, 50])
    axC.set_xlabel("training epoch (model recipe)")
    axC.set_ylabel("validation top-1 accuracy\n(MiniEcoset, 112 classes)")
    axC.legend(loc="lower right", fontsize=5.8, handlelength=2.0, labelspacing=0.25)
    sec = axC.secondary_xaxis("top", functions=(lambda e: 6 * e, lambda a: a / 6))
    sec.set_xticks([0, 60, 120, 180, 240, 300]); sec.tick_params(labelsize=5.8)
    sec.set_xlabel("diet age, model (months)", fontsize=6.0, color=S.MUTED, labelpad=1.5)
    S.letter(axC, "c", dx=-0.012, dy=0.075, title="The trajectory costs accuracy, not stability")

    axD = fig.add_subplot(gc[0, 1], sharey=axC)
    draw_curves(axD, curves(BA_CONDS), np.arange(1, 151))
    for e in range(25, 151, 25):
        axD.axvline(e, color="#E6E6E6", lw=0.6, zorder=0)
    axD.set_xlim(0, 152); axD.set_xticks([0, 25, 50, 75, 100, 125, 150])
    axD.set_xlabel("training epoch (thesis recipe)")
    axD.tick_params(labelleft=False)
    axD.legend(loc="lower right", fontsize=5.8, handlelength=2.0, labelspacing=0.25)
    S.note(axD, "thin: seeds; thick: seed mean; rules: saved checkpoints", x=0.02, y=0.985, size=5.8)

    S.save(fig, "fig1_model")


if __name__ == "__main__":
    main()
