"""Shared style for the manuscript figures.

One palette, one type scale, one panel-letter convention, so the figures read
as a set. Okabe–Ito colours throughout (colour-vision-deficiency safe).

Since 2026-10-20 the scripts draw PAPER_v3 (model of record dvd300) into
results/figures_v3/; the committed PAPER_v2 figures in results/figures_v2/ were
drawn by these scripts at commit 7673c18. FIG_OUT overrides the output directory.

PAPER_v3 networks (colour = network; the two blues hold a shape-biased magno stream)
  MODEL    fs2_dvd300_avg_w050_ls     luminance + diet maturing 0 -> 300 mo | L−M
  CONTROL  fs2_dvdfix300_avg_w050_ls  luminance + 300-mo diet from step 0  | L−M
  BA       dkl_bcs                    luminance + blur σ 4 px + CS 48 mo    | L−M (thesis recipe)
  BANODIET dkl_base                   luminance                             | L−M (thesis recipe)
  FIXEDREC fs2_bcs_avg_w050_ls        the BA diet under the model's recipe
  DVD48    fs2_dvd48_avg_w050_ls      diet maturing 0 -> 48 mo (one seed)

PAPER_v2 conditions (the BA network)
  DIET    dkl_bcs             luminance + developmental transform | L−M
  NODIET   dkl_base            luminance                            | L−M
  SWAP     dkl_bcs_swap        luminance                            | L−M + transform
  HIGHSF   dkl_bcs_parvo_highsf luminance + transform               | L−M + luminance
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.environ.get("FIG_OUT", "results/figures_v3")

DIET = "#0072B2"      # blue
DIETL = "#7FBDE3"     # light blue (second checkpoint policy / secondary diet arm)
NODIET = "#D55E00"    # vermillion
SWAP = "#009E73"      # green
HIGHSF = "#CC79A7"    # purple
MATCH = "#E69F00"     # orange: accuracy-matched control
SINGLE = "#8C8C8C"    # grey: single-stream controls
PUB = "#5A5A5A"       # published CNNs
HUMAN = "#000000"
UNTRAINED = "#6E6E6E"
INK = "#1A1A1A"
MUTED = "#6F6F6F"
RULE = "#C8C8C8"
MAGNO = DIET
PARVO = NODIET

MODEL = "#0072B2"     # blue: dvd300, the model of record
CONTROL = "#D55E00"   # vermillion: the adult diet from the first step
BA = "#56B4E9"        # sky blue: the BA network (fixed diet, thesis recipe)
BANODIET = "#E69F00"  # orange: the BA no-diet arm
FIXEDREC = "#009E73"  # green: the BA diet under the model's recipe
DVD48 = "#CC79A7"     # purple: the trajectory stopped at 48 months
NET_LABEL = {"model": "model (diet maturing 0→300 mo)", "control": "control (adult diet from step 0)",
             "BA": "BA network (fixed diet)", "BA no diet": "BA network, no diet"}

COND_COLOUR = {"diet": DIET, "no diet": NODIET, "swap": SWAP, "parvo_highsf": HIGHSF}
COND_LABEL = {"diet": "diet on luminance", "no diet": "no diet",
              "swap": "diet on L−M (swap)", "parvo_highsf": "parvo + high-SF luminance"}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7.5,
    "axes.labelsize": 7.5, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.edgecolor": "#8A8A8A", "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.labelcolor": INK, "text.color": INK,
    "legend.frameon": False, "legend.handlelength": 1.4,
    "figure.dpi": 200, "savefig.dpi": 300,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

FULL = 7.2   # double-column width, inches
HALF = 3.5


def letter(ax, s, dx=-0.02, dy=0.02, title=None):
    """Bold lowercase panel letter at the top-left of an axes, in figure space,
    optionally followed by a short sentence-case title."""
    fig = ax.figure
    bb = ax.get_position()
    x, y = bb.x0 + dx, bb.y1 + dy
    fig.text(x, y, s, fontsize=10, fontweight="bold", va="bottom", ha="left", color=INK)
    if title:
        fig.text(x + 0.022, y, title, fontsize=8, va="bottom", ha="left", color=INK)


def letter_at(fig, x, y, s, title=None):
    fig.text(x, y, s, fontsize=10, fontweight="bold", va="bottom", ha="left", color=INK)
    if title:
        fig.text(x + 0.022, y, title, fontsize=8, va="bottom", ha="left", color=INK)


def refline(ax, y=None, x=None, color=RULE, lw=0.8, ls=(0, (3, 3)), zorder=1):
    if y is not None:
        ax.axhline(y, color=color, lw=lw, ls=ls, zorder=zorder)
    if x is not None:
        ax.axvline(x, color=color, lw=lw, ls=ls, zorder=zorder)


def paired(ax, left, right, colour_left, colour_right, x=(0, 1), line=UNTRAINED,
           ms=4.5, label_fmt=None, side_labels=True):
    """Per-subject pairs (thin grey lines) with column means as thick bars."""
    left, right = np.asarray(left, float), np.asarray(right, float)
    for a, b in zip(left, right):
        ax.plot(x, [a, b], "-", color=line, lw=0.7, alpha=0.35, zorder=2)
    ax.plot([x[0]] * len(left), left, "o", color=colour_left, ms=ms, mec="white",
            mew=0.8, zorder=4)
    ax.plot([x[1]] * len(right), right, "o", color=colour_right, ms=ms, mec="white",
            mew=0.8, zorder=4)
    for xi, v, c, ha, dx in [(x[0], left, colour_left, "right", -7),
                             (x[1], right, colour_right, "left", 7)]:
        ax.plot([xi - 0.16, xi + 0.16], [v.mean()] * 2, "-", color=c, lw=2.2, zorder=5)
        if side_labels and label_fmt:
            ax.annotate(label_fmt.format(v.mean()), (xi, v.mean()), xytext=(dx, 0),
                        textcoords="offset points", color=c, fontsize=7.2,
                        fontweight="bold", va="center", ha=ha)


def seeds(ax, x, vals, colour, ms=3.2, jitter=0.0, mean_bar=True, bar_w=0.16, alpha=0.9):
    vals = np.asarray(vals, float)
    xs = x + (np.linspace(-jitter, jitter, len(vals)) if jitter else np.zeros(len(vals)))
    ax.plot(xs, vals, "o", color=colour, ms=ms, mec="white", mew=0.6, alpha=alpha, zorder=4)
    if mean_bar:
        ax.plot([x - bar_w, x + bar_w], [vals.mean()] * 2, "-", color=colour, lw=2.0, zorder=5)


def note(ax, s, x=0.02, y=0.97, ha="left", va="top", size=6.6, color=MUTED, style="normal"):
    ax.text(x, y, s, transform=ax.transAxes, ha=ha, va=va, fontsize=size,
            color=color, style=style)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/{name}.{ext}", bbox_inches="tight", facecolor="white",
                    pad_inches=0.04)
    print(f"saved {OUT}/{name}.{{pdf,png}}")
