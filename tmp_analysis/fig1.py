"""Figure 1: four instruments converge on the diet contrast.

PAPER_PLAN §46.3, amended by §47.4 and §49.

The figure argues a CONVERGENCE, not a construct. §47.4 showed that shape bias
and effective dimensionality factorise differently over identical inputs, so
the panels are drawn as four separate instruments that agree on one contrast --
diet vs no diet -- and the caption says exactly that. Do not draw them as four
views of one property.

  (a) the input 2x2 (§25). An INTERACTION plot, not bars: the claim is that the
      lines are non-parallel, and non-parallelism is the thing bars hide.
  (b) in-silico cue conflict -> BOLD (§43.1). Paired, because the pairing is
      the inference: same 8 subjects, two models.
  (c) the set point (§49). A slopegraph, because the result IS a convergence
      from divergent starts -- three arms enter from 0.59 / 1.40 / 1.63 and two
      of them leave at the same place. A bar chart of trained values alone
      would delete the entire argument.
  (d) the cost (§43.3). Not optional: three panels favour the diet and this one
      does not. Omitting it would be cherry-picking (§46.3).

Usage: PYTHONPATH=. python tmp_analysis/fig1.py
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sps

D = np.load("results/figures/fig1_data.npz", allow_pickle=True)

# Okabe-Ito. Validated for CVD separation and contrast against the panel
# surface; #56B4E9 sits below 3:1 and therefore carries a direct label, which
# every arm in panel (c) has.
DIET   = "#0072B2"   # blue      -- diet, normal front end
DIETHS = "#56B4E9"   # sky blue  -- diet, inverted (parvo_highsf) front end
NODIET = "#D55E00"   # vermillion
GREY   = "#5A5A5A"
INK    = "#1A1A1A"
MUTED  = "#767676"

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "axes.edgecolor": "#999999", "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "figure.dpi": 200, "font.family": "DejaVu Sans",
    "legend.frameon": False,
})

fig = plt.figure(figsize=(7.2, 6.4))
outer = fig.add_gridspec(2, 2, hspace=0.58, wspace=0.38,
                         left=0.090, right=0.960, top=0.855, bottom=0.070)

# Panel titles live in FIGURE coordinates. Anchoring them to an axis makes a
# two-sub-axis panel (a) collide with its own sub-labels; this cannot.
TAG_X = {"L": 0.020, "R": 0.520}
TAG_Y = {"T": 0.905, "B": 0.400}


def panel_tag(col, row, letter, title):
    x, y = TAG_X[col], TAG_Y[row]
    fig.text(x, y, letter, fontsize=11, fontweight="bold", va="bottom",
             ha="left", color=INK)
    fig.text(x + 0.028, y, title, fontsize=8.5, va="bottom", ha="left",
             color=INK)


def mean_label(ax, x, v, colour, side, fmt="{:+.2f}"):
    """Value labels go OUTSIDE the point column, never on top of it."""
    dx, ha = (-8, "right") if side == "left" else (8, "left")
    ax.annotate(fmt.format(v), (x, v), xytext=(dx, 0),
                textcoords="offset points", color=colour, fontsize=8.5,
                fontweight="bold", va="center", ha=ha)


def paired(ax, left, right, colour, subs_lw=0.85):
    """Per-subject lines plus the mean. The pairing is the inference."""
    for a, b in zip(left, right):
        ax.plot([0, 1], [a, b], "-", color=colour, lw=subs_lw, alpha=0.35,
                solid_capstyle="round", zorder=2)
    ax.plot([0, 1], [left.mean(), right.mean()], "-", color=colour, lw=2.6,
            zorder=4, solid_capstyle="round")
    ax.plot([0, 1], [left.mean(), right.mean()], "o", color=colour, ms=6,
            mec="white", mew=1.1, zorder=5)


# ============================================================== panel (a)
# Two sub-axes, NOT one axis with two y-scales. The measures have different
# units and ranges; a dual axis would be the single worst thing this figure
# could do.
gsa = outer[0, 0].subgridspec(1, 2, wspace=0.55)
CARRIER = [("chromatic", NODIET, "chromatic carrier (L−M)"),
           ("achromatic", DIET, "achromatic carrier (lum)")]
X = [0, 1]

for j, (measure, ylab, neutral, ylim) in enumerate(
        [("beh", "cue-conflict shape bias", 0.5, (0.28, 0.97)),
         ("probe", "probe shape − texture", 0.0, (-0.62, 0.97))]):
    ax = fig.add_subplot(gsa[0, j])
    ax.axhline(neutral, color="#CCCCCC", lw=0.8, ls=(0, (3, 3)), zorder=1)
    for carrier, c, _lab in CARRIER:
        y = [D[f"a__{carrier}__{t}__{measure}"] for t in ("none", "transform")]
        for xi, v in zip(X, y):                       # individual seeds
            ax.plot([xi] * len(v), v, "o", color=c, ms=2.6, alpha=0.55,
                    mec="none", zorder=3)
        m = [v.mean() for v in y]
        ax.plot(X, m, "-", color=c, lw=2.2, zorder=4, solid_capstyle="round")
        ax.plot(X, m, "o", color=c, ms=6, mec="white", mew=1.1, zorder=5)
    # Label only the conjunction and its nearest rival: a number on every point
    # is noise.
    conj = D[f"a__achromatic__transform__{measure}"].mean()
    ax.annotate(f"{conj:+.2f}" if measure == "probe" else f"{conj:.2f}",
                (1, conj), xytext=(0, 8), textcoords="offset points",
                color=DIET, fontsize=8, fontweight="bold", ha="center")
    ax.set_xticks(X)
    ax.set_xticklabels(["no\ntransform", "with\ntransform"])
    ax.set_xlim(-0.28, 1.28)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylab, fontsize=7.5)
    ax.tick_params(length=2.5)
    ax.set_title("behaviour" if j == 0 else "linear probe", fontsize=7.5,
                 color=MUTED, pad=4)

panel_tag("L", "T", "a", "Shape bias needs the conjunction")

handles = [plt.Line2D([], [], color=c, lw=2.2, marker="o", ms=5, mec="white",
                      mew=1.0, label=lab) for _, c, lab in CARRIER]
fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.020, 0.985),
           ncol=2, columnspacing=1.6, handlelength=1.6)

# ============================================================== panel (b)
axB = fig.add_subplot(outer[0, 1])
bd = D["b__diet__magno"] - D["b__diet__parvo"]
bn = D["b__nodiet__magno"] - D["b__nodiet__parvo"]
axB.axhline(0, color="#CCCCCC", lw=0.8, ls=(0, (3, 3)), zorder=1)
for a, b in zip(bn, bd):
    axB.plot([0, 1], [a, b], "-", color=GREY, lw=0.85, alpha=0.40, zorder=2)
axB.plot([0] * len(bn), bn, "o", color=NODIET, ms=5, mec="white", mew=1.0, zorder=4)
axB.plot([1] * len(bd), bd, "o", color=DIET, ms=5, mec="white", mew=1.0, zorder=4)
for x, v, c, side in [(0, bn, NODIET, "left"), (1, bd, DIET, "right")]:
    axB.plot([x - 0.18, x + 0.18], [v.mean()] * 2, "-", color=c, lw=2.4, zorder=5)
    mean_label(axB, x, v.mean(), c, side)
p = sps.wilcoxon(bd - bn).pvalue
axB.set_xticks([0, 1]); axB.set_xticklabels(["no diet", "with diet"])
axB.set_xlim(-0.72, 1.72)
axB.set_ylim(-0.18, 1.78)
axB.set_ylabel("stream dissociation\n(magno − parvo VSB)", fontsize=7.5)
axB.tick_params(length=2.5)
axB.text(0.03, 0.93, f"$p$ = {p:.4f}, {int((bd > bn).sum())}/{len(bd)} subjects",
         transform=axB.transAxes, ha="left", fontsize=7, color=MUTED)
panel_tag("R", "T", "b", "…and it is visible through cortex")

# ============================================================== panel (c)
axC = fig.add_subplot(outer[1, 0])
# dy is the label offset in points: the two diet arms converge to within 0.04
# and would print on top of each other at any shared offset.
ARMS = [("nodiet",      NODIET, "no diet",              0),
        ("diet_normal", DIET,   "diet",                 -8),
        ("diet_highsf", DIETHS, "diet,\ninverted front end", +8)]
axC.axhline(1.0, color="#CCCCCC", lw=0.8, ls=(0, (3, 3)), zorder=1)
axC.text(1.34, 1.0, "parity", fontsize=6.8, color=MUTED, va="center", ha="left")
for key, c, lab, dy in ARMS:
    f, t = D[f"c__{key}__front"], D[f"c__{key}__trained"]
    paired(axC, f, t, c)
    axC.annotate(f"{t.mean():.2f}", (1, t.mean()), xytext=(9, dy),
                 textcoords="offset points", color=c, fontsize=8,
                 fontweight="bold", va="center", ha="left")
    axC.annotate(lab, (0, f.mean()), xytext=(-10, 0), textcoords="offset points",
                 color=c, fontsize=7.2, va="center", ha="right")
axC.set_xticks([0, 1])
axC.set_xticklabels(["front end\n(no training)", "after training"])
axC.set_xlim(-1.30, 1.62)
axC.set_ylim(0.50, 1.92)
axC.set_ylabel("effective dimensionality\nratio (magno / parvo)", fontsize=7.5)
axC.tick_params(length=2.5)
panel_tag("L", "B", "c", "The diet installs a set point")

# ============================================================== panel (d)
axD = fig.add_subplot(outer[1, 1])
dd, dn = D["d__diet__r2nc"], D["d__nodiet__r2nc"]
for a, b in zip(dn, dd):
    axD.plot([0, 1], [a, b], "-", color=GREY, lw=0.85, alpha=0.40, zorder=2)
axD.plot([0] * len(dn), dn, "o", color=NODIET, ms=5, mec="white", mew=1.0, zorder=4)
axD.plot([1] * len(dd), dd, "o", color=DIET, ms=5, mec="white", mew=1.0, zorder=4)
for x, v, c, side in [(0, dn, NODIET, "left"), (1, dd, DIET, "right")]:
    axD.plot([x - 0.18, x + 0.18], [v.mean()] * 2, "-", color=c, lw=2.4, zorder=5)
    mean_label(axD, x, v.mean(), c, side, fmt="{:.3f}")
pd_ = sps.wilcoxon(dd - dn).pvalue
axD.set_xticks([0, 1]); axD.set_xticklabels(["no diet", "with diet"])
axD.set_xlim(-0.72, 1.72)
_lo, _hi = min(dd.min(), dn.min()), max(dd.max(), dn.max())
axD.set_ylim(_lo - 0.03 * (_hi - _lo), _hi + 0.22 * (_hi - _lo))
axD.set_ylabel("median $r^2$ / noise ceiling", fontsize=7.5)
axD.tick_params(length=2.5)
axD.text(0.03, 0.95,
         f"the diet costs {100 * (1 - dd.mean() / dn.mean()):.0f}%  "
         f"($p$ = {pd_:.4f}, {int((dd < dn).sum())}/{len(dd)})",
         transform=axD.transAxes, ha="left", fontsize=7, color=MUTED)
panel_tag("R", "B", "d", "…and this is what it costs")

os.makedirs("results/figures", exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(f"results/figures/fig1_convergence.{ext}", bbox_inches="tight")
print("wrote results/figures/fig1_convergence.{png,pdf}")
print(f"  (b) p={p:.4f}   (d) p={pd_:.4f}")
