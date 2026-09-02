"""Figure 1: the developmental diet decides what learning does to cortical
stream correspondence.

Panel A (lead) is a slopegraph because the result IS a divergence: both
untrained models start at the same place and training moves them in opposite
directions. A bar chart of four cells would hide the pairing; the slopegraph
shows it is the same subject moving, in every subject.

Panel B shows the gradient itself along the ventral hierarchy, so the reader
can see that panel A's summary statistic is not hiding a non-monotonic mess.

Panel C is the representational mechanism (leg a) that panel A's cortical
result mirrors -- magno's shape bias crosses zero without the diet.

Panel D is the control that matters most: after removing each subject's global
stream bias, is the ordering along the hierarchy still there? A general stream
factor was what defeated the thesis's RSA, so it gets a panel, not a footnote.

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

# Okabe-Ito: safe under the common colour-vision deficiencies.
GREY = "#5A5A5A"
NODIET = "#D55E00"     # vermillion -- learning erodes
DIET = "#0072B2"       # blue -- learning builds
SWAP = "#009E73"       # bluish-green -- same diet, wrong channel
VIS_LABELS = ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4"]

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 9,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8,
    "ytick.major.width": 0.8, "figure.dpi": 200,
    "font.family": "DejaVu Sans",
})

fig = plt.figure(figsize=(7.2, 5.4))
gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.15], height_ratios=[1, 1],
                      hspace=0.52, wspace=0.42,
                      left=0.085, right=0.985, top=0.90, bottom=0.085)

# Trained cells average their two independent models per subject, so a line is
# "this subject, this condition", not "this subject, this checkpoint".
nodiet_tr = np.nanmean([D["nodiet_s0__profile"], D["nodiet_s1__profile"]], 0)
diet_tr = np.nanmean([D["diet_thesis__profile"], D["diet_ours_s0__profile"]], 0)
cross = lambda p: p[:, 6] - p[:, 0]

pairs = [
    ("no diet", D["untrained_nodiet__profile"], nodiet_tr, NODIET),
    ("with diet", D["untrained_diet__profile"], diet_tr, DIET),
]

# ---------------------------------------------------------------- panel A
axA = fig.add_subplot(gs[0, 0])
for label, unt, tr, c in pairs:
    a, b = cross(unt), cross(tr)
    for i in range(len(a)):
        axA.plot([0, 1], [a[i], b[i]], "-", color=c, lw=0.9, alpha=0.42,
                 solid_capstyle="round", zorder=2)
    axA.plot([0, 1], [a.mean(), b.mean()], "-", color=c, lw=2.6, zorder=4)
    axA.plot([0, 1], [a.mean(), b.mean()], "o", color=c, ms=5.5,
             mec="white", mew=1.0, zorder=5, label=label)
    axA.annotate(f"{b.mean():+.2f}", (1, b.mean()), xytext=(7, 0),
                 textcoords="offset points", color=c, fontsize=8,
                 fontweight="bold", va="center")

axA.axhline(0, color="#BBBBBB", lw=0.7, zorder=1)
axA.set_xlim(-0.46, 1.42)
axA.set_xticks([0, 1])
axA.set_xticklabels(["untrained", "trained"])
axA.set_ylabel("hV4 − V1v  (stream preference)")
axA.set_title("A   Learning diverges on the diet", loc="left", fontweight="bold")
axA.legend(frameon=False, loc="lower left", handlelength=1.2)
axA.text(0.5, 0.97, "same start, opposite ends", transform=axA.transAxes,
         ha="center", va="top", fontsize=7, color="#666666", style="italic")
axA.text(-0.42, np.mean([cross(D["untrained_nodiet__profile"]).mean(),
                         cross(D["untrained_diet__profile"]).mean()]),
         "−0.48", fontsize=8, color=GREY, va="center", ha="left",
         fontweight="bold")

# ---------------------------------------------------------------- panel B
axB = fig.add_subplot(gs[0, 1])
x = np.arange(7)
# Centred within subject: each profile has its own across-ROI mean removed,
# so the three curves are compared on SHAPE, not on level. The trained models
# sit on opposite sides of the global-bias axis (§17.2) and plotting them raw
# would let that offset masquerade as the gradient.
ctr = lambda p: p - np.nanmean(p, axis=1, keepdims=True)
series = [
    ("untrained", ctr(np.nanmean([D["untrained_nodiet__profile"],
                                  D["untrained_diet__profile"]], 0)),
     GREY, "--"),
    ("trained, no diet", ctr(nodiet_tr), NODIET, "-"),
    ("trained, with diet", ctr(diet_tr), DIET, "-"),
    ("trained, diet on rg", ctr(D["swap_s0__profile"]), SWAP, "--"),
]
for label, p, c, ls in series:
    m = np.nanmean(p, 0)
    se = np.nanstd(p, 0, ddof=1) / np.sqrt(np.isfinite(p).sum(0))
    axB.plot(x, m, ls, color=c, lw=2.0, label=label, zorder=3)
    axB.fill_between(x, m - se, m + se, color=c, alpha=0.16, lw=0, zorder=2)
axB.axhline(0, color="#BBBBBB", lw=0.7, zorder=1)
axB.set_xticks(x)
axB.set_xticklabels(VIS_LABELS)
axB.set_xlabel("retinotopic area  (V1 → hV4)")
axB.set_ylabel("stream preference\n(relative to each subject's mean)")
axB.set_title("B   The gradient, global bias removed", loc="left",
              fontweight="bold")
axB.legend(frameon=False, loc="lower left", handlelength=1.6)
axB.text(0.985, 0.94, "↑ magno", transform=axB.transAxes, ha="right",
         fontsize=7, color="#666666")
axB.text(0.985, 0.04, "↓ parvo", transform=axB.transAxes, ha="right",
         fontsize=7, color="#666666")

# ---------------------------------------------------------------- panel C
# Mirrors panel A deliberately: same slopegraph idiom, same colours, so the
# reader can see that the cortical divergence and the representational one are
# the same shape. Magno's arms diverge; parvo's land on top of each other.
axC = fig.add_subplot(gs[1, 0])
GAP = 2.45
# The swap arm carries the causal claim: identical transform, moved onto the
# chromatic channel. Without it panel C shows only that the diet does
# something, not that it must act on an ACHROMATIC channel to do it.
arms = [("no diet", "nodiet", NODIET), ("with diet", "diet", DIET),
        ("diet on rg", "swap", SWAP)]
for stream, x0 in (("magno", 0.0), ("parvo", GAP)):
    for label, cell, c in arms:
        u = D[f"probe__untrained_{cell}__{stream}"].mean()
        t = D[f"probe__trained_{cell}__{stream}"].mean()
        swap = cell == "swap"
        style = dict(dashes=(3.4, 2.2)) if swap else {}
        axC.plot([x0, x0 + 1], [u, t], "--" if swap else "-", color=c,
                 lw=1.9 if swap else 2.4, zorder=6 if swap else 3, **style)
        axC.plot([x0, x0 + 1], [u, t], "o", color=c, ms=4.2 if swap else 5.0,
                 mec="white", mew=1.0, zorder=6 if swap else 4)
        pts = D[f"probe__trained_{cell}__{stream}"]
        axC.plot([x0 + 1] * len(pts), pts, "o", color="white", ms=2.4,
                 mec=c, mew=0.7, zorder=5)
        if swap and abs(t - D[f"probe__trained_nodiet__{stream}"].mean()) < 0.02:
            # Coincides with the no-diet arm; one number, annotated as shared.
            axC.annotate("swap ≡ no diet", (x0 + 1, t), xytext=(7, -13),
                         fontsize=6.2, textcoords="offset points", color=c,
                         va="center", style="italic")
        else:
            axC.annotate(f"{t:+.2f}", (x0 + 1, t),
                         xytext=(6, -10 if swap else 0), fontsize=7,
                         textcoords="offset points", color=c, va="center",
                         fontweight="bold")

from matplotlib.lines import Line2D
axC.legend(handles=[
    Line2D([], [], color=NODIET, lw=2.4, label="no diet"),
    Line2D([], [], color=DIET, lw=2.4, label="diet on luminance"),
    Line2D([], [], color=SWAP, lw=1.9, ls="--", dashes=(3.4, 2.2),
           label="diet on red-green"),
], loc="upper right", bbox_to_anchor=(1.0, 0.90), fontsize=6.2,
    frameon=False, handlelength=1.9, borderpad=0.1, labelspacing=0.28)
axC.axhline(0, color="#444444", lw=0.9, zorder=2)
axC.axvline(1.62, color="#DDDDDD", lw=0.9, zorder=1)
axC.set_xlim(-0.42, GAP + 1.72)
axC.set_xticks([0, 1, GAP, GAP + 1])
axC.set_xticklabels(["untr.", "trained", "untr.", "trained"])
axC.set_ylabel("shape − texture bias")
axC.set_title("C   Same divergence, in the streams", loc="left",
              fontweight="bold")
axC.set_ylim(-0.78, 1.34)
axC.text(0.5, 1.26, "MAGNO", ha="center", fontsize=7.5, fontweight="bold",
         color="#444444")
axC.text(GAP + 0.5, 1.26, "PARVO", ha="center", fontsize=7.5,
         fontweight="bold", color="#444444")
axC.annotate("shape bias destroyed",
             (1, -0.226), xytext=(-0.38, -0.62),
             fontsize=6.6, color=NODIET, ha="left",
             arrowprops=dict(arrowstyle="->", color=NODIET, lw=0.9,
                             shrinkA=1, shrinkB=3))
axC.text(GAP + 0.45, 0.30, "texture bias either way;\nthe swap only partly\n"
         "relieves it",
         ha="center", fontsize=6.4, color="#777777", style="italic")

# ---------------------------------------------------------------- panel D
axD = fig.add_subplot(gs[1, 1])
rho_sets = [
    ("untrained", np.concatenate([D["untrained_nodiet__rho"],
                                  D["untrained_diet__rho"]]), GREY),
    ("trained,\nno diet", np.concatenate([D["nodiet_s0__rho"],
                                          D["nodiet_s1__rho"]]), NODIET),
    ("trained,\nwith diet", np.concatenate([D["diet_thesis__rho"],
                                            D["diet_ours_s0__rho"]]), DIET),
]
rng = np.random.default_rng(0)
for i, (lab, v, c) in enumerate(rho_sets):
    axD.plot(i + rng.uniform(-0.13, 0.13, len(v)), v, "o", color=c, ms=3.4,
             alpha=0.62, mec="none", zorder=3)
    axD.plot([i - 0.26, i + 0.26], [v.mean()] * 2, "-", color=c, lw=2.4,
             zorder=4, solid_capstyle="round")
    axD.annotate(f"{v.mean():+.2f}", (i, v.mean()), xytext=(0, 11),
                 textcoords="offset points", ha="center", color=c,
                 fontsize=7.5, fontweight="bold", zorder=6,
                 bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none",
                           alpha=0.85))
axD.axhline(-1, color="#BBBBBB", lw=0.7, ls=":", zorder=1)
axD.set_xticks(range(3))
axD.set_xticklabels([r[0] for r in rho_sets])
axD.set_ylabel("ρ (hierarchy)  per subject")
axD.set_ylim(-1.12, 0.28)
axD.set_title("D   Not a global stream bias", loc="left", fontweight="bold")
axD.set_xlim(-0.55, 2.80)
axD.text(0.5, 0.965, "rank-based: unchanged by ANY global stream bias",
         transform=axD.transAxes, ha="center", va="top", fontsize=6.6,
         color="#666666", style="italic")
axD.text(2.78, -1.0, "perfectly\nmonotonic", fontsize=6.2,
         color="#AAAAAA", va="center", ha="right")

fig.suptitle("The developmental visual diet decides whether learning builds "
             "or erodes cortical correspondence",
             fontsize=10, fontweight="bold", x=0.085, ha="left", y=0.975)

os.makedirs("results/figures", exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(f"results/figures/fig1_diet_gradient.{ext}",
                bbox_inches="tight")
print("wrote results/figures/fig1_diet_gradient.{png,pdf}")

# Numbers quoted in the caption, printed so the caption cannot drift from them.
print("\ncaption numbers")
for label, unt, tr, _ in pairs:
    a, b = cross(unt), cross(tr)
    d = b - a
    print(f"  {label:<10} untrained {a.mean():+.4f}  trained {b.mean():+.4f}  "
          f"p={sps.wilcoxon(d)[1]:.4f}  n={len(a)}")
