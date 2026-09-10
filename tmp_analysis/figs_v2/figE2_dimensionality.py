"""Extended Data Figure 2 — effective dimensionality: the diet fixes which side of
parity training ends on, not a level.

Magno/parvo ratio of the participation ratio of each stream's features on 1,000
NSD images per subject (n = 8). Left: each arm's front end before training
(untrained network), right: every training seed (final checkpoints), fanned
from the front end it started at. The claim is the side of parity; no level is
labelled because the within-arm seed range exceeds the between-arm gaps.

Source: results/figures/fig1_data.npz (panel c arrays).
Usage: PYTHONPATH=.:tmp_analysis/figs_v2 python tmp_analysis/figs_v2/figE2_dimensionality.py
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import style as S

D = np.load("results/figures/fig1_data.npz", allow_pickle=True)
ARMS = [("nodiet", S.NODIET, "no diet"), ("diet_highsf", S.HIGHSF, "diet on luminance, parvo_highsf front end"),
        ("diet_normal", S.DIET, "diet on luminance"), ("diet_swap", S.SWAP, "diet on L−M (swap)")]


def main():
    fig, ax = plt.subplots(figsize=(S.HALF + 0.6, 3.0))
    fig.subplots_adjust(left=0.16, right=0.62, top=0.88, bottom=0.14)
    ax.axhspan(1.0, 5.6, color=S.DIET, alpha=0.05, lw=0, zorder=0)
    ax.axhline(1.0, color="#7A7A7A", lw=1.1, ls=(0, (4, 3)), zorder=2)
    ax.text(-0.42, 1.06, "magno-dominant", fontsize=6, color=S.MUTED, va="bottom", style="italic")
    ax.text(-0.42, 0.94, "parity", fontsize=6, color=S.MUTED, va="top")
    labels = []
    for key, c, lab in ARMS:
        f = D[f"c__{key}__front"]; fm = f.mean(); sm = D[f"c__{key}__seedmeans"]
        ax.plot([0, 0], [f.min(), f.max()], "-", color=c, lw=1.0, alpha=0.5, zorder=3)
        ax.plot([0], [fm], "o", color=c, ms=4.8, mec="white", mew=0.7, zorder=5)
        for v in sm:
            ax.plot([0, 1], [fm, v], "-", color=c, lw=0.9, alpha=0.6, zorder=3)
            ax.plot([1], [v], "o", color=c, ms=4.0, mec="white", mew=0.6, zorder=5)
        n_ok = sum(int(D[f"c__{key}__above{s}"][0]) for s in range(3)); n_tot = sum(int(D[f"c__{key}__above{s}"][1]) for s in range(3))
        if key == "nodiet":
            n_ok = n_tot - n_ok
        labels.append([sm.mean(), f"{lab}\n{n_ok}/{n_tot} subject-seeds {'at or below' if key == 'nodiet' else 'above'} parity", c])
        print(f"  {key:12s} front {fm:.3f}  seeds {np.round(sm, 3)}  {n_ok}/{n_tot}")
    labels.sort(key=lambda e: e[0])
    for i in range(1, len(labels)):
        if labels[i][0] - labels[i - 1][0] < 0.42:
            labels[i][0] = labels[i - 1][0] + 0.42
    for y, txt, c in labels:
        ax.annotate(txt, (1, y), xytext=(10, 0), textcoords="offset points", fontsize=5.8, color=c, va="center")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["front end\n(untrained)", "after training\n(3 seeds, final ckpt)"])
    ax.set_xlim(-0.45, 1.3); ax.set_ylim(0.4, 5.6)
    ax.set_ylabel("effective dimensionality ratio\n(magno / parvo, participation ratio)")
    fig.text(0.02, 0.95, "The diet fixes the side of parity, not a level", fontsize=8, va="bottom", ha="left", color=S.INK)
    S.save(fig, "figE2_dimensionality")


if __name__ == "__main__":
    main()
