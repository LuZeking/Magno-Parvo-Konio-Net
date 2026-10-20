"""Extended Data Figure 2 — effective dimensionality: the trajectory decides which
side of parity training ends on, not a level.

Magno/parvo ratio of the participation ratio of each stream's features on 1,000
NSD images per subject (n = 8). Left: the untrained network before training,
right: every training seed (final checkpoints), fanned from the starting point.
The model and its control share one untrained network and one test input; the BA
network's two arms start from their own front ends. The claim is the side of
parity; no level is labelled because the within-arm seed range exceeds the
between-arm gaps.

Sources: results/figures_v2/fs2_pr_ratio.json (tmp_analysis/fs2_pr_ratio.py; model,
control, untrained), results/figures/fig1_data.npz (panel c arrays; BA arms).
Usage: PYTHONPATH=.:tmp_analysis/figs_v2 python tmp_analysis/figs_v2/figE2_dimensionality.py
"""
from __future__ import annotations

import json

import numpy as np
import matplotlib.pyplot as plt

import style as S
import v3runs as R

PR = json.load(open("results/figures_v2/fs2_pr_ratio.json"))["runs"]
BA = np.load("results/figures/fig1_data.npz", allow_pickle=True)


def fs2_arm(tmpl):
    """(untrained per-subject ratios, per-seed list of per-subject ratios) for the seeds with PR entries."""
    ratios = lambda run: np.array([PR[run][s]["ratio"] for s in sorted(PR[run])])
    seeds = [ratios(tmpl.format(i)) for i in range(3) if tmpl.format(i) in PR]
    return ratios("untrained_dvd300_seed0"), seeds


def ba_arm(key):
    return BA[f"c__{key}__front"], [BA[f"c__{key}__trained{s}"] for s in range(3)]


ARMS = [("BA network, no diet", S.BANODIET, ba_arm("nodiet"), -0.08),
        ("BA network", S.BA, ba_arm("diet_normal"), -0.08),
        ("control", S.CONTROL, fs2_arm(R.CONTROL), 0.0),
        ("model", S.MODEL, fs2_arm(R.MODEL), 0.0)]


def main():
    fig, ax = plt.subplots(figsize=(S.HALF + 0.6, 3.0))
    fig.subplots_adjust(left=0.16, right=0.62, top=0.88, bottom=0.14)
    ax.axhspan(1.0, 3.2, color=S.MODEL, alpha=0.05, lw=0, zorder=0)
    ax.axhline(1.0, color="#7A7A7A", lw=1.1, ls=(0, (4, 3)), zorder=2)
    ax.text(-0.42, 1.04, "magno-dominant", fontsize=6, color=S.MUTED, va="bottom", style="italic")
    ax.text(-0.42, 0.96, "parity", fontsize=6, color=S.MUTED, va="top")
    labels = []
    for lab, c, (front, seeds), dx in ARMS:
        fm = front.mean(); sm = np.array([s.mean() for s in seeds])
        ax.plot([dx, dx], [front.min(), front.max()], "-", color=c, lw=1.0, alpha=0.5, zorder=3)
        ax.plot([dx], [fm], "o", color=c, ms=4.8, mec="white", mew=0.7, zorder=5)
        for v in sm:
            ax.plot([dx, 1], [fm, v], "-", color=c, lw=0.9, alpha=0.6, zorder=3)
            ax.plot([1], [v], "o", color=c, ms=4.0, mec="white", mew=0.6, zorder=5)
        above = sum(int((s > 1.0).sum()) for s in seeds); tot = sum(len(s) for s in seeds)
        below = lab in ("control", "BA network, no diet")
        n_ok = tot - above if below else above
        labels.append([sm.mean(), f"{lab}\n{n_ok}/{tot} subject-seeds {'below' if below else 'above'} parity", c])
        print(f"  {lab:22s} untrained {fm:.3f}  seeds {np.round(sm, 3)}  {n_ok}/{tot} {'below' if below else 'above'}")
    labels.sort(key=lambda e: e[0])
    for i in range(1, len(labels)):
        labels[i][0] = max(labels[i][0], labels[i - 1][0] + 0.30)
    for y, txt, c in labels:
        ax.annotate(txt, (1, y), xytext=(10, 0), textcoords="offset points", fontsize=5.8, color=S.INK, va="center")
        ax.plot([1.06], [y], "s", color=c, ms=3, mew=0, clip_on=False)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["untrained", "after training\n(final ckpts)"])
    ax.set_xlim(-0.45, 1.3); ax.set_ylim(0.6, 3.2)
    ax.set_ylabel("effective dimensionality ratio\n(magno / parvo, participation ratio)")
    fig.text(0.02, 0.95, "The trajectory fixes the side of parity, not a level", fontsize=8, va="bottom", ha="left", color=S.INK)
    S.save(fig, "figE2_dimensionality")


if __name__ == "__main__":
    main()
