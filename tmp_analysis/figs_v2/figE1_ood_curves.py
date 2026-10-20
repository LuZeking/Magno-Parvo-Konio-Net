"""Extended Data Figure 1 — the full out-of-distribution degradation curves
behind Figure 5a: relative accuracy against corruption severity, seven
families; the model and its control (final checkpoints) and the BA network's
two arms (best saved checkpoints), each seed thin with the seed mean thick.

Usage: PYTHONPATH=.:tmp_analysis/figs_v2 python tmp_analysis/figs_v2/figE1_ood_curves.py
"""
from __future__ import annotations

import glob
import json

import numpy as np
import matplotlib.pyplot as plt

import style as S
import v3runs as R

CONDS = [("BA network, no diet", sorted(glob.glob("results/ood_bestep/dkl_base_seed?_ep*.json")), S.BANODIET),
         ("BA network", sorted(glob.glob("results/ood_bestep/dkl_bcs_seed?_ep*.json")), S.BA),
         ("control", [f"results/ood/{r}.json" for r in R.runs(R.CONTROL, "results/ood/{}.json")], S.CONTROL),
         ("model", [f"results/ood/{r}.json" for r in R.runs(R.MODEL, "results/ood/{}.json")], S.MODEL)]
FAMILIES = ["uniform-noise", "high-pass", "false-colour", "phase-scrambling", "rotation", "contrast", "low-pass"]


def main():
    data = {lab: [json.load(open(f)) for f in files] for lab, files, _ in CONDS}
    fig, axes = plt.subplots(2, 4, figsize=(S.FULL, 3.6))
    fig.subplots_adjust(left=0.065, right=0.99, top=0.90, bottom=0.14, hspace=0.75, wspace=0.35)
    axes = axes.ravel()
    for i, fam in enumerate(FAMILIES):
        ax = axes[i]
        sev = data["model"][0]["families"][fam]["severities"]
        for lab, _f, c in CONDS:
            acc = np.array([r["families"][fam]["accuracy"] for r in data[lab]], float)
            rel = acc / acc[:, :1]
            x = np.arange(rel.shape[1])
            for r in rel:
                ax.plot(x, r, "-", color=c, lw=0.5, alpha=0.4)
            ax.plot(x, rel.mean(0), "-o", color=c, lw=1.9 if lab == "model" else 1.3, ms=2.4, mew=0,
                    label=f"{lab} ({R.fmt_seeds(len(rel))})")
        ax.axhline(1.0, color="#D0D0D0", lw=0.6, ls=(0, (3, 3)))
        ax.set_xticks(range(len(sev)))
        ax.set_xticklabels(["clean"] + [str(s).rstrip("0").rstrip(".") if "." in str(s) else str(s) for s in sev[1:]],
                           fontsize=5.6, rotation=45 if len(sev) > 4 else 0)
        ax.set_ylim(0, 1.08)
        ax.set_title(fam + ("  (trained in)" if fam in ("contrast", "low-pass") else ""), fontsize=7, loc="left")
        if i % 4 == 0:
            ax.set_ylabel("accuracy /\nclean accuracy")
        S.letter(ax, "abcdefg"[i], dx=-0.035, dy=0.02)
    axes[-1].axis("off")
    h, l = axes[0].get_legend_handles_labels()
    axes[-1].legend(h[::-1], l[::-1], loc="center", fontsize=6.4, handlelength=1.6)
    S.save(fig, "figE1_ood_curves")


if __name__ == "__main__":
    main()
