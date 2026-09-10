"""Extended Data Figure 1 — the full out-of-distribution degradation curves
behind Figure 5a: relative accuracy against corruption severity, seven
families, four conditions, 3 seeds each (thin) with seed mean (thick).

Usage: PYTHONPATH=.:tmp_analysis/figs_v2 python tmp_analysis/figs_v2/figE1_ood_curves.py
"""
from __future__ import annotations

import glob
import json

import numpy as np
import matplotlib.pyplot as plt

import style as S

OOD = "results/ood_bestep"
CONDS = [("diet", "dkl_bcs_seed", S.DIET), ("no diet", "dkl_base_seed", S.NODIET),
         ("swap", "dkl_bcs_swap_seed", S.SWAP), ("parvo_highsf", "dkl_bcs_parvo_highsf_seed", S.HIGHSF)]
FAMILIES = ["uniform-noise", "high-pass", "false-colour", "phase-scrambling", "rotation", "contrast", "low-pass"]


def main():
    data = {lab: [json.load(open(f)) for f in sorted(glob.glob(f"{OOD}/{p}*.json"))] for lab, p, _ in CONDS}
    fig, axes = plt.subplots(2, 4, figsize=(S.FULL, 3.6))
    fig.subplots_adjust(left=0.065, right=0.99, top=0.90, bottom=0.14, hspace=0.75, wspace=0.35)
    axes = axes.ravel()
    for i, fam in enumerate(FAMILIES):
        ax = axes[i]
        sev = data["diet"][0]["families"][fam]["severities"]
        for lab, _p, c in CONDS:
            acc = np.array([r["families"][fam]["accuracy"] for r in data[lab]], float)
            rel = acc / acc[:, :1]
            x = np.arange(rel.shape[1])
            for r in rel:
                ax.plot(x, r, "-", color=c, lw=0.5, alpha=0.4)
            ax.plot(x, rel.mean(0), "-o", color=c, lw=1.6, ms=2.4, mew=0, label=S.COND_LABEL[lab])
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
    axes[-1].legend(h, l, loc="center", fontsize=6.6, handlelength=1.6)
    S.save(fig, "figE1_ood_curves")


if __name__ == "__main__":
    main()
