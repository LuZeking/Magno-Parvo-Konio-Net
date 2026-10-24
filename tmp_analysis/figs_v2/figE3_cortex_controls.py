"""Extended Data Figure 3 — the cortical gradient against its nuisance axes (model of record).

  (a, b) the normalised index of Fig. 3a and the raw statistic (median over voxels of
         U_m − U_p) per ROI, for the untrained network on the shared front end, the control,
         the channel swap and the model (trained arms: 3 seeds, averaged per subject).
  (c)    the gradient hV4 − mean(V1v, V1d) within each of NSD's five eccentricity bands
         (`prf-eccrois`). One point per subject, per band.
  (d)    the same gradient within pRF-size tertiles (per subject, pooled V1 + hV4 voxels,
         pRF R² > 5%).

Labels in (c, d): subjects in which the model's gradient is below the control's, exact Wilcoxon p.
Source: results/figures_v3/cortex_controls.json (tmp_analysis/fs2_cortex_controls.py). The
PAPER_v2 version (the BA network's two arms) is this script at commit 7673c18.
Usage: PYTHONPATH=.:tmp_analysis/figs_v2 python tmp_analysis/figs_v2/figE3_cortex_controls.py
"""
from __future__ import annotations

import json

import numpy as np
import matplotlib.pyplot as plt

import style as S

J = json.load(open("results/figures_v3/cortex_controls.json"))
ROI = J["rois"]
BANDS = ["0–0.5°", "0.5–1°", "1–2°", "2–4°", "> 4°"]
ARMS = [("untrained", "untrained (shared front end)", S.UNTRAINED),
        ("control", "control (3 seeds)", S.CONTROL),
        ("channel swap", "channel swap (3 seeds)", S.SWAP),
        ("model", "model (3 seeds)", S.MODEL)]
STRAT_ARMS = ARMS[1:]


def main():
    fig = plt.figure(figsize=(S.FULL, 5.4))
    gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.32, left=0.08, right=0.985, top=0.92, bottom=0.09)

    # (a, b) profiles on both statistics
    for j, (stat, ylab, ttl) in enumerate([("norm", "normalised index\nmedian of (Um − Up)/(Um + Up)", "Normalised index (as Fig. 3a)"),
                                           ("raw", "raw statistic\nmedian of Um − Up (units of r² / NC)", "Raw statistic")]):
        ax = fig.add_subplot(gs[0, j]); S.refline(ax, y=0)
        for k, lab, c in ARMS:
            prof = np.array(J["per_subject"][k][stat]["profile"], float)
            mu, se = np.nanmean(prof, 0), np.nanstd(prof, 0, ddof=1) / np.sqrt(np.isfinite(prof).sum(0))
            ax.fill_between(range(7), mu - se, mu + se, color=c, alpha=0.18, lw=0)
            ax.plot(range(7), mu, "-o", color=c, lw=1.8, ms=3, mec="white", mew=0.5,
                    ls=(0, (3, 2)) if k == "untrained" else "-", label=lab)
            print(f"  ({'ab'[j]}) {stat:4s} {k:13s} " + " ".join(f"{v:+.3f}" for v in mu))
        ax.set_xticks(range(7)); ax.set_xticklabels(ROI); ax.set_ylabel(ylab)
        if j == 0:
            ax.legend(loc="lower left", fontsize=6.0, handlelength=1.6)
        S.letter(ax, "ab"[j], dx=-0.07, dy=0.03, title=ttl)

    # (c) eccentricity bands, (d) pRF-size tertiles; normalised statistic
    lo = min(np.nanmin(np.array(J["per_subject"][k]["norm"][key], float)) for k, _, _ in STRAT_ARMS
             for key in ("ecc_bands", "size_tertiles"))
    hi = max(np.nanmax(np.array(J["per_subject"][k]["norm"][key], float)) for k, _, _ in STRAT_ARMS
             for key in ("ecc_bands", "size_tertiles"))
    y0 = lo - 0.30
    for j, (key, labels, xlab, ttl) in enumerate([("ecc_bands", BANDS, "pRF eccentricity band (NSD prf-eccrois)", "Gradient within eccentricity bands"),
                                                  ("size_tertiles", ["small", "medium", "large"], "pRF-size tertile (per subject, V1 + hV4 voxels)", "Gradient within pRF-size tertiles")]):
        ax = fig.add_subplot(gs[1, j]); S.refline(ax, y=0)
        for a_i, (k, lab, c) in enumerate(STRAT_ARMS):
            g = np.array(J["per_subject"][k]["norm"][key], float)
            for b in range(g.shape[1]):
                x = b + (a_i - 1) * 0.26
                v = g[:, b]; v = v[np.isfinite(v)]
                if len(v) == 0:
                    continue
                ax.plot(np.full(len(v), x) + np.linspace(-0.045, 0.045, len(v)), v, "o", color=c, ms=2.8, mec="white", mew=0.4, alpha=0.9, zorder=3)
                ax.plot([x - 0.10, x + 0.10], [v.mean()] * 2, "-", color=c, lw=2.0, zorder=4)
                print(f"  ({'cd'[j]}) {k:13s} {labels[b]:8s} n={len(v)} mean {v.mean():+.3f} range {v.min():+.3f}..{v.max():+.3f} neg {(v < 0).sum()}/{len(v)}")
        for b, t in enumerate(J["tests"]["norm"]["model_minus_control"][key]):
            ax.text(b, y0 + 0.02, f"{t['n_neg']}/{t['n']}\np = {t['p']:.3f}", ha="center", va="bottom", fontsize=5.4, color=S.MUTED)
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=6.4)
        ax.set_xlim(-0.6, len(labels) - 0.4); ax.set_ylim(y0, max(hi, 0) + 0.70)
        ax.set_xlabel(xlab); ax.set_ylabel("gradient hV4 − mean(V1v, V1d)\nwithin stratum (normalised)")
        h = [plt.Line2D([], [], color=c, lw=2, label=k) for k, _, c in STRAT_ARMS]
        ax.legend(handles=h, loc="upper left", bbox_to_anchor=(0.0, 0.91), ncol=3, fontsize=6.0, handlelength=1.3,
                  columnspacing=1.0, frameon=False)
        S.note(ax, "labels: subjects with model < control; exact Wilcoxon p", x=0.02, y=0.97, va="top", size=5.6)
        S.letter(ax, "cd"[j], dx=-0.07, dy=0.03, title=ttl)

    S.save(fig, "figE3_cortex_controls")


if __name__ == "__main__":
    main()
