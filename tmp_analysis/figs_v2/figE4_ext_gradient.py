"""Extended Data Figure 4 — the gradient beyond hV4: ventral, lateral and dorsal.

  (a) stream-preference index (median over reliable voxels of (Um − Up)/(Um + Up))
      on the 22 Kastner-2015 ROIs that survive the ≥ 20-voxel rule, grouped as
      early (V1–hV4), ventral (VO1/2, PHC1/2), lateral (LO1/2, TO1/2) and dorsal
      (V3A/B, IPS0–2). Untrained network on the diet front end, trained no-diet
      and diet arms (3 seeds each, seed-mean per subject; thin lines = subjects).
  (b) the two components: noise-ceiling-normalised unique variance of the magno
      band (Um, dashed) and the parvo band (Up, solid), same ROIs, both arms.
  (c) regional contrasts per subject (each region's mean index minus the V1 mean;
      dorsal − ventral; lateral − ventral; LO − hV4), both arms.

Source: results/figures_v2/ext_gradient.json (tmp_analysis/figs_v2/ext_gradient.py).
Usage: PYTHONPATH=.:tmp_analysis/figs_v2 python tmp_analysis/figs_v2/figE4_ext_gradient.py
"""
from __future__ import annotations

import json

import numpy as np
import matplotlib.pyplot as plt

import style as S

D = json.load(open("results/figures_v2/ext_gradient.json"))
GROUPS = [("early", ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4"]),
          ("ventral", ["VO1", "VO2", "PHC1", "PHC2"]),
          ("lateral", ["LO1", "LO2", "TO1", "TO2"]),
          ("dorsal", ["V3A", "V3B", "IPS0", "IPS1", "IPS2"])]
ROIS = [r for _, rs in GROUPS for r in rs]
ARMS = [("untrained", S.UNTRAINED), ("no diet", S.NODIET), ("diet", S.DIET)]
GAP = 0.8


def xpos():
    x, out = 0.0, {}
    for g, rs in GROUPS:
        for r in rs:
            out[r] = x
            x += 1
        x += GAP
    return out


X = xpos()


def group_labels(ax, y):
    for g, rs in GROUPS:
        xs = [X[r] for r in rs]
        ax.plot([min(xs) - 0.3, max(xs) + 0.3], [y, y], "-", color=S.RULE, lw=0.8, clip_on=False)
        ax.text(np.mean(xs), y - 0.02, g, ha="center", va="top", fontsize=6.8, color=S.MUTED, clip_on=False)


def main():
    fig = plt.figure(figsize=(S.FULL, 5.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1.55, 1], hspace=0.62, wspace=0.28,
                          left=0.07, right=0.99, top=0.94, bottom=0.10)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    # (a) index profile
    K = D["Kastner2015"]
    for arm, c in ARMS:
        per = np.array([K[arm][r]["per_subject"] for r in ROIS])  # (roi, subj)
        xs = np.array([X[r] for r in ROIS])
        for j in range(per.shape[1]):
            ax_a.plot(xs, per[:, j], "-", color=c, lw=0.6, alpha=0.22, zorder=2)
        ax_a.plot(xs, np.nanmean(per, 1), "o-", color=c, lw=1.8, ms=3.6, mec="white", mew=0.6, zorder=4,
                  label={"untrained": "untrained (diet front end)", "no diet": "trained, no diet", "diet": "trained, diet"}[arm])
    S.refline(ax_a, y=0)
    ax_a.set_xticks([X[r] for r in ROIS]); ax_a.set_xticklabels(ROIS, rotation=90, fontsize=6.3)
    ax_a.set_ylim(-1.08, 1.08); ax_a.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax_a.set_ylabel("stream preference\n(Um − Up)/(Um + Up)")
    ax_a.text(-0.6, 0.93, "magno band", fontsize=6.6, color=S.MUTED, va="center")
    ax_a.text(-0.6, -0.93, "parvo band", fontsize=6.6, color=S.MUTED, va="center")
    ax_a.legend(loc="center", bbox_to_anchor=(0.55, 0.80), ncol=3, handletextpad=0.5, columnspacing=1.2)
    group_labels(ax_a, -1.55)
    S.letter(ax_a, "a", title="beyond hV4 the index saturates in every direction: no dorsal or lateral reversal")

    # (b) components
    C = D["components"]
    for arm, c in [("no diet", S.NODIET), ("diet", S.DIET)]:
        um = np.array([C[arm].get(r, [np.nan] * 3)[0] for r in ROIS])
        up = np.array([C[arm].get(r, [np.nan] * 3)[1] for r in ROIS])
        xs = np.array([X[r] for r in ROIS])
        ax_b.plot(xs, up, "o-", color=c, lw=1.6, ms=3.2, mec="white", mew=0.5, zorder=4)
        ax_b.plot(xs, um, "s--", color=c, lw=1.2, ms=3.0, mec="white", mew=0.5, zorder=3, alpha=0.9)
    ax_b.plot([], [], "o-", color=S.INK, lw=1.4, ms=3.2, label="Up (parvo band)")
    ax_b.plot([], [], "s--", color=S.INK, lw=1.2, ms=3.0, label="Um (magno band)")
    ax_b.legend(loc="upper left", ncol=1)
    ax_b.set_xticks([X[r] for r in ROIS]); ax_b.set_xticklabels(ROIS, rotation=90, fontsize=6.3)
    ax_b.set_ylabel("unique variance / NC\n(median over voxels)")
    ax_b.set_ylim(0, 0.30)
    group_labels(ax_b, -0.065)
    S.note(ax_b, "diet: Up rises, Um collapses", x=0.98, y=0.30, ha="right", color=S.DIET)
    S.note(ax_b, "no diet: Um rises, Up stays ≈ 0", x=0.98, y=0.21, ha="right", color=S.NODIET)
    S.letter(ax_b, "b", title="both components move: Up rises, Um collapses")

    # (c) regional contrasts
    con = D["contrasts"]
    keys = [("ventral(VO,PHC) - early", "ventral − V1"), ("lateral(LO,TO) - early", "lateral − V1"),
            ("dorsal(V3AB,IPS0-3) - early", "dorsal − V1"), ("dorsal - ventral", "dorsal − ventral"),
            ("lateral - ventral", "lateral − ventral"), ("LO - hV4", "LO − hV4")]
    for i, (k, lab) in enumerate(keys):
        for arm, c, dx in [("no diet", S.NODIET, -0.17), ("diet", S.DIET, 0.17)]:
            v = np.array(con[k][arm]["per_subject"], float)
            S.seeds(ax_c, i + dx, v, c, ms=3.0, jitter=0.06, bar_w=0.14)
            p = con[k][arm]["p_exact"]
            if p is not None and p < 0.05 and abs(v.mean()) > 0.1:
                ax_c.text(i + dx, v.mean() + (0.06 if v.mean() > 0 else -0.06), "✱", ha="center",
                          va="bottom" if v.mean() > 0 else "top", fontsize=7, color=c)
    S.refline(ax_c, y=0)
    ax_c.axvline(2.5, color=S.RULE, lw=0.7)
    ax_c.set_xticks(range(len(keys))); ax_c.set_xticklabels([l for _, l in keys], rotation=35, ha="right", fontsize=6.6)
    ax_c.set_ylabel("Δ stream preference")
    ax_c.set_ylim(-1.1, 0.35)
    S.note(ax_c, "✱ exact Wilcoxon p < .05, n = 8", x=0.98, y=0.06, ha="right", va="bottom")
    ax_c.plot([], [], "o", color=S.NODIET, ms=3, label="no diet"); ax_c.plot([], [], "o", color=S.DIET, ms=3, label="diet")
    ax_c.legend(loc="upper right", ncol=2)
    S.letter(ax_c, "c", title="region − V1 falls; region − region does not")

    S.save(fig, "figE4_ext_gradient")


if __name__ == "__main__":
    main()
