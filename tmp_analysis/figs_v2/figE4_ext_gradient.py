"""Extended Data Figure 4 — the gradient beyond hV4: ventral, lateral and dorsal.

  (a) stream-preference index (median over reliable voxels of (Um − Up)/(Um + Up))
      on the 22 Kastner-2015 ROIs that survive the ≥ 20-voxel rule, grouped as
      early (V1–hV4), ventral (VO1/2, PHC1/2), lateral (LO1/2, TO1/2) and dorsal
      (V3A/B, IPS0–2). The untrained network on the model's test input, the
      control and the model (seed mean per subject; thin lines = subjects); the
      BA network's mean as a thin line.
  (b) the two components: noise-ceiling-normalised unique variance of the magno
      band (Um, dashed) and the parvo band (Up, solid), same ROIs, model and control.
  (c) regional contrasts per subject (each region's mean index minus the V1 mean;
      dorsal − ventral; lateral − ventral; LO − hV4), model and control.

Source: results/figures_v3/ext_gradient.json (tmp_analysis/figs_v2/ext_gradient.py --set v3).
Usage: PYTHONPATH=.:tmp_analysis/figs_v2 python tmp_analysis/figs_v2/figE4_ext_gradient.py
"""
from __future__ import annotations

import json

import numpy as np
import matplotlib.pyplot as plt

import style as S

D = json.load(open("results/figures_v3/ext_gradient.json"))
GROUPS = [("early", ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4"]),
          ("ventral", ["VO1", "VO2", "PHC1", "PHC2"]),
          ("lateral", ["LO1", "LO2", "TO1", "TO2"]),
          ("dorsal", ["V3A", "V3B", "IPS0", "IPS1", "IPS2"])]
ROIS = [r for _, rs in GROUPS for r in rs]
ARMS = [("untrained", S.UNTRAINED), ("control", S.CONTROL), ("model", S.MODEL)]
PAIR = [("control", S.CONTROL), ("model", S.MODEL)]
LABEL = {"untrained": "untrained, model's test input", "control": "control", "model": "model", "BA": "BA network (mean)"}
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
        ax_a.plot(xs, np.nanmean(per, 1), "o-", color=c, lw=1.8, ms=3.6, mec="white", mew=0.6, zorder=4, label=LABEL[arm])
        print(f"  (a) {arm:10s} " + " ".join(f"{r} {np.nanmean(K[arm][r]['per_subject']):+.2f}" for r in ("V1v", "hV4", "VO1", "LO1", "V3A", "IPS2")))
    ba = np.array([np.nanmean(K["BA"][r]["per_subject"]) for r in ROIS])
    ax_a.plot([X[r] for r in ROIS], ba, "-", color=S.BA, lw=1.1, zorder=3, label=LABEL["BA"])
    S.refline(ax_a, y=0)
    ax_a.set_xticks([X[r] for r in ROIS]); ax_a.set_xticklabels(ROIS, rotation=90, fontsize=6.3)
    ax_a.set_ylim(-1.08, 1.08); ax_a.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax_a.set_ylabel("stream preference\n(Um − Up)/(Um + Up)")
    ax_a.text(-0.6, 0.93, "magno band", fontsize=6.6, color=S.MUTED, va="center")
    ax_a.text(-0.6, -0.93, "parvo band", fontsize=6.6, color=S.MUTED, va="center")
    ax_a.legend(loc="center", bbox_to_anchor=(0.55, 0.42), ncol=4, handletextpad=0.5, columnspacing=1.2)
    group_labels(ax_a, -1.55)
    S.letter(ax_a, "a", title="beyond hV4 the model's index stays parvo-explained in every direction: no dorsal or lateral reversal")

    # (b) components
    C = D["components"]
    for arm, c in PAIR:
        um = np.array([C[arm].get(r, [np.nan] * 3)[0] for r in ROIS])
        up = np.array([C[arm].get(r, [np.nan] * 3)[1] for r in ROIS])
        xs = np.array([X[r] for r in ROIS])
        ax_b.plot(xs, up, "o-", color=c, lw=1.6, ms=3.2, mec="white", mew=0.5, zorder=4)
        ax_b.plot(xs, um, "s--", color=c, lw=1.2, ms=3.0, mec="white", mew=0.5, zorder=3, alpha=0.9)
    ax_b.plot([], [], "o-", color=S.INK, lw=1.4, ms=3.2, label="Up (parvo band)")
    ax_b.plot([], [], "s--", color=S.INK, lw=1.2, ms=3.0, label="Um (magno band)")
    for arm, c in PAIR:
        ax_b.plot([], [], "-", color=c, lw=2.0, label=arm)
    ax_b.legend(loc="upper left", ncol=2, columnspacing=1.0)
    ax_b.set_xticks([X[r] for r in ROIS]); ax_b.set_xticklabels(ROIS, rotation=90, fontsize=6.3)
    ax_b.set_ylabel("unique variance / NC\n(median over voxels)")
    ax_b.set_ylim(0, 0.42)
    group_labels(ax_b, -0.10)
    S.letter(ax_b, "b", title="model: Up past V1 and Um ≈ 0; control: Um everywhere")

    # (c) regional contrasts
    con = D["contrasts"]
    keys = [("hV4 - early", "hV4 − V1"), ("ventral(VO,PHC) - early", "ventral − V1"), ("lateral(LO,TO) - early", "lateral − V1"),
            ("dorsal(V3AB,IPS0-3) - early", "dorsal − V1"), ("dorsal - ventral", "dorsal − ventral"),
            ("lateral - ventral", "lateral − ventral"), ("LO - hV4", "LO − hV4")]
    for i, (k, lab) in enumerate(keys):
        for (arm, c), dx in zip(PAIR, (-0.17, 0.17)):
            v = np.array(con[k][arm]["per_subject"], float)
            S.seeds(ax_c, i + dx, v, c, ms=3.0, jitter=0.06, bar_w=0.14)
            p = con[k][arm]["p_exact"]
            print(f"  (c) {lab:18s} {arm:8s} {np.nanmean(v):+.3f}  {int((v < 0).sum())}/8 < 0  p {p}")
            if p is not None and p < 0.05 and abs(v.mean()) > 0.1:
                ax_c.text(i + dx, v.max() + 0.04 if v.mean() > 0 else v.min() - 0.04, "✱", ha="center",
                          va="bottom" if v.mean() > 0 else "top", fontsize=7, color=S.INK)
    S.refline(ax_c, y=0)
    ax_c.axvline(3.5, color=S.RULE, lw=0.7)
    ax_c.set_xticks(range(len(keys))); ax_c.set_xticklabels([l for _, l in keys], rotation=35, ha="right", fontsize=6.6)
    ax_c.set_ylabel("Δ stream preference")
    ax_c.set_ylim(-1.8, 0.45)
    for arm, c in PAIR:
        ax_c.plot([], [], "o", color=c, ms=3, label=arm)
    ax_c.legend(loc="lower right", bbox_to_anchor=(1.0, 0.02), ncol=1)
    S.letter(ax_c, "c", title="region − V1 falls; region − region does not")

    S.save(fig, "figE4_ext_gradient")


if __name__ == "__main__":
    main()
