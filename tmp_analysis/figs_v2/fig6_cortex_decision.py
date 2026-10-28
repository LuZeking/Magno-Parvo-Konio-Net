"""Figure 6 — the cortex-to-decision test (Results 2.5; registered in
docs/PLAN_CORTEX_DECISION_2026-10-26.md; reading of record
CORTEX_NOT_TEXTURE_BIASED_ON_THESE_IMAGES).

  (a) cortical shape-bias rate B per region: model, control, channel swap
      (seeds averaged per subject; points subjects, bars means), the untrained
      network and the input-level (oriented-energy) floor; reference lines at
      0.5, the model's behavioural rate (0.682) and the observers (0.96).
  (b) correspondence with the observers per region, % of each target's
      residual split-half ceiling: shape margin v per-image shape rate (top),
      cue margin v per-image resolution rate (bottom); model (filled) against
      the input floor (open); grey band = ±97.5th percentile of the within-cell
      permutation null of the model's subject-mean r (shape target from
      results/cortex_decision/ctd_h_null_fix.json — the record's shape-target
      null is vacuous, ledger §3); n = subjects in which the model exceeds the
      floor.
  (c) the same voxels under the magno-only, parvo-only and joint estimators of
      the same fit: B per band and region in the model (P-C3).
  (d) P-C4: extrastriate (hV4, ventral, lateral mean) correspondence per seed,
      model v control v channel swap, both targets; floor and untrained dashed.
  (e) hV4: the subject-mean residualised shape margin of the model against the
      observers' residualised shape rate, one point per image (1,194 with a
      resolved trial), with the exemplar-cluster bootstrap interval of r.

Sources: results/cortex_decision/ctd_analysis.json (the record, analysis job
26719989, code 376fb05), ctd_h_null_fix.json (tmp_analysis/ctd_h_null_fix.py),
ctd_perimage_hV4.json (tmp_analysis/ctd_perimage_margins.py).
Usage: PYTHONPATH=.:tmp_analysis/figs_v2 python tmp_analysis/figs_v2/fig6_cortex_decision.py
"""
from __future__ import annotations

import json

import numpy as np
import matplotlib.pyplot as plt

import style as S

REC = "results/cortex_decision/ctd_analysis.json"
HFIX = "results/cortex_decision/ctd_h_null_fix.json"
PERIMAGE = "results/cortex_decision/ctd_perimage_hV4.json"
ROIS = ["V1", "V2", "V3", "hV4", "ventral", "lateral", "parietal"]
EXTRA = ["hV4", "ventral", "lateral"]
ARMS = [("model", S.MODEL, "o", "model (3 seeds)"), ("control", S.CONTROL, "o", "control (3 seeds)"),
        ("swap", S.SWAP, "o", "channel swap (3 seeds)"), ("untrained", S.UNTRAINED, "s", "untrained"),
        ("gabor", S.SINGLE, "^", "input floor (oriented energy)")]
FLOOR = "#8C8C8C"


def per_subject(d, arm, roi, band, key):
    ps = d["per_subject"][arm]
    return np.array([ps[s][roi][band][key] for s in sorted(ps, key=int) if roi in ps[s]], float)


def main():
    d = json.load(open(REC))
    hf = json.load(open(HFIX))
    pim = json.load(open(PERIMAGE))
    P = d["predictions"]
    ceil_h, ceil_v = d["settings"]["ceilings"]["ceiling_h"], d["settings"]["ceilings"]["ceiling_v"]
    behav = d["settings"]["behavioural_rate_model"]
    print(f"reading of record: {d['reading_of_record']}; " + ", ".join(f"{k}={'MET' if v.get('met') else 'NOT MET'}" for k, v in P.items() if "met" in v))

    fig = plt.figure(figsize=(S.FULL, 6.6))
    gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.50, left=0.07, right=0.985, top=0.93, bottom=0.08,
                          width_ratios=[1.45, 1.0, 1.0], height_ratios=[1.0, 1.0])
    x = np.arange(len(ROIS))

    # ------------------------------------------------------------ (a) rates
    ax = fig.add_subplot(gs[0, 0])
    S.refline(ax, y=0.5, color=S.INK, lw=0.7)
    S.refline(ax, y=behav, color=S.MODEL)
    S.refline(ax, y=0.959, color=S.HUMAN)
    ax.text(6.45, behav + 0.012, f"model's readout {behav:.2f}", fontsize=6.0, color=S.MODEL, ha="right", va="bottom")
    ax.text(6.45, 0.959 - 0.012, "observers 0.96", fontsize=6.0, color=S.HUMAN, ha="right", va="top")
    offs = {"model": -0.26, "control": -0.13, "swap": 0.0, "untrained": 0.13, "gabor": 0.26}
    for arm, col, mk, lab in ARMS:
        mus = []
        for i, roi in enumerate(ROIS):
            v = per_subject(d, arm, roi, "joint", "B")
            mus.append(v.mean())
            open_ = arm in ("gabor", "untrained")
            ax.plot(np.full(len(v), i + offs[arm]), v, mk, color=col, ms=2.4, mec=col if open_ else "white",
                    mfc="white" if open_ else col, mew=0.5, alpha=0.85, zorder=3)
            ax.plot([i + offs[arm] - 0.06, i + offs[arm] + 0.06], [v.mean()] * 2, "-", color=col, lw=2.0, zorder=4)
        ax.plot(x + offs[arm], mus, "-", color=col, lw=0.7, alpha=0.5, zorder=2, label=lab)
        print(f"  (a) {arm:9s} B joint " + " ".join(f"{m:.3f}" for m in mus))
    ax.set_xticks(x); ax.set_xticklabels(ROIS, fontsize=6.4, rotation=25, ha="right")
    ax.set_xlim(-0.6, 6.6); ax.set_ylim(0.30, 1.0)
    ax.set_ylabel("cortical shape-bias rate B\nP(shape template closer than texture template)")
    ax.legend(fontsize=5.8, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.92), ncol=1, handlelength=1.6)
    S.letter(ax, "a", dx=-0.05, dy=0.03, title="Predicted cortex is not texture-biased on these images")

    # ------------------------------------------------------------ (b) correspondence, h and v
    sub = gs[0, 1:].subgridspec(2, 1, hspace=0.12)
    for row, (target, key, pkey, ceil, lab) in enumerate([("h", "pct_h", "P-C2", ceil_h, "shape margin v shape rate"),
                                                          ("v", "pct_v", "P-C2", ceil_v, "cue margin v resolution rate")]):
        ax = fig.add_subplot(sub[row, 0])
        S.refline(ax, y=0, color=S.INK, lw=0.7)
        null = np.array([100 * hf["per_arm_roi"]["model"][r]["null_h_97.5"] / ceil_h if target == "h"
                         else 100 * d["permutation"]["model"][r]["null_v_97.5"] / ceil_v for r in ROIS])
        ax.fill_between(x, -null, null, step=None, color="#E6E6E6", lw=0, zorder=1)
        for i, roi in enumerate(ROIS):
            m = per_subject(d, "model", roi, "joint", key)
            g = per_subject(d, "gabor", roi, "joint", key)
            ax.plot(np.full(len(m), i - 0.14), m, "o", color=S.MODEL, ms=2.6, mec="white", mew=0.4, zorder=4)
            ax.plot([i - 0.22, i - 0.06], [m.mean()] * 2, "-", color=S.MODEL, lw=2.0, zorder=5)
            ax.plot(np.full(len(g), i + 0.14), g, "^", mfc="white", mec=FLOOR, ms=2.8, mew=0.7, zorder=4)
            ax.plot([i + 0.06, i + 0.22], [g.mean()] * 2, "-", color=FLOOR, lw=2.0, zorder=5)
            n_above = int((m > g).sum())
            ax.text(i, -4.5, f"{n_above}/8", fontsize=5.6, color=S.MUTED, ha="center", va="top")
            p = hf["per_arm_roi"]["model"][roi]["p_h"] if target == "h" else d["permutation"]["model"][roi]["p_v"]
            print(f"  (b) {target} {roi:8s} model {m.mean():+5.1f} [{m.min():+5.1f},{m.max():+5.1f}] floor {g.mean():+5.1f} "
                  f"null97.5 {null[i]:4.1f} p={p:.4f} n(model>floor)={n_above}")
        ax.set_xticks(x)
        ax.set_xticklabels(ROIS if row == 1 else [""] * 7, fontsize=6.6)
        ax.set_xlim(-0.6, 6.6)
        ax.set_ylim(-8, 25 if row == 0 else 20)
        ax.set_yticks([0, 10, 20])
        ax.set_ylabel(f"{lab}\n% of ceiling ({ceil:.2f})", fontsize=6.4)
        if row == 0:
            ax.plot([], [], "o", color=S.MODEL, ms=3, label="model (per subject)")
            ax.plot([], [], "^", mfc="white", mec=FLOOR, ms=3, label="input floor")
            ax.fill_between([], [], [], color="#E6E6E6", label="permutation null, 97.5th pct")
            ax.legend(fontsize=5.6, frameon=False, loc="upper right", ncol=3, handlelength=1.2, columnspacing=0.7)
            S.letter(ax, "b", dx=-0.09, dy=0.05, title="Evidence present, but matched by the input floor beyond hV4")

    # ------------------------------------------------------------ (c) bands
    ax = fig.add_subplot(gs[1, 0])
    S.refline(ax, y=0.5, color=S.INK, lw=0.7)
    for band, col, lab, off in [("magno", S.MAGNO, "luminance band (magno)", -0.18), ("joint", S.INK, "joint fit", 0.0),
                                ("parvo", S.PARVO, "L−M band (parvo)", 0.18)]:
        mus = []
        for i, roi in enumerate(ROIS):
            v = per_subject(d, "model", roi, band, "B")
            mus.append(v.mean())
            ax.plot(np.full(len(v), i + off), v, "o", color=col, ms=2.4, mec="white", mew=0.4, alpha=0.85, zorder=3)
        ax.plot(x + off, mus, "-o", color=col, lw=1.4, ms=0, zorder=4, label=lab)
        print(f"  (c) {band:6s} B " + " ".join(f"{m:.3f}" for m in mus))
    S.note(ax, "magno > parvo: 8/8 subjects, 24/24 seed-cells,\nevery region; joint between them in 8/8", x=0.02, y=0.97)
    ax.set_xticks(x); ax.set_xticklabels(ROIS, fontsize=6.4, rotation=25, ha="right")
    ax.set_xlim(-0.6, 6.6); ax.set_ylim(0.30, 0.85)
    ax.set_ylabel("cortical shape-bias rate B, model\nsame voxels, one band of the fit")
    ax.legend(fontsize=5.8, frameon=False, loc="lower left", handlelength=1.6)
    S.letter(ax, "c", dx=-0.05, dy=0.03, title="Opposite evidence in the same voxels")

    # ------------------------------------------------------------ (d) P-C4 seeds
    ax = fig.add_subplot(gs[1, 1])
    c4 = P["P-C4"]
    xs = {"h": 0.0, "v": 1.5}
    for t, x0 in xs.items():
        for j, (arm, col) in enumerate([("model", S.MODEL), ("control", S.CONTROL), ("swap", S.SWAP)]):
            vals = np.array(list(c4[t][f"{arm}_seed_means"].values()))
            S.seeds(ax, x0 + (j - 1) * 0.36, vals, col, ms=3.4, bar_w=0.13)
            print(f"  (d) {t} {arm:8s} extrastriate seeds " + " ".join(f"{v:5.1f}" for v in vals))
        for arm, ls, col in [("gabor", (0, (3, 2)), FLOOR), ("untrained", (0, (1, 1.5)), S.UNTRAINED)]:
            key = "pct_h_mean" if t == "h" else "pct_v_mean"
            fl = np.mean([P["P-C5"][arm][r][key] for r in EXTRA])
            ax.plot([x0 - 0.55, x0 + 0.55], [fl] * 2, ls=ls, color=col, lw=0.9, zorder=2)
            print(f"  (d) {t} {arm:8s} extrastriate mean {fl:5.1f}")
        mw = c4[t]["mw_vs_control"]["p_two"], c4[t]["mw_vs_swap"]["p_two"]
        ax.text(x0, 0.5, f"MW p\n{mw[0]:.2f} / {mw[1]:.2f}", fontsize=5.4, color=S.MUTED, ha="center", va="bottom")
    S.refline(ax, y=0, color=S.INK, lw=0.7)
    ax.set_xticks([0, 1.5]); ax.set_xticklabels(["shape\ntarget", "resolution\ntarget"], fontsize=6.6)
    ax.set_xlim(-0.8, 2.3); ax.set_ylim(0, 21)
    ax.set_ylabel("extrastriate correspondence per seed\n% of ceiling (hV4, ventral, lateral mean)")
    ax.plot([], [], "o", color=S.MODEL, ms=3, label="model"); ax.plot([], [], "o", color=S.CONTROL, ms=3, label="control")
    ax.plot([], [], "o", color=S.SWAP, ms=3, label="swap"); ax.plot([], [], ls=(0, (3, 2)), color=FLOOR, label="floor")
    ax.plot([], [], ls=(0, (1, 1.5)), color=S.UNTRAINED, label="untrained")
    ax.legend(fontsize=5.6, frameon=False, loc="upper left", ncol=2, handlelength=1.4, columnspacing=0.6)
    S.letter(ax, "d", dx=-0.09, dy=0.03, title="The trajectory adds nothing")

    # ------------------------------------------------------------ (e) hV4 scatter
    ax = fig.add_subplot(gs[1, 2])
    arm = pim["arms"]["model"]
    mx = np.asarray(arm["subject_mean_m_resid"], float)
    hy = np.asarray(pim["h_resid"], float)
    keep = np.isfinite(mx) & np.isfinite(hy)
    ax.plot(mx[keep], hy[keep], "o", color=S.MODEL, ms=1.6, mec="none", alpha=0.35, zorder=2)
    b, a = np.polyfit(mx[keep], hy[keep], 1)
    xx = np.linspace(mx[keep].min(), mx[keep].max(), 2)
    ax.plot(xx, a + b * xx, "-", color=S.INK, lw=1.2, zorder=3)
    r = arm["r_h_resid_subject_mean"]; lo, hi = arm["r_h_resid_subject_mean_ci_exemplar"]
    rs = per_subject(d, "model", "hV4", "joint", "r_h_resid")
    S.note(ax, f"r = {r:.3f} [{lo:.3f}, {hi:.3f}] exemplar bootstrap\n{100 * r / ceil_h:.0f}% of ceiling; "
               f"per subject {100 * rs.min() / ceil_h:.0f}–{100 * rs.max() / ceil_h:.0f}%\nn = {int(keep.sum())} images", x=0.03, y=0.97)
    ax.set_ylim(-0.95, 0.48)
    print(f"  (e) hV4 subject-mean margin r = {r:+.4f} CI [{lo:+.4f}, {hi:+.4f}] ({100 * r / ceil_h:+.1f}% of ceiling); "
          f"per-subject r {rs.min():.3f}–{rs.max():.3f}; n={int(keep.sum())}")
    ax.set_xlabel("predicted hV4 shape margin, residualised\n(subject mean; Fisher z)", fontsize=6.6)
    ax.set_ylabel("observers' shape rate, residualised", fontsize=6.6)
    S.letter(ax, "e", dx=-0.12, dy=0.03, title="hV4, image by image")

    S.save(fig, "fig6_cortex_decision")


if __name__ == "__main__":
    main()
