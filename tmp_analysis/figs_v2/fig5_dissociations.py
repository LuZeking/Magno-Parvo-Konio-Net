"""Figure 5 — three properties the field treats as travelling with shape bias do
not follow from it.

  (a) out-of-distribution robustness: relative-accuracy AUC per corruption
      family, one marker per seed, the model against its control (same test
      input; final checkpoints), the BA network's two arms as small markers (best
      saved checkpoints). Two families are trained in by the diets themselves.
  (b) trial-level agreement with the ten human observers (error consistency,
      Cohen's κ as % of the human–human ceiling) against shape-bias rate, for every
      readout of the w sweep and the three published CNNs.
  (c) the positive control, measured on the BA network (the model's per-image run
      is pending): the same machinery recovers more structure from image
      difficulty (whether either cue resolves) than from the shape–texture
      decision, as % of each target's own split-half ceiling.
  (d) …and the crossed front-end 2×2 that withdraws the BA diet reading of (c).

Sources: results/ood/, results/ood_bestep/, results/sweep_error_consistency/,
/scratch/byrong/mpk/perimage_behavior/results/2026-09-07_geirhos_perimage/perimage_correspondence.json
Usage: PYTHONPATH=.:tmp_analysis/figs_v2 python tmp_analysis/figs_v2/fig5_dissociations.py
"""
from __future__ import annotations

import glob
import json

import numpy as np
import matplotlib.pyplot as plt

import style as S
import v3runs as R

SWEEP = "results/sweep_error_consistency"
PERIMAGE = "/scratch/byrong/mpk/perimage_behavior/results/2026-09-07_geirhos_perimage/perimage_correspondence.json"
PUBLISHED = [("AlexNet", 0.429, 0.032), ("GoogLeNet", 0.312, 0.025), ("VGG-16", 0.172, 0.017)]  # Geirhos et al. 2020
FAMILIES = ["uniform-noise", "high-pass", "false-colour", "rotation", "phase-scrambling", "contrast", "low-pass"]
TRAINED_IN = {"contrast", "low-pass"}


def rel_auc(files):
    runs = [json.load(open(f)) for f in files]
    out = {}
    for fam in FAMILIES:
        acc = np.array([r["families"][fam]["accuracy"] for r in runs], float)
        out[fam] = (acc / acc[:, :1]).mean(1)
    return out


def main():
    fig = plt.figure(figsize=(S.FULL, 2.95))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.25, 1.35, 0.9, 0.85], wspace=0.6, left=0.07, right=0.99, top=0.85, bottom=0.24)

    # ------------------------------------------------------------ (a) OOD
    ax = fig.add_subplot(gs[0, 0])
    A = {"model": rel_auc([f"results/ood/{r}.json" for r in R.runs(R.MODEL, "results/ood/{}.json")]),
         "control": rel_auc([f"results/ood/{r}.json" for r in R.runs(R.CONTROL, "results/ood/{}.json")]),
         "BA": rel_auc(sorted(glob.glob("results/ood_bestep/dkl_bcs_seed?_ep*.json"))),
         "BA no diet": rel_auc(sorted(glob.glob("results/ood_bestep/dkl_base_seed?_ep*.json")))}
    ys = np.arange(len(FAMILIES))[::-1]
    for y, fam in zip(ys, FAMILIES):
        if fam in TRAINED_IN:
            ax.axhspan(y - 0.45, y + 0.45, color="#F2F2F2", lw=0, zorder=0)
        d, n = A["model"][fam], A["control"][fam]
        ax.plot([d.mean(), n.mean()], [y, y], "-", color="#C8C8C8", lw=1.2, zorder=1)
        for k, c in (("BA", S.BA), ("BA no diet", S.BANODIET)):
            ax.plot(A[k][fam], np.full(len(A[k][fam]), y - 0.24), "s", color=c, ms=2.4, mew=0, alpha=0.9, zorder=2)
        ax.plot(n, np.full(len(n), y), "o", color=S.CONTROL, ms=3.8, mec="white", mew=0.5, zorder=3)
        ax.plot(d, np.full(len(d), y), "o", color=S.MODEL, ms=3.8, mec="white", mew=0.5, zorder=3)
        sep = (d.max() < n.min()) or (d.min() > n.max())
        ax.text(1.07, y, "✱" if sep else "≈", fontsize=7 if sep else 8, color=S.INK, va="center", ha="center")
        print(f"  (a) {fam:16s} model {d.mean():.3f} [{d.min():.3f},{d.max():.3f}]  control {n.mean():.3f} (n={len(n)})  "
              f"model − control {d.mean() - n.mean():+.3f}  sep={sep}")
    ax.set_yticks(ys); ax.set_yticklabels([f + ("  (trained in)" if f in TRAINED_IN else "") for f in FAMILIES], fontsize=6.2)
    ax.set_xlim(0.3, 1.1); ax.set_ylim(-0.6, len(FAMILIES) + 0.4)
    ax.set_xlabel("relative-accuracy AUC under corruption\n(higher = more robust)")
    h = [plt.Line2D([], [], marker="o", color=S.MODEL, lw=0, ms=4, label="model"),
         plt.Line2D([], [], marker="o", color=S.CONTROL, lw=0, ms=4, label="control"),
         plt.Line2D([], [], marker="s", color=S.BA, lw=0, ms=3, label="BA network"),
         plt.Line2D([], [], marker="s", color=S.BANODIET, lw=0, ms=3, label="BA, no diet")]
    ax.legend(handles=h, loc="upper left", fontsize=5.6, ncol=2, columnspacing=0.8, handletextpad=0.3, bbox_to_anchor=(-0.02, 1.02))
    S.note(ax, "✱ every model seed beyond the control; ≈ within it", x=0.0, y=-0.30, va="top", size=5.6)
    S.letter(ax, "a", dx=-0.19, dy=0.03, title="Robustness does not follow")

    # ------------------------------------------------------------ (b) error consistency vs rate
    ax = fig.add_subplot(gs[0, 1])
    groups = [("BA network, best ckpts", "dkl_bcs_seed*_best", S.BA, 0.45),
              ("control", "fs2_dvdfix300_avg_w050_ls_seed*_final", S.CONTROL, 0.85),
              ("model", "fs2_dvd300_avg_w050_ls_seed*_final", S.MODEL, 0.85)]
    ceil, hsb, best, n_model = None, None, -np.inf, 0
    for lab, pat, c, alpha in groups:
        runs = [json.load(open(f)) for f in sorted(glob.glob(f"{SWEEP}/{pat}_kappa_sweep.json"))]
        ceil = runs[0]["human_human_ceiling"]; hsb = runs[0]["human_shape_bias"]
        for r in runs:
            sb = np.array([p["shape_bias"] for p in r["sweep"]]); pc = np.array([p["kappa_pct_ceiling"] for p in r["sweep"]])
            ax.plot(sb, pc, "-", color=c, lw=0.6, alpha=0.35, zorder=2)
            ax.plot(sb, pc, "o", color=c, ms=2.4, mew=0, alpha=alpha, zorder=3)
            if lab == "model":
                best = max(best, pc.max()); n_model += len(pc)
        ax.plot([], [], "o", color=c, ms=3.2, mew=0, label=f"{lab} ({R.fmt_seeds(len(runs))})")
    for name, sb_p, k_p in PUBLISHED:
        ax.plot(sb_p, 100 * k_p / ceil, "D", color=S.PUB, ms=3.4, mew=0, zorder=4)
        ax.annotate(name, (sb_p, 100 * k_p / ceil), xytext=(0, -6), textcoords="offset points",
                    ha="center", va="top", fontsize=5.2, color=S.PUB, rotation=90)
    ax.axhline(100, color=S.HUMAN, lw=0.9, zorder=4); ax.text(0.99, 101, f"human–human ceiling (κ = {ceil:.3f})", fontsize=5.8, va="bottom", ha="right")
    ax.axhline(best, color="#8A8A8A", lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.text(0.14, best + 2, f"best of the model's {n_model} readouts: {best:.1f}%", fontsize=5.8, color="#6A6A6A", va="bottom")
    ax.axvline(hsb, color=S.HUMAN, lw=0.7, ls=(0, (2, 2)), zorder=1)
    ax.text(hsb - 0.01, -22, f"human rate {hsb:.3f}", fontsize=5.6, rotation=90, va="center", ha="right")
    ax.set_xlim(0.12, 1.0); ax.set_ylim(-40, 116)
    ax.set_xlabel("cue-conflict shape-bias rate")
    ax.set_ylabel("error consistency with 10 observers\n(κ, % of human–human ceiling)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.86), fontsize=5.6, handletextpad=0.3, labelspacing=0.25)
    print(f"  (b) best of the model's {n_model} readouts {best:.1f}% of ceiling; human rate {hsb:.3f}; ceiling {ceil:.4f}")
    S.letter(ax, "b", dx=-0.07, dy=0.03, title="Agreement does not follow")

    # ------------------------------------------------------------ (c) positive control (BA network)
    M = json.load(open(PERIMAGE))["models"]
    ax = fig.add_subplot(gs[0, 2])
    meas = [("shape_rate_resid__pearson", "cue\npicked"), ("valid_rate_resid__pearson", "any cue\nresolves")]
    arms = [("dkl_bcs_seed{}_final", S.BA, "BA network"), ("dkl_base_seed{}_final", S.BANODIET, "BA, no diet")]
    for mi, (key, mlab) in enumerate(meas):
        for ai, (tmpl, c, alab) in enumerate(arms):
            v = np.array([M[tmpl.format(s)][key]["oos_pct_ceiling"] for s in range(3)])
            x0 = mi + (ai - 0.5) * 0.38
            ax.bar(x0, v.mean(), width=0.3, color=c, alpha=0.3, edgecolor=c, lw=0.8, zorder=2)
            ax.plot(np.full(3, x0), v, "o", color=c, ms=2.8, mew=0, zorder=4)
            print(f"  (c) {mlab.replace(chr(10), ' '):24s} {alab:12s} {v.mean():5.1f}% [{v.min():.1f}, {v.max():.1f}]")
    ax.axhline(0, color="#BBBBBB", lw=0.7)
    ax.set_xticks([0, 1]); ax.set_xticklabels([m[1] for m in meas], fontsize=5.6)
    ax.set_xlim(-0.75, 1.75); ax.set_ylim(-2, 32)
    ax.set_ylabel("held-out correspondence\n(% of split-half ceiling)")
    h = [plt.Line2D([], [], color=c, lw=3, alpha=0.5, label=l) for _t, c, l in arms]
    ax.legend(handles=h, loc="upper left", fontsize=5.6, handlelength=1.0, borderpad=0.1)
    S.note(ax, "BA network; the model's\nper-image run is pending", x=0.5, y=-0.36, ha="center", va="top", size=5.4, style="italic")
    S.letter(ax, "c", dx=-0.095, dy=0.03, title="The measure works…")

    # ------------------------------------------------------------ (d) crossed 2x2 (BA network)
    ax = fig.add_subplot(gs[0, 3])
    cells = [("dkl_bcs_seed{}_final", 1, 1, S.BA), ("dkl_bcs_seed{}_final_nofe", 1, 0, S.BA),
             ("dkl_base_seed{}_final_fe", 0, 1, S.BANODIET), ("dkl_base_seed{}_final", 0, 0, S.BANODIET)]
    pts = {}
    for tmpl, wrow, fe, c in cells:
        v = np.array([M[tmpl.format(s)]["valid_rate_resid__pearson"]["oos_pct_ceiling"] for s in range(3)])
        matched = wrow == fe
        ax.plot([fe - 0.15, fe + 0.15], [v.mean()] * 2, "-", color=c, lw=2.0, alpha=1.0 if matched else 0.4, zorder=3)
        ax.plot(np.full(3, fe), v, "o", color=c, ms=2.8, mew=0, alpha=0.95 if matched else 0.4, zorder=4)
        pts.setdefault(wrow, []).append((fe, v.mean()))
        print(f"  (d) weights={'diet' if wrow else 'no diet':8s} front end={'on' if fe else 'off'}  {v.mean():+5.1f}%  {'matched' if matched else 'crossed'}")
    for wrow, c in [(1, S.BA), (0, S.BANODIET)]:
        p = sorted(pts[wrow]); ax.plot([q[0] for q in p], [q[1] for q in p], "-", color=c, lw=0.9, alpha=0.5, zorder=2)
    ax.axhline(0, color="#BBBBBB", lw=0.7)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["front end\noff", "front end\non"], fontsize=6.2)
    ax.set_xlim(-0.5, 1.5); ax.set_ylim(-14, 32)
    ax.set_ylabel("difficulty correspondence\n(% of ceiling)")
    h = [plt.Line2D([], [], color=c, lw=2, label=l) for c, l in [(S.BA, "BA diet weights"), (S.BANODIET, "no-diet weights")]]
    ax.legend(handles=h, loc="upper left", fontsize=5.6, handlelength=1.0, borderpad=0.1)
    S.note(ax, "solid: train/test matched; pale: crossed", x=0.5, y=-0.36, ha="center", va="top", size=5.4, style="italic")
    S.letter(ax, "d", dx=-0.095, dy=0.03, title="…but the diet reading dies")

    S.save(fig, "fig5_dissociations")


if __name__ == "__main__":
    main()
