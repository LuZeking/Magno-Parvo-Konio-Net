"""Figure 3 — in human visual cortex the magno stream carries V1 and the
texture-biased stream the rest of the hierarchy, and only with a maturing input.

Voxelwise banded-ridge encoding on NSD (8 subjects), one feature band per
stream; every voxel's explained variance is partitioned into the part only the
magno band accounts for (Um) and the part only the parvo band does (Up).
Stream preference per ROI = median over reliable voxels of (Um − Up)/(Um + Up)
(noise ceiling > 0.1, Um + Up > 1e-3, ≥ 20 voxels) — the normalised statistic of Methods 4.7.

  (a) stream-preference profile V1v → hV4: the untrained network on the model's
      test input, the model, its control and the BA network's two arms (each
      subject's seed mean; line = mean over subjects, shading ± SEM).
  (b) the gradient hV4 − mean(V1v, V1d) per subject: untrained → model and
      control (one untrained network on one test input), and the BA network.
  (c) Spearman ρ between hierarchical rank and stream preference per subject
      (seed-mean profile), against the exact null over all 630 labellings of the ROIs.
  (d) in-silico cue conflict: the 1,200 cue-conflict images pushed through the
      fitted encoding models give a per-voxel predicted shape-bias index; the
      magno − parvo difference per subject.
  (e) median r² / noise ceiling (reliable voxels, nsdgeneral) per subject.

Sources: docs/slides_2026-10-12_ba_vs_fusion/figures/fs2_cortex_summary.json
(tmp_analysis/fs2_cortex_summary.py: every trained arm), the untrained network's
fits in /scratch/byrong/mpk/encoding/results/2026-10-16_fs2 (panels a, b).
Usage: PYTHONPATH=.:tmp_analysis/figs_v2 python tmp_analysis/figs_v2/fig3_cortex.py
"""
from __future__ import annotations

from itertools import permutations

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as sps

import style as S
import v3runs as R
from src.analysis.encoding.stats import load_results

NC_MIN, MIN_VOX = 0.1, 20
VIS = [1, 2, 3, 4, 5, 6, 7]
ROI = ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4"]
HIER = np.array([1, 1, 2, 2, 3, 3, 4])
FS_ENC = "/scratch/byrong/mpk/encoding/results/2026-10-16_fs2"
UNTRAINED = "untrained_dvd300_seed0"
ARMS = [("model", "dvd300", S.MODEL), ("control", "control", S.CONTROL),
        ("BA", "BA diet", S.BA), ("BA no diet", "BA no diet", S.BANODIET)]


def spi_profile(out_dir: str, model: str) -> np.ndarray:
    """(subjects × 7) stream-preference profile: per-voxel index (Um − Up)/(Um + Up),
    median over reliable voxels (visualrois.py convention: nc > 0.1, Um + Up > 1e-3, ≥ 20 voxels)."""
    res = load_results(out_dir, model)
    subs = sorted(res)
    prof = np.full((len(subs), len(VIS)), np.nan)
    for i, s in enumerate(subs):
        z = res[s]
        roi, nc = z["roi_prf-visualrois"], z["noise_ceiling"]
        um, up = z["unique_magno"], z["unique_parvo"]
        for j, lab in enumerate(VIS):
            v = (roi == lab) & (nc > NC_MIN) & ((um + up) > 1e-3)
            if v.sum() >= MIN_VOX:
                r = (um[v] - up[v]) / (um[v] + up[v])
                prof[i, j] = np.median(r[np.isfinite(r)])
    return prof


def grad(prof):                       # hV4 − mean(V1v, V1d), Methods 4.7
    return prof[..., 6] - prof[..., :2].mean(-1)


def rho(prof):
    return np.array([sps.spearmanr(HIER, p)[0] for p in prof])


def null_q025():
    """2.5th percentile of ρ over the 630 distinct labellings of the seven ROIs (a profile without ties)."""
    labs = set(permutations(HIER))
    null = np.array([sps.spearmanr(np.array(l), np.arange(7))[0] for l in labs])
    return len(labs), np.percentile(null, 2.5)


def columns(ax, vals, colours, labels, fmt):
    """Per-subject values in columns, subjects joined by thin lines, column means as bars."""
    V = np.array(vals)                                   # columns × subjects
    for s in range(V.shape[1]):
        ax.plot(range(len(V)), V[:, s], "-", color=S.UNTRAINED, lw=0.6, alpha=0.3, zorder=2)
    for i, (v, c) in enumerate(zip(V, colours)):
        ax.plot(np.full(len(v), i), v, "o", color=c, ms=3.4, mec="white", mew=0.6, zorder=4)
        ax.plot([i - 0.2, i + 0.2], [v.mean()] * 2, "-", color=c, lw=2.2, zorder=5)
        ax.annotate(fmt.format(v.mean()), (i + 0.22, v.mean()), xytext=(2, 0), textcoords="offset points",
                    fontsize=6.0, color=S.INK, fontweight="bold", va="center")
    ax.set_xticks(range(len(V))); ax.set_xticklabels(labels, fontsize=6.2)
    ax.set_xlim(-0.5, len(V) - 0.1)


def main():
    C = R.cortex()["arms"]
    P = {k: np.array(C[a]["profile_seedmean_per_subject"]) for k, a, _ in ARMS}       # subj × roi
    P["untrained"] = spi_profile(FS_ENC, UNTRAINED)
    n_ctl = len(C["control"]["grad_per_seed"])

    fig = plt.figure(figsize=(S.FULL, 5.0))
    gs = fig.add_gridspec(2, 3, hspace=0.62, wspace=0.52, left=0.075, right=0.985, top=0.91, bottom=0.10,
                          width_ratios=[1.35, 1, 1])

    # ------------------------------------------------------------ (a) profiles
    ax = fig.add_subplot(gs[0, :2])
    x = np.arange(7)
    S.refline(ax, y=0)
    lines = [("untrained", S.UNTRAINED, (0, (3, 2)), "untrained, model's test input"),
             ("BA no diet", S.BANODIET, "-", "BA network, no diet (3 seeds)"),
             ("control", S.CONTROL, "-", f"control ({R.fmt_seeds(n_ctl)})"),
             ("BA", S.BA, "-", "BA network (3 seeds)"),
             ("model", S.MODEL, "-", "model (3 seeds)")]
    ends = []
    for k, c, ls, lab in lines:
        m = P[k]
        mu, se = m.mean(0), m.std(0, ddof=1) / np.sqrt(m.shape[0])
        lw = 2.4 if k == "model" else 1.6
        ax.fill_between(x, mu - se, mu + se, color=c, alpha=0.15, lw=0, zorder=2)
        ax.plot(x, mu, ls=ls, color=c, lw=lw, zorder=3, marker="o", ms=3.0, mec="white", mew=0.6)
        ends.append([mu[-1], lab, c])
        print(f"  (a) {k:10s} profile " + " ".join(f"{v:+.3f}" for v in mu))
    ends.sort(key=lambda e: e[0])
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.13)
    for y, lab, c in ends:
        ax.annotate(lab, (6, y), xytext=(10, 0), textcoords="offset points", fontsize=6.0, color=S.INK, va="center")
        ax.plot([6.22], [y], "s", color=c, ms=3, mew=0, clip_on=False)
    ax.set_xticks(x); ax.set_xticklabels(ROI)
    ax.set_xlim(-0.3, 9.2)
    ax.set_ylabel("stream preference per ROI\nmedian of (Um − Up) / (Um + Up)")
    ax.set_xlabel("retinotopic area, V1 → hV4 (Wang et al. 2015 atlas)")
    ax.set_ylim(-1.02, 1.22)
    ax.text(-0.25, 1.20, "↑ magno-explained", fontsize=6.2, color=S.MUTED, va="top")
    ax.text(-0.25, -1.0, "↓ parvo-explained", fontsize=6.2, color=S.MUTED, va="bottom")
    S.letter(ax, "a", dx=-0.05, dy=0.03, title="Which stream explains which level of the hierarchy")

    # ------------------------------------------------------------ (b) the gradient
    ax = fig.add_subplot(gs[0, 2])
    S.refline(ax, y=0)
    gu = grad(P["untrained"])
    G = {k: np.array(C[a]["grad_seedmean_per_subject"]) for k, a, _ in ARMS}
    for k, c in (("control", S.CONTROL), ("model", S.MODEL)):
        for a, b in zip(gu, G[k]):
            ax.plot([0, 1], [a, b], "-", color=c, lw=0.6, alpha=0.35, zorder=2)
        ax.plot([0, 1], [gu.mean(), G[k].mean()], "-", color=c, lw=2.4, zorder=4)
        ax.plot([1], [G[k].mean()], "o", color=c, ms=5.5, mec="white", mew=0.9, zorder=5)
        ax.annotate(f"{G[k].mean():+.2f}  {k}", (1, G[k].mean()), xytext=(7, 0), textcoords="offset points",
                    fontsize=6.2, color=S.INK, fontweight="bold", va="center")
    ax.plot([0], [gu.mean()], "o", color=S.UNTRAINED, ms=5.5, mec="white", mew=0.9, zorder=5)
    ax.annotate(f"{gu.mean():+.2f}", (0, gu.mean()), xytext=(-7, 0), textcoords="offset points",
                fontsize=6.2, color=S.INK, va="center", ha="right")
    S.seeds(ax, 2.2, G["BA"], S.BA, ms=3.4, bar_w=0.18)
    ax.annotate(f"{G['BA'].mean():+.2f}", (2.2, G["BA"].mean()), xytext=(8, 0), textcoords="offset points",
                fontsize=6.2, color=S.INK, va="center")
    d = G["model"] - G["control"]
    p = sps.wilcoxon(d).pvalue
    print(f"  (b) untrained {gu.mean():+.3f} (n={len(gu)}, {(gu < 0).sum()}/8 < 0)  model {G['model'].mean():+.3f}  "
          f"control {G['control'].mean():+.3f}  BA {G['BA'].mean():+.3f}  model<control {(d < 0).sum()}/8 p={p:.4f}  "
          f"model<BA {(G['model'] < G['BA']).sum()}/8")
    ax.set_xticks([0, 1, 2.2]); ax.set_xticklabels(["untrained", "trained", "BA\nnetwork"], fontsize=6.4)
    ax.set_xlim(-0.9, 2.75); ax.set_ylim(-1.55, 0.12)
    ax.set_ylabel("gradient  hV4 − mean(V1v, V1d)")
    S.note(ax, f"model steeper than control\nin {(d < 0).sum()}/8 subjects, p = {p:.4f}", x=0.98, y=0.03, ha="right", va="bottom", size=5.6)
    S.letter(ax, "b", dx=-0.06, dy=0.03, title="Learning keeps it or erases it")

    # ------------------------------------------------------------ (c) monotonicity vs exact null
    ax = fig.add_subplot(gs[1, 0])
    n_lab, q025 = null_q025()
    ax.axhspan(q025, 1.0, color="#F0F0F0", lw=0, zorder=0)
    ax.axhline(q025, color="#B0B0B0", lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.text(0.03, 0.97, f"shaded: within the exact null over\nall {n_lab} labellings of the 7 ROIs\n(2.5th percentile as dashed rule)",
            transform=ax.transAxes, fontsize=5.6, color=S.MUTED, va="top", ha="left")
    S.refline(ax, y=-1.0, color="#D8D8D8")
    cols = [("untrained", P["untrained"], S.UNTRAINED)] + [(k, P[k], c) for k, _a, c in ARMS]
    for i, (k, prof, c) in enumerate(cols):
        r = rho(prof)
        xs = i + np.linspace(-0.14, 0.14, len(r))
        ax.plot(xs, r, "o", color=c, ms=3.2, mec="white", mew=0.5, zorder=4)
        ax.plot([i - 0.22, i + 0.22], [r.mean()] * 2, "-", color=c, lw=2.2, zorder=5)
        print(f"  (c) {k:10s} rho mean {r.mean():+.3f}  range {r.min():+.3f}..{r.max():+.3f}")
    for k, a, _c in ARMS[:2]:
        print(f"  (c) {k} rho per seed (summary) " + " ".join(f"{v:+.3f}" for v in C[a]["rho_per_seed"]))
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(["untr.", "model", "control", "BA", "BA no\ndiet"], fontsize=6.0)
    ax.set_xlim(-0.5, len(cols) - 0.5); ax.set_ylim(-1.12, 1.05)
    ax.set_ylabel("Spearman ρ, hierarchical rank\nvs stream preference (per subject)")
    S.letter(ax, "c", dx=-0.06, dy=0.03, title="The model's gradient is monotonic")

    # ------------------------------------------------------------ (d) in-silico cue conflict
    ax = fig.add_subplot(gs[1, 1])
    S.refline(ax, y=0)
    cc = [np.array(C[a]["cc_seedmean_per_subject"]) for _k, a, _c in ARMS[:3]]
    columns(ax, cc, [S.MODEL, S.CONTROL, S.BA], ["model", "control", "BA"], "{:+.2f}")
    p = sps.wilcoxon(cc[0] - cc[1]).pvalue
    print(f"  (d) model {cc[0].mean():+.3f}  control {cc[1].mean():+.3f}  BA {cc[2].mean():+.3f}  model>control {(cc[0] > cc[1]).sum()}/8 p={p:.4f}")
    ax.set_ylabel("predicted shape-bias index\nmagno − parvo (per subject)")
    ax.set_ylim(-0.15, 1.75)
    S.letter(ax, "d", dx=-0.06, dy=0.03, title="Cue conflict, through the fitted cortex")

    # ------------------------------------------------------------ (e) prediction
    ax = fig.add_subplot(gs[1, 2])
    r2 = [np.array(C[a]["r2nc_seedmean_per_subject"]) for _k, a, _c in ARMS[:3]]
    columns(ax, r2, [S.MODEL, S.CONTROL, S.BA], ["model", "control", "BA"], "{:.3f}")
    p1, p2 = sps.wilcoxon(r2[0] - r2[1]).pvalue, sps.wilcoxon(r2[0] - r2[2]).pvalue
    print(f"  (e) model {r2[0].mean():.3f}  control {r2[1].mean():.3f}  BA {r2[2].mean():.3f}  "
          f"model<control {(r2[0] < r2[1]).sum()}/8 p={p1:.4f}  model>BA {(r2[0] > r2[2]).sum()}/8 p={p2:.4f}")
    ax.set_ylabel("median r² / noise ceiling\n(reliable voxels, nsdgeneral)")
    S.letter(ax, "e", dx=-0.06, dy=0.03, title="What the trajectory costs")

    S.save(fig, "fig3_cortex")


if __name__ == "__main__":
    main()
