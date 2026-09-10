"""Figure 3 — in human visual cortex, the texture-biased stream carries the
ventral end of the hierarchy, and the diet decides whether training builds or
erodes that gradient.

Voxelwise banded-ridge encoding on NSD (8 subjects), one feature band per
stream; every voxel's explained variance is partitioned into the part only the
magno band accounts for (Um) and the part only the parvo band does (Up).
Stream preference per ROI = median over reliable voxels of (Um − Up)/(Um + Up)
(noise ceiling > 0.1, Um + Up > 1e-3, ≥ 20 voxels) — the normalised statistic of §4.7.

  (a) stream-preference profile V1v → hV4 for the untrained network on the diet
      front end, the trained no-diet and diet arms (3 seeds each, seed-mean of
      per-subject profiles) and the channel-swap control.
  (b) the gradient hV4 − mean(V1v, V1d) per subject, untrained → trained, both arms.
  (c) Spearman ρ between hierarchical rank and stream preference, per subject,
      against the exact permutation null over all 630 labellings of the ROIs.
  (d) in-silico cue conflict: the 1,200 cue-conflict images pushed through the
      fitted encoding models give a per-voxel predicted shape-bias index; the
      magno − parvo difference, paired within subject.
  (e) what the diet costs in prediction: median r² / noise ceiling.

Sources: results/encoding/, results/encoding_cc/ (npz), results/hierarchy_permutation_null.json,
results/figures/fig1_data.npz (panels d, e).
Usage: PYTHONPATH=.:tmp_analysis/figs_v2 python tmp_analysis/figs_v2/fig3_cortex.py
"""
from __future__ import annotations

import json

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as sps

import style as S
from src.analysis.encoding.stats import load_results

NC_MIN, MIN_VOX = 0.1, 20
VIS = [1, 2, 3, 4, 5, 6, 7]
ROI = ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4"]
HIER = np.array([1, 1, 2, 2, 3, 3, 4])
D = np.load("results/figures/fig1_data.npz", allow_pickle=True)


def spi_profile(out_dir: str, model: str) -> np.ndarray:
    """(subjects × 7) stream-preference profile: per-voxel index (Um − Up)/(Um + Up),
    median over reliable voxels — the normalised statistic of the manuscript
    (visualrois.py convention: nc > 0.1, Um + Up > 1e-3, ≥ 20 voxels)."""
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


def rho(prof):
    return np.array([sps.spearmanr(HIER, p)[0] for p in prof])


def main():
    arms = {
        "untrained": ("results/encoding", ["untrained_seed0"], S.UNTRAINED),
        "no diet": ("results/encoding_cc", [f"dkl_base_seed{s}_trained" for s in range(3)], S.NODIET),
        "diet": ("results/encoding_cc", [f"dkl_bcs_seed{s}_trained" for s in range(3)], S.DIET),
        "swap": ("results/encoding", ["dkl_bcs_swap_seed0_trained"], S.SWAP),
        "untrained, no-diet front end": ("results/encoding", ["untrained_seed0_noblur"], "#A9A9A9"),
    }
    P = {k: np.stack([spi_profile(d, m) for m in ms]) for k, (d, ms, c) in arms.items()}  # seeds × subj × roi

    fig = plt.figure(figsize=(S.FULL, 5.0))
    gs = fig.add_gridspec(2, 3, hspace=0.62, wspace=0.50, left=0.075, right=0.985, top=0.91, bottom=0.10,
                          width_ratios=[1.35, 1, 1])

    # ------------------------------------------------------------ (a) profiles
    ax = fig.add_subplot(gs[0, :2])
    x = np.arange(7)
    S.refline(ax, y=0)
    order = ["untrained", "no diet", "diet", "swap"]
    for k in order:
        prof = P[k]                          # seeds × subj × roi
        m = prof.mean(0)                     # seed-mean per subject
        mu, se = m.mean(0), m.std(0, ddof=1) / np.sqrt(m.shape[0])
        c = arms[k][2]
        ls = (0, (3, 2)) if k in ("untrained",) else "-"
        ax.fill_between(x, mu - se, mu + se, color=c, alpha=0.18, lw=0, zorder=2)
        ax.plot(x, mu, ls=ls, color=c, lw=2.0, zorder=3, marker="o", ms=3.2, mec="white", mew=0.6)
        lab = {"untrained": "untrained (diet front end)", "no diet": "trained, no diet (3 seeds)",
               "diet": "trained, diet (3 seeds)", "swap": "trained, swap (1 seed)"}[k]
        ax.annotate(lab, (6, mu[-1]), xytext=(8, 0), textcoords="offset points", fontsize=6.2, color=c, va="center")
        print(f"  (a) {k:10s} profile " + " ".join(f"{v:+.3f}" for v in mu))
    ax.set_xticks(x); ax.set_xticklabels(ROI)
    ax.set_xlim(-0.3, 8.9)
    ax.set_ylabel("stream preference per ROI\nmedian of (Um − Up) / (Um + Up)")
    ax.set_xlabel("retinotopic area, V1 → hV4 (Wang et al. 2015 atlas)")
    ax.set_ylim(-1.02, 1.22)
    ax.text(-0.25, 1.20, "↑ magno-explained", fontsize=6.2, color=S.MUTED, va="top")
    ax.text(-0.25, -1.0, "↓ parvo-explained", fontsize=6.2, color=S.MUTED, va="bottom")
    S.note(ax, "line: mean over 8 subjects; shading: ± SEM", x=0.99, y=0.70, ha="right", va="center", size=6.0)
    S.letter(ax, "a", dx=-0.05, dy=0.03, title="Which stream explains which level of the ventral hierarchy")

    # ------------------------------------------------------------ (b) hV4 − V1v
    ax = fig.add_subplot(gs[0, 2])
    S.refline(ax, y=0)
    def grad(prof):                       # hV4 − mean(V1v, V1d), the manuscript's §4.7 contrast
        return prof[..., 6] - prof[..., :2].mean(-1)
    g_unt_d = grad(P["untrained"][0])
    g_unt_n = grad(P["untrained, no-diet front end"][0])
    for k in ("diet", "no diet"):
        for sd in range(3):
            print(f"  (b) {k:8s} seed {sd}: hV4 − mean(V1v,V1d) = {grad(P[k][sd]).mean():+.4f}")
    g_d = grad(P["diet"]).mean(0)         # seed-mean per subject
    g_n = grad(P["no diet"]).mean(0)
    for a, b in zip(g_unt_n, g_n):
        ax.plot([0, 1], [a, b], "-", color=S.NODIET, lw=0.6, alpha=0.35, zorder=2)
    for a, b in zip(g_unt_d, g_d):
        ax.plot([0, 1], [a, b], "-", color=S.DIET, lw=0.6, alpha=0.35, zorder=2)
    for u, t, c in [(g_unt_n, g_n, S.NODIET), (g_unt_d, g_d, S.DIET)]:
        ax.plot([0, 1], [u.mean(), t.mean()], "-", color=c, lw=2.4, zorder=4)
        ax.plot([0, 1], [u.mean(), t.mean()], "o", color=c, ms=5.5, mec="white", mew=0.9, zorder=5)
        ax.annotate(f"{t.mean():+.2f}  {'diet' if c == S.DIET else 'no diet'}", (1, t.mean()), xytext=(7, 0), textcoords="offset points",
                    fontsize=6.4, color=c, fontweight="bold", va="center")
    ax.annotate(f"{g_unt_d.mean():+.2f}", (0, g_unt_d.mean()), xytext=(-7, 0), textcoords="offset points",
                fontsize=6.4, color=S.UNTRAINED, va="center", ha="right")
    pd_ = sps.wilcoxon(g_d - g_n).pvalue
    print(f"  (b) untrained (diet fe) {g_unt_d.mean():+.3f}  untrained (noblur fe) {g_unt_n.mean():+.3f}  "
          f"diet {g_d.mean():+.3f}  no diet {g_n.mean():+.3f}  diet<nodiet {(g_d < g_n).sum()}/8 p={pd_:.4f}")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["untrained", "trained"])
    ax.set_xlim(-0.55, 2.05); ax.set_ylim(-1.1, 0.08)
    ax.set_ylabel("gradient  hV4 − mean(V1v, V1d)")
    S.note(ax, f"diet steeper in {(g_d < g_n).sum()}/8\nsubjects, p = {pd_:.4f}", x=0.02, y=0.03, va="bottom", size=6.0)
    S.letter(ax, "b", dx=-0.06, dy=0.03, title="Learning builds it or erodes it")

    # ------------------------------------------------------------ (c) monotonicity vs exact null
    ax = fig.add_subplot(gs[1, 0])
    null = json.load(open("results/hierarchy_permutation_null.json"))
    q025 = np.mean([e["null_q025"] for e in null if e["stat"] == "norm"])
    ax.axhspan(q025, 1.0, color="#F0F0F0", lw=0, zorder=0)
    ax.axhline(q025, color="#B0B0B0", lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.text(0.03, 0.97, "shaded: within the exact null over\nall 630 labellings of the 7 ROIs\n(2.5th percentile as dashed rule)", transform=ax.transAxes,
            fontsize=5.8, color=S.MUTED, va="top", ha="left")
    S.refline(ax, y=-1.0, color="#D8D8D8")
    cols = [("untrained", P["untrained"][0], S.UNTRAINED), ("no diet", P["no diet"].mean(0), S.NODIET),
            ("diet", P["diet"].mean(0), S.DIET)]
    for i, (k, prof, c) in enumerate(cols):
        r = rho(prof)
        xs = i + np.linspace(-0.14, 0.14, len(r))
        ax.plot(xs, r, "o", color=c, ms=3.6, mec="white", mew=0.6, zorder=4)
        ax.plot([i - 0.22, i + 0.22], [r.mean()] * 2, "-", color=c, lw=2.2, zorder=5)
        ax.annotate(f"{r.mean():+.2f}", (i + 0.24, r.mean()), xytext=(3, 0), textcoords="offset points",
                    fontsize=6.6, color=c, fontweight="bold", ha="left", va="center")
        print(f"  (c) {k:10s} rho mean {r.mean():+.3f}  range {r.min():+.3f}..{r.max():+.3f}")
    pdiet = [e["p_exact_one_sided"] for e in null if e["arm"] == "diet" and e["stat"] == "norm"]
    ax.set_xticks(range(3)); ax.set_xticklabels(["untrained", "trained,\nno diet", "trained,\ndiet"], fontsize=6.4)
    ax.set_xlim(-0.5, 2.85); ax.set_ylim(-1.12, 1.05)
    ax.set_ylabel("Spearman ρ, hierarchical rank\nvs stream preference (per subject)")
    S.note(ax, f"diet: the most extreme\nof all 630 labellings,\n3/3 seeds, p = {pdiet[0]:.4f}", x=0.98, y=0.66, ha="right", va="top", size=5.8, color=S.DIET)
    S.letter(ax, "c", dx=-0.06, dy=0.03, title="The diet gradient is near-perfectly monotonic")

    # ------------------------------------------------------------ (d) in-silico cue conflict
    ax = fig.add_subplot(gs[1, 1])
    S.refline(ax, y=0)
    bn = D["b__nodiet__magno"] - D["b__nodiet__parvo"]
    bd = D["b__diet__magno"] - D["b__diet__parvo"]
    S.paired(ax, bn, bd, S.NODIET, S.DIET, label_fmt="{:+.2f}")
    p = sps.wilcoxon(bd - bn).pvalue
    ax.set_xticks([0, 1]); ax.set_xticklabels(["no diet", "diet"]); ax.set_xlim(-0.7, 1.7)
    ax.set_ylabel("predicted shape-bias index\nmagno − parvo (per subject)")
    S.note(ax, f"p = {p:.4f}\n{(bd > bn).sum()}/8 subjects", x=0.03, y=0.80, size=6.2)
    print(f"  (d) no diet {bn.mean():+.3f}  diet {bd.mean():+.3f}  p={p:.4f}")
    S.letter(ax, "d", dx=-0.06, dy=0.03, title="Cue conflict, through the fitted cortex")

    # ------------------------------------------------------------ (e) cost
    ax = fig.add_subplot(gs[1, 2])
    dn, dd = D["d__nodiet__r2nc"], D["d__diet__r2nc"]
    S.paired(ax, dn, dd, S.NODIET, S.DIET, label_fmt="{:.3f}")
    p = sps.wilcoxon(dd - dn).pvalue
    ax.set_xticks([0, 1]); ax.set_xticklabels(["no diet", "diet"]); ax.set_xlim(-0.7, 1.7)
    lo, hi = min(dd.min(), dn.min()), max(dd.max(), dn.max())
    ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.3 * (hi - lo))
    ax.set_ylabel("median r² / noise ceiling\n(reliable voxels, nsdgeneral)")
    S.note(ax, f"the diet predicts BOLD {100 * (1 - dd.mean() / dn.mean()):.0f}% worse\np = {p:.4f}, {(dd < dn).sum()}/8 subjects", x=0.03, y=0.97, size=6.2)
    print(f"  (e) no diet {dn.mean():.3f}  diet {dd.mean():.3f}  p={p:.4f}")
    S.letter(ax, "e", dx=-0.06, dy=0.03, title="What the diet costs")

    S.save(fig, "fig3_cortex")


if __name__ == "__main__":
    main()
