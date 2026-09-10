"""Figure 2 — the two streams specialise in opposite directions, by different routes.

  (a) linear-probe shape−texture bias per stream before and after training,
      with and without the diet. The magno stream's shape bias is supplied by
      the input and survives training only under the diet; the parvo stream's
      texture bias is built by learning.
  (b, c) the carrier × transform 2×2 on the probe (b) and on cue-conflict
      behaviour (c): only an achromatic carrier WITH the transform crosses the
      shape/texture boundary. Seeds are the replication unit (3, or 2 for one cell).
  (d, e) accuracy-matched control: a no-diet network brought to the diet's
      accuracy does not acquire the shape bias (probe d, behaviour e). One
      matched run — read as n = 1.
  (f) how the streams decide cue-conflict images: shape / texture / neither.
      The denominator caveat, drawn.

Sources: results/shape_texture*/, results/cue_conflict*/, results/figures/fig1_data.npz.
Usage: PYTHONPATH=.:tmp_analysis/figs_v2 python tmp_analysis/figs_v2/fig2_streams.py
"""
from __future__ import annotations

import glob
import json

import numpy as np
import matplotlib.pyplot as plt

import style as S

ST, ST_BEST, ST_MATCH = "results/shape_texture", "results/shape_texture_bestep", "results/shape_texture_matched"
CC, CC_BEST, CC_MATCH = "results/cue_conflict", "results/cue_conflict_bestep", "results/cue_conflict_matched"
D = np.load("results/figures/fig1_data.npz", allow_pickle=True)


def probe(path, stream):
    return json.load(open(path))["streams"][stream]["bias"]


def probe_arm(pattern, stream):
    fs = sorted(glob.glob(pattern))
    assert fs, pattern
    return np.array([probe(f, stream) for f in fs])


def cc_arm(pattern, cond, key="shape_bias"):
    fs = sorted(f for f in glob.glob(pattern) if f.endswith(".json"))
    assert fs, pattern
    return np.array([json.load(open(f))["conditions"][cond][key] for f in fs])


def main():
    fig = plt.figure(figsize=(S.FULL, 4.9))
    gs = fig.add_gridspec(2, 3, hspace=0.70, wspace=0.45, left=0.075, right=0.985,
                          top=0.90, bottom=0.12, width_ratios=[1.15, 1, 1])

    # ------------------------------------------------------------ (a) routes
    ax = fig.add_subplot(gs[0, 0])
    um_d, up_d = probe(f"{ST}/untrained_blur_cs.json", "magno"), probe(f"{ST}/untrained_blur_cs.json", "parvo")
    um_n, up_n = probe(f"{ST}/untrained_noblur.json", "magno"), probe(f"{ST}/untrained_noblur.json", "parvo")
    tm_d, tp_d = probe_arm(f"{ST}/dkl_bcs_seed?_trained.json", "magno"), probe_arm(f"{ST}/dkl_bcs_seed?_trained.json", "parvo")
    tm_n, tp_n = probe_arm(f"{ST}/dkl_base_seed?_trained.json", "magno"), probe_arm(f"{ST}/dkl_base_seed?_trained.json", "parvo")
    S.refline(ax, y=0)
    routes = [("magno, diet", um_d, tm_d, S.DIET, "-"),
              ("magno, no diet", um_n, tm_n, S.DIET, (0, (3, 2))),
              ("parvo, diet", up_d, tp_d, S.NODIET, "-"),
              ("parvo, no diet", up_n, tp_n, S.NODIET, (0, (3, 2)))]
    ends = []
    for lab, u, t, c, ls in routes:
        ax.plot([0, 1], [u, t.mean()], ls=ls, color=c, lw=1.8, zorder=3)
        ax.plot([0], [u], "o", color=c, ms=4.5, mec="white", mew=0.8, zorder=4)
        ax.plot([1, 1], [t.min(), t.max()], "-", color=c, lw=3.5, alpha=0.30, zorder=2, solid_capstyle="butt")
        ax.plot([1], [t.mean()], "o", color=c, ms=4.5, mec="white", mew=0.8, zorder=4)
        ends.append([t.mean(), f"{lab}  {t.mean() - u:+.2f}", c])
        print(f"  (a) {lab:15s} untrained {u:+.3f} -> trained {t.mean():+.3f} (3-seed range {t.min():+.3f}..{t.max():+.3f})")
    ends.sort(key=lambda e: e[0])
    for i in range(1, len(ends)):                       # push labels apart
        if ends[i][0] - ends[i - 1][0] < 0.11:
            ends[i][0] = ends[i - 1][0] + 0.11
    for y, txt, c in ends:
        ax.annotate(txt, (1, y), xytext=(7, 0), textcoords="offset points", fontsize=6.2, color=c, va="center")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["untrained", "trained\n(150 epochs)"])
    ax.set_xlim(-0.25, 1.9); ax.set_ylim(-0.72, 1.12)
    ax.set_ylabel("linear-probe bias\nshape − texture accuracy")
    ax.text(-0.22, 0.03, "shape", fontsize=6, color=S.MUTED, va="bottom")
    ax.text(-0.22, -0.03, "texture", fontsize=6, color=S.MUTED, va="top")
    h = [plt.Line2D([], [], color=S.MUTED, lw=1.8, ls="-", label="diet"),
         plt.Line2D([], [], color=S.MUTED, lw=1.8, ls=(0, (3, 2)), label="no diet")]
    ax.legend(handles=h, loc="upper left", fontsize=6.2, handlelength=1.6, bbox_to_anchor=(-0.02, 1.04), ncol=2, columnspacing=1.0)
    S.letter(ax, "a", dx=-0.06, dy=0.03, title="Two routes to opposite biases")

    # ---------------------------------------------- (b, c) carrier × transform
    for j, (measure, ylab, neutral, ylim, letter, title) in enumerate([
            ("probe", "linear-probe bias (shape − texture)", 0.0, (-0.65, 1.0), "b", "Carrier × transform: probe"),
            ("beh", "cue-conflict shape bias", 0.5, (0.25, 1.0), "c", "…and behaviour")]):
        ax = fig.add_subplot(gs[0, 1 + j])
        S.refline(ax, y=neutral)
        for carrier, c, lab in [("chromatic", S.NODIET, "chromatic carrier (L−M)"),
                                ("achromatic", S.DIET, "achromatic carrier (luminance)")]:
            ys = [D[f"a__{carrier}__{t}__{measure}"] for t in ("none", "transform")]
            for xi, v in zip((0, 1), ys):
                ax.plot([xi] * len(v), v, "o", color=c, ms=2.8, alpha=0.6, mec="none", zorder=3)
            m = [v.mean() for v in ys]
            ax.plot([0, 1], m, "-", color=c, lw=2.0, zorder=4)
            ax.plot([0, 1], m, "o", color=c, ms=5, mec="white", mew=0.9, zorder=5, label=lab)
            print(f"  ({letter}) {carrier:10s} none {m[0]:+.3f}  transform {m[1]:+.3f}")
        conj = D[f"a__achromatic__transform__{measure}"].mean()
        ax.annotate(f"{conj:+.2f}" if measure == "probe" else f"{conj:.2f}", (1, conj), xytext=(0, 7),
                    textcoords="offset points", color=S.DIET, fontsize=7.2, fontweight="bold", ha="center")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["no\ntransform", "with\ntransform"])
        ax.set_xlim(-0.3, 1.3); ax.set_ylim(*ylim)
        ax.set_ylabel(ylab)
        if j == 0:
            ax.legend(loc="upper left", fontsize=6.2, handlelength=1.2, bbox_to_anchor=(-0.02, 1.0))
        S.letter(ax, letter, dx=-0.055, dy=0.03, title=title)

    # ------------------------------------------- (d, e) accuracy-matched control
    d_probe = probe_arm(f"{ST_BEST}/dkl_bcs_seed?_ep*.json", "magno")
    n_probe = probe_arm(f"{ST}/dkl_base_seed?_trained.json", "magno")
    m_probe = probe_arm(f"{ST_MATCH}/*.json", "magno")
    d_beh = cc_arm(f"{CC_BEST}/dkl_bcs_seed?_ep*.json", "magno_only")
    n_beh = cc_arm(f"{CC}/dkl_base_seed?.json", "magno_only")
    m_beh = cc_arm(f"{CC_MATCH}/*.json", "magno_only")
    for j, (vals, ylab, neutral, ylim, letter, title) in enumerate([
            ((d_probe, n_probe, m_probe), "magno-stream probe bias", 0.0, (-0.45, 1.0), "e", "Accuracy-matched control: probe"),
            ((d_beh, n_beh, m_beh), "magno-stream shape bias\n(cue-conflict)", 0.5, (0.3, 1.0), "f", "…and behaviour")]):
        ax = fig.add_subplot(gs[1, 1 + j])
        S.refline(ax, y=neutral)
        for x, v, c in [(0, vals[0], S.DIET), (1, vals[1], S.NODIET), (2, vals[2], S.MATCH)]:
            S.seeds(ax, x, v, c, ms=3.6, bar_w=0.2)
        gap = vals[0].mean() - vals[1].mean()
        closed = (vals[2].mean() - vals[1].mean()) / gap
        ax.annotate("", (2.35, vals[0].mean()), (2.35, vals[1].mean()),
                    arrowprops=dict(arrowstyle="<->", color=S.MUTED, lw=0.7))
        ax.text(2.42, (vals[0].mean() + vals[1].mean()) / 2, f"gap\n{gap:+.2f}", fontsize=5.8, color=S.MUTED, va="center")
        ax.text(2, vals[2].mean() - (0.07 if j == 0 else 0.03), f"closes {100 * closed:.1f}%\nof the gap", ha="center",
                va="top", fontsize=6.0, color=S.MATCH)
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["diet\n0.556", "no diet\n0.70", "no diet,\nmatched\n0.548 (n=1)"], fontsize=6.2)
        ax.set_xlabel("validation accuracy at the scored checkpoint", fontsize=6.2, color=S.MUTED)
        ax.set_xlim(-0.5, 2.9); ax.set_ylim(*ylim)
        ax.set_ylabel(ylab)
        print(f"  ({letter}) diet {vals[0].mean():+.3f}  no diet {vals[1].mean():+.3f}  matched {vals[2].mean():+.3f}  closes {100*closed:.1f}%")
        S.letter(ax, letter, dx=-0.055, dy=0.03, title=title)

    # ------------------------------------------------- (f) decision composition
    ax = fig.add_subplot(gs[1, 0])
    conds = [("magno_only", "magno\nonly"), ("full", "both\nw = 0.5"), ("parvo_only", "parvo\nonly")]
    arms = [("diet", f"{CC_BEST}/dkl_bcs_seed?_ep*.json", S.DIET), ("no diet", f"{CC}/dkl_base_seed?.json", S.NODIET)]
    for ci, (cond, clab) in enumerate(conds):
        for ai, (alab, pat, c) in enumerate(arms):
            ns = cc_arm(pat, cond, "n_shape").mean(); nt = cc_arm(pat, cond, "n_texture").mean()
            no = cc_arm(pat, cond, "n_other").mean()
            x = ci * 2.4 + ai * 1.0
            ax.bar(x, ns, width=0.85, color=c, alpha=0.95, zorder=3)
            ax.bar(x, nt, bottom=ns, width=0.85, color=c, alpha=0.45, zorder=3)
            ax.bar(x, no, bottom=ns + nt, width=0.85, color="#E4E4E4", zorder=3)
            sb = ns / (ns + nt)
            ax.text(x, 1230, f"{sb:.2f}", ha="center", va="bottom", fontsize=6.4, color=c, fontweight="bold")
            ax.text(x, -60, alab, ha="center", va="top", fontsize=5.8, color=c)
            print(f"  (d) {cond:10s} {alab:8s} shape {ns:5.0f} texture {nt:5.0f} neither {no:5.0f}  bias {sb:.3f}")
        ax.text(ci * 2.4 + 0.5, -230, clab, ha="center", va="top", fontsize=6.2, color=S.INK, linespacing=1.0)
    ax.set_xticks([])
    ax.set_ylim(0, 1750); ax.set_yticks([0, 400, 800, 1200]); ax.set_ylabel("cue-conflict images (of 1,200)")
    ax.text(-0.95, 1345, "shape bias", fontsize=5.8, color=S.MUTED, va="bottom", ha="left", style="italic")
    h = [plt.Rectangle((0, 0), 1, 1, color=S.MUTED, alpha=0.95, label="decided as shape"),
         plt.Rectangle((0, 0), 1, 1, color=S.MUTED, alpha=0.45, label="decided as texture"),
         plt.Rectangle((0, 0), 1, 1, color="#E4E4E4", label="neither category")]
    ax.legend(handles=h, loc="upper right", fontsize=5.9, ncol=1, handlelength=1.0, bbox_to_anchor=(1.03, 1.03), labelspacing=0.3)
    ax.set_xlim(-1.0, 6.3)
    S.letter(ax, "d", dx=-0.06, dy=0.03, title="How each readout decides the images")

    S.save(fig, "fig2_streams")


if __name__ == "__main__":
    main()
