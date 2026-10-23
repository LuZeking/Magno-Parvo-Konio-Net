"""Figure 4 — behaviour reads out from both streams, and the shape-bias rate is a
property of that readout.

  (a) cue-conflict shape bias against the readout weight w on the magno stream
      (full = classifier(w·y₁ + (1−w)·y₂)), 21 values: the model (3 seeds), its
      control and the channel swap (final checkpoints), the BA network (best saved
      checkpoints) for comparison. Representations never change along the x-axis. The ten human
      observers and the three published CNNs are drawn on the same y-axis.
  (b) does the span survive a readout that is refitted, and one that is not
      affine? Spans of shape bias over w for the frozen affine head, a refitted
      linear head and a two-layer nonlinear head, per seed of the model; dashed:
      within 5 points of peak validation accuracy; open: the BA network's medians.
  (c) where every model sits: shape bias against validation top-1 at the scored
      checkpoint, for each network's two streams alone and integrated.

Sources: results/fusion_sweep/, results/nonlinear_readout_summary.json,
results/cue_conflict*/, run histories (v3runs.py), human observers via
tmp_analysis/human_shapebias.py.
Usage: PYTHONPATH=.:tmp_analysis:tmp_analysis/figs_v2 python tmp_analysis/figs_v2/fig4_readout.py
"""
from __future__ import annotations

import contextlib
import glob
import io
import json
import os
import re

import numpy as np
import matplotlib.pyplot as plt

import style as S
import v3runs as R

PUBLISHED = [("AlexNet", 0.429), ("GoogLeNet", 0.312), ("VGG-16", 0.172)]   # Geirhos et al. 2018a release
READOUTS = [("magno_only", "^", "magno alone"), ("full", "o", "integrated"), ("parvo_only", "v", "parvo alone")]


def humans():
    import sys
    sys.path.insert(0, "tmp_analysis")
    with contextlib.redirect_stdout(io.StringIO()):
        from human_shapebias import human_observers
        return np.array([o["shape_bias"] for o in human_observers()])


def sweeps(runs_ckpt):
    rows = [R.sweep(r) for r in runs_ckpt]
    return rows[0][0], np.array([r[1] for r in rows])


def ba_best():
    """BA diet best saved checkpoints: (run, epoch) per seed from results/cue_conflict_bestep/."""
    out = []
    for f in sorted(glob.glob("results/cue_conflict_bestep/dkl_bcs_seed?_ep*.json")):
        m = re.search(r"(dkl_bcs_seed\d)_ep(\d+)", os.path.basename(f))
        out.append((m.group(1), int(m.group(2)), f))
    return out


def main():
    fig = plt.figure(figsize=(S.FULL, 2.95))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 0.85, 1.25], wspace=0.45, left=0.065, right=0.99, top=0.86, bottom=0.17)
    model, control = R.runs(R.MODEL), R.runs(R.CONTROL)

    # ------------------------------------------------------------ (a) the sweep
    ax = fig.add_subplot(gs[0, 0])
    w, M = sweeps([f"{r}_final" for r in model]); _, Ct = sweeps([f"{r}_final" for r in control])
    _, Sw = sweeps([f"{r}_final" for r in R.runs(R.SWAP)])   # the ramp on L−M: flat like the control
    _, B = sweeps([f"{r}_best" for r in R.runs(R.BA)])
    H = humans()
    S.refline(ax, y=0.5)
    ax.axhspan(H.min(), H.max(), color="#EBEBEB", lw=0, zorder=0)
    ax.axhline(H.mean(), color=S.HUMAN, lw=0.9, zorder=2)
    ax.text(0.01, H.mean() + 0.012, f"human observers  {H.mean():.3f}  (n = 10, range {H.min():.2f}–{H.max():.2f})",
            fontsize=6.0, va="bottom", color=S.HUMAN)
    for rows, c, lw, lab in [(B, S.BA, 1.3, f"BA network ({R.fmt_seeds(len(B))}, best ckpts)"),
                             (Sw, S.SWAP, 1.3, f"channel swap ({R.fmt_seeds(len(Sw))})"),
                             (Ct, S.CONTROL, 1.8, f"control ({R.fmt_seeds(len(Ct))})"),
                             (M, S.MODEL, 2.4, f"model ({R.fmt_seeds(len(M))})")]:
        for r in rows:
            ax.plot(w, r, "-", color=c, lw=0.6, alpha=0.45, zorder=3)
        ax.plot(w, rows.mean(0), "-", color=c, lw=lw, zorder=4, label=lab)
    for name, v in PUBLISHED:
        ax.plot([0.90], [v], "D", color=S.PUB, ms=3, mew=0, zorder=4)
        ax.text(0.88, v, name, fontsize=5.6, color=S.PUB, va="center", ha="right")
    ax.text(0.99, 0.06, "published CNNs (Geirhos et al. 2018a)", fontsize=5.2, color=S.PUB, va="bottom", ha="right")
    ax.axvline(0.5, color="#B8B8B8", lw=0.7, ls=(0, (2, 2)), zorder=1)
    ax.text(0.515, 0.245, "w = 0.5\n(the trained model)", fontsize=5.6, color=S.MUTED, va="center", ha="left")
    ax.set_xlim(0, 1.0); ax.set_ylim(0.04, 1.02)
    ax.set_xlabel("w — weight on the magno stream in the readout")
    ax.set_ylabel("cue-conflict shape bias")
    ax.text(-0.005, 0.505, "parity", fontsize=5.8, color=S.MUTED, va="bottom", ha="left")
    lo, hi = M.mean(0)[0], M.mean(0)[-1]
    ax.annotate("", (0.985, hi), (0.985, lo), arrowprops=dict(arrowstyle="<->", color=S.INK, lw=0.8, shrinkA=0, shrinkB=0))
    ax.text(0.965, 0.665, f"{hi - lo:.2f} spanned by\nre-weighting alone", fontsize=6.0, va="center", ha="right")
    ax.legend(loc="upper left", fontsize=5.8, handlelength=1.6, bbox_to_anchor=(0.0, 0.90))
    i05 = int(np.argmin(np.abs(w - 0.5)))
    print(f"  (a) model {lo:.3f} -> {hi:.3f} (w=0.5 {M.mean(0)[i05]:.3f}); control {Ct.mean(0).min():.3f}..{Ct.mean(0).max():.3f}; "
          f"swap {Sw.mean(0).min():.3f}..{Sw.mean(0).max():.3f}; "
          f"BA best {B.mean(0)[0]:.3f} -> {B.mean(0)[-1]:.3f}; humans {H.mean():.3f} [{H.min():.3f},{H.max():.3f}]")
    S.letter(ax, "a", dx=-0.05, dy=0.03, title="One network, frozen streams, one scalar")

    # ------------------------------------------------------------ (b) refitted heads
    ax = fig.add_subplot(gs[0, 1])
    NL = json.load(open("results/nonlinear_readout_summary.json"))
    heads = [("affine", "frozen\naffine"), ("linear", "refitted\nlinear"), ("mlp", "refitted\nnonlinear")]
    for i, (k, lab) in enumerate(heads):
        spans = np.array([NL[f"{r}_final"][k]["span"] for r in model])
        S.seeds(ax, i, spans, S.MODEL, ms=3.8, mean_bar=False)
        ax.plot([i - 0.22, i + 0.22], [np.median(spans)] * 2, "-", color=S.MODEL, lw=2.0, zorder=5)  # the text's median
        ba = np.median([NL[f"{r}_best"][k]["span"] for r in R.runs(R.BA)])
        ax.plot([i + 0.30], [ba], "o", mfc="white", mec=S.BA, ms=4.2, mew=1.0, zorder=5)
        if k != "affine":
            gated = np.array([NL[f"{r}_final"][k]["gated"] for r in model])
            ax.plot([i - 0.22, i + 0.22], [np.median(gated)] * 2, color=S.MODEL, lw=1.0, ls=(0, (2, 2)), zorder=5)
            print(f"  (b) {k:7s} gated per seed {np.round(gated, 3)}  median {np.median(gated):.3f}")
        print(f"  (b) {k:7s} span per seed {np.round(spans, 3)}  median {np.median(spans):.3f}   BA median {ba:.3f}")
    ax.set_xticks(range(3)); ax.set_xticklabels([h[1] for h in heads], fontsize=6.2)
    ax.set_xlim(-0.5, 2.6); ax.set_ylim(0, 0.62)
    ax.set_ylabel("span of shape bias over w\n(max − min)")
    S.note(ax, "filled: model seeds, bar = median\ndashed: within 5 points of\npeak validation accuracy\nopen: BA network, median",
           x=0.03, y=0.03, va="bottom", size=5.4)
    S.letter(ax, "b", dx=-0.09, dy=0.03, title="The span is not the head")

    # ------------------------------------------------------------ (c) the field of models
    ax = fig.add_subplot(gs[0, 2])
    S.refline(ax, y=0.5)
    ax.axhspan(H.min(), H.max(), color="#EBEBEB", lw=0, zorder=0)
    ax.text(0.285, H.mean(), "human observers", fontsize=5.8, color=S.HUMAN, va="center", ha="left")
    nets = [("model", model, None, S.MODEL), ("control", control, None, S.CONTROL),
            ("BA", None, ba_best(), S.BA), ("BA no diet", R.runs(R.BANODIET), None, S.BANODIET)]
    for net, finals, best, c in nets:
        for cond, mk, _lab in READOUTS:
            if net == "BA no diet" and cond != "full":
                continue
            if best is not None:
                pts = [(R.history(run)[ep - 1]["val_top1"], json.load(open(f))["conditions"][cond]["shape_bias"]) for run, ep, f in best]
            else:
                pts = [(R.final_top1(r), R.cc(r, cond)) for r in finals]
            xs, ys = np.array(pts).T
            ax.plot(xs, ys, mk, color=c, ms=4.4, mec="white", mew=0.6, alpha=0.95, zorder=4)
            print(f"  (c) {net:10s} {cond:10s} acc {xs.mean():.3f} [{xs.min():.3f},{xs.max():.3f}]  shape bias {ys.mean():.3f} [{ys.min():.3f},{ys.max():.3f}]")
    for name, v in PUBLISHED:
        ax.plot([0.29], [v], "D", color=S.PUB, ms=2.6, mew=0, zorder=3)
    ax.text(0.283, 0.515, "published CNNs\n(ImageNet; accuracy\nnot comparable)", fontsize=5.0, color=S.PUB, va="bottom", ha="left")
    ax.set_xlim(0.27, 0.80); ax.set_ylim(0.1, 1.02)
    ax.set_xlabel("validation top-1 at the scored checkpoint")
    ax.set_ylabel("cue-conflict shape bias")
    h = [plt.Line2D([], [], color=c, marker="o", lw=0, ms=4, label=k) for k, c in
         (("model", S.MODEL), ("control", S.CONTROL), ("BA", S.BA), ("BA no diet", S.BANODIET))]
    h += [plt.Line2D([], [], color=S.MUTED, marker=mk, lw=0, ms=4, label=lab) for _c, mk, lab in READOUTS]
    ax.legend(handles=h, loc="lower right", bbox_to_anchor=(1.0, 0.0), fontsize=4.9, handlelength=1.0, labelspacing=0.2,
              borderpad=0.2, handletextpad=0.3, ncol=2, columnspacing=0.6)
    S.letter(ax, "c", dx=-0.07, dy=0.03, title="Where every readout sits")

    S.save(fig, "fig4_readout")


if __name__ == "__main__":
    main()
