"""Figure 4 — behaviour reads out from both streams, and the shape-bias rate is a
property of that readout.

  (a) cue-conflict shape bias against the readout weight w on the magno stream
      (full = classifier(w·y₁ + (1−w)·y₂)), 21 values, 3 seeds, both checkpoint
      policies. Representations never change along the x-axis. The ten human
      observers and the three published CNNs are drawn on the same y-axis.
  (b) does the span survive a readout that is refitted, and one that is not
      affine? Spans of shape bias over w for the frozen affine head, a refitted
      linear head and a two-layer nonlinear head, per seed (best checkpoints).
  (c) where every model sits: shape bias against validation accuracy at the
      scored checkpoint, for the two-stream arms, the single-stream controls
      (half- and parameter-matched capacity) and the diet's two streams alone.

Sources: results/fusion_sweep/, results/nonlinear_readout_summary.json,
results/cue_conflict*/, /scratch/byrong/VST_data/mpk_runs/*/history.json, human
observers via tmp_analysis/human_shapebias.py.
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

RUNS = "/scratch/byrong/VST_data/mpk_runs"
PUBLISHED = [("AlexNet", 0.429), ("GoogLeNet", 0.312), ("VGG-16", 0.172)]   # Geirhos et al. 2018a release


def load_sweep(policy):
    rows, ws = [], None
    for f in sorted(glob.glob(f"results/fusion_sweep/*_{policy}_wsweep.json")):
        pts = sorted(json.load(open(f))["sweep"], key=lambda r: r["w"])
        ws = np.array([r["w"] for r in pts]); rows.append([r["shape_bias"] for r in pts])
    return ws, np.array(rows)


def humans():
    import sys
    sys.path.insert(0, "tmp_analysis")
    with contextlib.redirect_stdout(io.StringIO()):
        from human_shapebias import human_observers
        return np.array([o["shape_bias"] for o in human_observers()])


def cc(pattern, cond="full"):
    out = []
    for f in sorted(glob.glob(pattern)):
        if not f.endswith(".json"):
            continue
        d = json.load(open(f))
        if cond in d["conditions"]:
            ep = re.search(r"_ep(\d+)", os.path.basename(f))
            out.append((os.path.basename(f), int(ep.group(1)) if ep else None, d["conditions"][cond]["shape_bias"]))
    return out


def acc_at(run, epoch):
    h = json.load(open(f"{RUNS}/{run}/history.json"))
    v = [e["val_top1"] for e in h]
    return v[epoch - 1] if epoch else v[-1]


def main():
    fig = plt.figure(figsize=(S.FULL, 2.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 0.85, 1.25], wspace=0.45, left=0.065, right=0.99, top=0.86, bottom=0.17)

    # ------------------------------------------------------------ (a) the sweep
    ax = fig.add_subplot(gs[0, 0])
    w, best = load_sweep("best"); _, final = load_sweep("final")
    H = humans()
    S.refline(ax, y=0.5)
    ax.axhspan(H.min(), H.max(), color="#EBEBEB", lw=0, zorder=0)
    ax.axhline(H.mean(), color=S.HUMAN, lw=0.9, zorder=2)
    ax.text(0.01, H.mean() + 0.012, f"human observers  {H.mean():.3f}  (n = 10, range {H.min():.2f}–{H.max():.2f})",
            fontsize=6.0, va="bottom", color=S.HUMAN)
    for rows, c, lab in [(final, S.DIETL, "final checkpoints"), (best, S.DIET, "best saved checkpoints")]:
        for r in rows:
            ax.plot(w, r, "-", color=c, lw=0.7, alpha=0.5, zorder=3)
        ax.plot(w, rows.mean(0), "-", color=c, lw=2.2, zorder=4, label=lab)
    for name, v in PUBLISHED:
        ax.plot([0.90], [v], "D", color=S.PUB, ms=3, mew=0, zorder=4)
        ax.text(0.88, v, name, fontsize=5.6, color=S.PUB, va="center", ha="right")
    ax.text(0.62, 0.115, "published CNNs\n(Geirhos et al. 2018a)", fontsize=5.2, color=S.PUB, va="bottom", ha="center")
    ax.axvline(0.5, color="#B8B8B8", lw=0.7, ls=(0, (2, 2)), zorder=1)
    ax.text(0.49, 0.865, "w = 0.5\n(the trained model)", fontsize=5.6, color=S.MUTED, va="top", ha="right")
    ax.set_xlim(0, 1.0); ax.set_ylim(0.1, 1.02)
    ax.set_xlabel("w — weight on the magno stream in the readout")
    ax.set_ylabel("cue-conflict shape bias")
    ax.text(-0.005, 0.505, "parity", fontsize=5.8, color=S.MUTED, va="bottom", ha="left")
    lo, hi = best.mean(0)[0], best.mean(0)[-1]
    ax.annotate("", (0.985, hi), (0.985, lo), arrowprops=dict(arrowstyle="<->", color=S.INK, lw=0.8))
    ax.text(0.965, 0.60, f"{hi - lo:.2f} spanned by\nre-weighting alone", fontsize=6.0, va="center", ha="right")
    ax.legend(loc="upper left", fontsize=6.0, handlelength=1.6, bbox_to_anchor=(0.0, 0.88))
    print(f"  (a) best {lo:.3f} -> {hi:.3f}; final {final.mean(0)[0]:.3f} -> {final.mean(0)[-1]:.3f}; humans {H.mean():.3f} [{H.min():.3f},{H.max():.3f}]")
    S.letter(ax, "a", dx=-0.05, dy=0.03, title="One network, frozen streams, one scalar")

    # ------------------------------------------------------------ (b) refitted heads
    ax = fig.add_subplot(gs[0, 1])
    NL = json.load(open("results/nonlinear_readout_summary.json"))
    heads = [("affine", "frozen\naffine", S.DIET), ("linear", "refitted\nlinear", "#5A9BD5"), ("mlp", "refitted\nnonlinear", "#9C6ADE")]
    for i, (k, lab, c) in enumerate(heads):
        spans = np.array([NL[f"dkl_bcs_seed{s}_best"][k]["span"] for s in range(3)])
        S.seeds(ax, i, spans, c, ms=3.8, bar_w=0.22)
        if k != "affine":
            gated = np.array([NL[f"dkl_bcs_seed{s}_best"][k]["gated"] for s in range(3)])
            ax.plot([i - 0.22, i + 0.22], [gated.mean()] * 2, "-", color=c, lw=1.0, ls=(0, (2, 2)), zorder=5)
        print(f"  (b) {k:7s} span per seed {np.round(spans, 3)}  median {np.median(spans):.3f}")
    ax.set_xticks(range(3)); ax.set_xticklabels([h[1] for h in heads], fontsize=6.2)
    ax.set_xlim(-0.5, 2.5); ax.set_ylim(0, 0.62)
    ax.set_ylabel("span of shape bias over w\n(max − min, 3 seeds)")
    S.note(ax, "dashed: span within 5 points\nof peak validation accuracy", x=0.03, y=0.14, va="bottom", size=5.6)
    S.letter(ax, "b", dx=-0.09, dy=0.03, title="The span is not the head")

    # ------------------------------------------------------------ (c) the field of models
    ax = fig.add_subplot(gs[0, 2])
    S.refline(ax, y=0.5)
    ax.axhspan(H.min(), H.max(), color="#EBEBEB", lw=0, zorder=0)
    ax.text(0.755, H.mean(), "human observers", fontsize=5.8, color=S.HUMAN, va="center", ha="right")
    groups = [
        ("diet, magno stream alone", "results/cue_conflict_bestep/dkl_bcs_seed?_ep*.json", "magno_only", "dkl_bcs_seed{}", S.DIET, "^"),
        ("diet, both streams", "results/cue_conflict_bestep/dkl_bcs_seed?_ep*.json", "full", "dkl_bcs_seed{}", S.DIET, "o"),
        ("diet, parvo stream alone", "results/cue_conflict_bestep/dkl_bcs_seed?_ep*.json", "parvo_only", "dkl_bcs_seed{}", S.DIET, "v"),
        ("single stream, ResNet-34 (matched)", "results/cue_conflict_single34_bestep/*.json", "full", "dkl_bcs_single34_seed{}", S.SINGLE, "s"),
        ("single stream, ResNet-18 (half)", "results/cue_conflict_single_bestep/*.json", "full", "dkl_bcs_single_seed{}", S.SINGLE, "D"),
        ("no diet", "results/cue_conflict/dkl_base_seed?.json", "full", "dkl_base_seed{}", S.NODIET, "o"),
        ("swap", "results/cue_conflict/dkl_bcs_swap_seed?.json", "full", "dkl_bcs_swap_seed{}", S.SWAP, "o"),
        ("parvo_highsf", "results/cue_conflict/dkl_bcs_parvo_highsf_seed?.json", "full", "dkl_bcs_parvo_highsf_seed{}", S.HIGHSF, "o"),
    ]
    for lab, pat, cond, runtmpl, c, mk in groups:
        pts = cc(pat, cond)
        xs, ys = [], []
        for fname, ep, sb in pts:
            seed = int(re.search(r"seed(\d)", fname).group(1))
            xs.append(acc_at(runtmpl.format(seed), ep)); ys.append(sb)
        xs, ys = np.array(xs), np.array(ys)
        ax.plot(xs, ys, mk, color=c, ms=4.2 if mk != "o" else 4.6, mec="white", mew=0.6, alpha=0.9, zorder=4, label=lab)
        print(f"  (c) {lab:36s} acc {xs.mean():.3f} [{xs.min():.3f},{xs.max():.3f}]  shape bias {ys.mean():.3f} [{ys.min():.3f},{ys.max():.3f}]")
    for name, v in PUBLISHED:
        ax.plot([0.29], [v], "D", color=S.PUB, ms=2.6, mew=0, zorder=3)
    ax.text(0.305, 0.245, "published CNNs (ImageNet;\naccuracy not comparable)", fontsize=5.0, color=S.PUB, va="center", ha="left")
    ax.set_xlim(0.27, 0.76); ax.set_ylim(0.1, 1.02)
    ax.set_xlabel("validation accuracy at the scored checkpoint")
    ax.set_ylabel("cue-conflict shape bias")
    ax.legend(loc="upper left", bbox_to_anchor=(0.05, 0.93), fontsize=5.0, handlelength=1.0, labelspacing=0.22, borderpad=0.2, markerscale=0.85, handletextpad=0.4)
    S.letter(ax, "c", dx=-0.07, dy=0.03, title="Where every model sits")

    S.save(fig, "fig4_readout")


if __name__ == "__main__":
    main()
