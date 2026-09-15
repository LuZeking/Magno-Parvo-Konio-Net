"""Figure 6 — the readout account beyond the two-stream architecture.

Inputs: results/figures_v2/generality_behaviour.json (tmp_analysis/figs_v2/generality_behaviour.py; includes the accuracy gate),
        results/figures_v2/generality_cortex.json (tmp_analysis/figs_v2/generality_cortex.py; optional).

(a) For each frozen network, the cue-conflict shape bias reachable by one gain on its own
    shape-leaning units with its own head (bar, w = 0 → 1), the trained readout (dot), and the
    part of the sweep within 5 points of the trained top-1 (thick segment).
(b) Rate against w for the six ImageNet networks: the field's ordering is not readout-invariant.
(c) Cortex: the shape-leaning − texture-leaning index across the hierarchy for the single trunks,
    with the two-stream diet model's magno − parvo profile for reference.
(d) Cortex-optimal mixing weight of the lossy sum per region, single trunks.
Usage: PYTHONPATH=.:tmp_analysis/figs_v2 python tmp_analysis/figs_v2/fig6_generality.py
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import matplotlib.pyplot as plt

import style as S

RUN = os.environ.get("GEN_RUNID", "2026-09-14_gen")
BEH_DIR = f"/scratch/byrong/mpk/generality/behaviour/{RUN}"
ORDER = [("concat", "two-stream, diet (streams as halves)", S.DIET),
         ("single34", "single ResNet-34, diet", S.DIETL),
         ("single18", "single ResNet-18, diet", S.DIETL),
         ("rgb_tsn", "single ResNet-18, no diet", S.NODIET),
         ("vit_b_16", "ImageNet ViT-B/16", S.PUB),
         ("resnet152", "ImageNet ResNet-152", S.PUB),
         ("convnext_base", "ImageNet ConvNeXt-B", S.PUB),
         ("convnext_small", "ImageNet ConvNeXt-S", S.PUB),
         ("resnet50", "ImageNet ResNet-50", S.PUB),
         ("vgg16", "ImageNet VGG-16", S.PUB)]
IMAGENET = ["vit_b_16", "resnet152", "convnext_base", "convnext_small", "resnet50", "vgg16"]
IN_COL = {"vit_b_16": "#000000", "resnet152": "#0072B2", "convnext_base": "#009E73", "convnext_small": "#56B4E9",
          "resnet50": "#D55E00", "vgg16": "#CC79A7"}
REG_LABEL = {"V1": "V1", "V2": "V2", "V3": "V3", "hV4": "hV4", "ventral": "ventral", "lateral": "lateral", "dorsal": "dorsal"}
FAM_STYLE = {"two_stream_diet": ("two-stream, diet (magno − parvo)", S.DIET, "-", "o"),
             "single34": ("single ResNet-34, diet", S.DIETL, "-", "s"),
             "rgb_tsn": ("single ResNet-18, no diet", S.NODIET, "-", "^"),
             "resnet50": ("ImageNet ResNet-50", S.PUB, "-", "D")}


def gain_accuracy(seeds):
    """w-range within 5 points of trained top-1, averaged over seeds: (w_lo, w_hi) or None (from the JSON)."""
    los = [min(r["gain_accuracy"]["w_within_5pts"]) for r in seeds if r.get("gain_accuracy")]
    his = [max(r["gain_accuracy"]["w_within_5pts"]) for r in seeds if r.get("gain_accuracy")]
    return (float(np.mean(los)), float(np.mean(his))) if los else None


def main():
    beh = json.load(open("results/figures_v2/generality_behaviour.json"))
    fam = {f["key"]: f for f in beh["families"]}
    ws = np.array(fam["resnet50"]["seeds"][0]["ws"])
    ctx = json.load(open("results/figures_v2/generality_cortex.json")) if os.path.exists("results/figures_v2/generality_cortex.json") else None

    fig = plt.figure(figsize=(S.FULL, 5.6))
    ax_a = fig.add_axes([0.26, 0.60, 0.36, 0.33])
    ax_b = fig.add_axes([0.745, 0.60, 0.17, 0.33])
    ax_c = fig.add_axes([0.08, 0.08, 0.38, 0.35])
    ax_d = fig.add_axes([0.60, 0.08, 0.38, 0.35])

    # ---------------- (a) reachable span per network
    ys = np.arange(len(ORDER))[::-1]
    ax_a.axvspan(0.919, 0.987, color="#EFEFEF", zorder=0)
    ax_a.text(0.953, 4.5, "human observers", ha="center", va="center", fontsize=6.0, color=S.MUTED, rotation=90)
    S.refline(ax_a, x=0.5)
    for y, (key, label, col) in zip(ys, ORDER):
        if key not in fam:
            continue
        f = fam[key]
        curve = np.mean([r["mean_curve"] for r in f["seeds"]], 0)
        lo, hi = curve.min(), curve.max()
        ax_a.plot([curve[0], curve[-1]], [y, y], "-", color=col, lw=2.2, alpha=0.45, solid_capstyle="butt", zorder=2)
        acc = gain_accuracy(f["seeds"])
        if acc is not None:
            i0, i1 = int(np.argmin(np.abs(ws - acc[0]))), int(np.argmin(np.abs(ws - acc[1])))
            ax_a.plot([curve[i0], curve[i1]], [y, y], "-", color=col, lw=4.0, solid_capstyle="butt", zorder=3)
        ax_a.plot(f["trained"], y, "o", color=col, ms=5.5, mec="white", mew=0.9, zorder=5)
        ax_a.plot(curve[-1], y, ">", color=col, ms=4.2, mec="white", mew=0.5, zorder=4)
        ax_a.plot(curve[0], y, "<", color=col, ms=4.2, mec="white", mew=0.5, zorder=4)
        ax_a.text(-0.02, y, label, ha="right", va="center", fontsize=6.8, transform=ax_a.get_yaxis_transform(), color=S.INK)
        ax_a.text(1.015, y, f"{f['span']:+.2f}", ha="left", va="center", fontsize=6.2, color=S.MUTED, transform=ax_a.get_yaxis_transform())
    ax_a.set_xlim(0.05, 1.0); ax_a.set_ylim(-1.3, len(ORDER) - 0.4)
    ax_a.set_yticks([]); ax_a.spines["left"].set_visible(False)
    ax_a.set_xlabel("cue-conflict shape bias (held-out images)")
    ax_a.text(1.015, len(ORDER) - 0.45, "span", ha="left", va="bottom", fontsize=6.2, color=S.MUTED, transform=ax_a.get_yaxis_transform())
    ax_a.plot([], [], "o", color=S.MUTED, label="trained readout (w = 0.5)")
    ax_a.plot([], [], "-", color=S.MUTED, lw=2.2, alpha=0.45, label="reachable, w = 0 → 1, head fixed")
    ax_a.plot([], [], "-", color=S.MUTED, lw=4, label="within 5 points of trained top-1")
    ax_a.legend(loc="lower left", bbox_to_anchor=(0.36, -0.02), fontsize=5.9, handlelength=1.6, labelspacing=0.3)
    S.letter(ax_a, "a", dx=-0.24, title="one gain on a network's own shape-leaning units moves its rate")

    # ---------------- (b) ImageNet curves
    S.refline(ax_b, x=0.5)
    ends = []
    for key in IMAGENET:
        if key not in fam:
            continue
        curve = np.mean([r["mean_curve"] for r in fam[key]["seeds"]], 0)
        ax_b.plot(ws, curve, "-", color=IN_COL[key], lw=1.3)
        ax_b.plot(0.5, fam[key]["trained"], "o", color=IN_COL[key], ms=4, mec="white", mew=0.6, zorder=5)
        ends.append((float(curve[-1]), ORDER[[k for k, _, _ in ORDER].index(key)][1].replace("ImageNet ", ""), IN_COL[key]))
    # right-end labels, nudged apart
    ends.sort()
    ypos = [e[0] for e in ends]
    for i in range(1, len(ypos)):
        ypos[i] = max(ypos[i], ypos[i - 1] + 0.028)
    for (yv, lab, col), yp in zip(ends, ypos):
        ax_b.text(1.03, yp, lab, ha="left", va="center", fontsize=5.9, color=col, clip_on=False)
    ax_b.set_xlabel("gain weight w on the shape-leaning half"); ax_b.set_ylabel("cue-conflict shape bias")
    ax_b.set_xlim(0, 1); ax_b.set_xticks([0, 0.5, 1]); ax_b.set_ylim(0.05, 0.5)
    S.letter(ax_b, "b", dx=-0.09, title="the ranking is not readout-invariant")

    # ---------------- (c) cortex index profile
    regions = ["V1", "V2", "V3", "hV4", "ventral", "lateral", "dorsal"]
    x = np.arange(len(regions))
    S.refline(ax_c, y=0)
    if ctx:
        for famkey, (label, col, ls, mk) in FAM_STYLE.items():
            if famkey not in ctx["families"]:
                continue
            fo = ctx["families"][famkey]
            m = np.array([np.nanmean(fo["index"][r]) for r in regions])
            se = np.array([np.nanstd(fo["index"][r], ddof=1) / np.sqrt(np.sum(np.isfinite(fo["index"][r]))) for r in regions])
            ax_c.errorbar(x, m, yerr=se, fmt=ls + mk, color=col, ms=4, lw=1.2, capsize=0, mec="white", mew=0.6, label=label, zorder=4)
        ax_c.legend(loc="upper right", fontsize=6.2, handlelength=1.6)
    else:
        S.note(ax_c, "cortex results pending", x=0.5, y=0.5, ha="center", va="center")
    ax_c.set_xticks(x); ax_c.set_xticklabels([REG_LABEL[r] for r in regions]); ax_c.set_xlim(-0.5, 6.5)
    ax_c.set_ylabel("stream-preference index\n(U_S − U_T) / (U_S + U_T)")
    ax_c.set_ylim(-1.05, 1.05)
    ax_c.axvspan(3.5, 6.5, color="#F4F4F4", zorder=0)
    ax_c.text(5.0, -1.0, "extrastriate (Kastner2015 groups)", ha="center", va="bottom", fontsize=6.2, color=S.MUTED)
    S.letter(ax_c, "c", dx=-0.07, title="which half carries cortex, with and without a stream split")

    # ---------------- (d) cortex-optimal w
    S.refline(ax_d, y=0.5)
    if ctx:
        k = 0
        fams = [f for f in ("rgb_tsn", "single34", "resnet50") if f in ctx["families"] and ctx["families"][f].get("w_star")]
        off = np.linspace(-0.22, 0.22, max(len(fams), 1))
        for j, famkey in enumerate(fams):
            label, col, ls, mk = FAM_STYLE[famkey]
            fo = ctx["families"][famkey]
            for i, r in enumerate(regions):
                v = np.array(fo["w_star"].get(r, []), float); v = v[np.isfinite(v)]
                if len(v) == 0:
                    continue
                ax_d.plot(np.full(len(v), i + off[j]), v, mk, color=col, ms=2.8, alpha=0.55, mec="none", zorder=3)
                ax_d.plot([i + off[j] - 0.1, i + off[j] + 0.1], [np.median(v)] * 2, "-", color=col, lw=2.0, zorder=5)
        S.note(ax_d, "symbols as in c", x=0.98, y=0.03, ha="right", va="bottom")
    else:
        S.note(ax_d, "cortex results pending", x=0.5, y=0.5, ha="center", va="center")
    ax_d.set_xticks(x); ax_d.set_xticklabels([REG_LABEL[r] for r in regions]); ax_d.set_xlim(-0.5, 6.5)
    ax_d.set_ylabel("cortex-optimal w\n(weight on the shape-leaning half)")
    ax_d.set_ylim(-0.02, 1.02)
    ax_d.axvspan(3.5, 6.5, color="#F4F4F4", zorder=0)
    S.letter(ax_d, "d", dx=-0.07, title="the mixture each region prefers")

    S.save(fig, "fig6_generality")


if __name__ == "__main__":
    main()
