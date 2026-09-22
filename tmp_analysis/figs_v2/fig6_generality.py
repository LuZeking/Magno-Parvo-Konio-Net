"""Figure 6 — the readout account beyond the two-stream architecture (subspace rule; PLAN_SUBSPACE_2026-09-15).

Inputs: results/figures_v2/subspace_behaviour.json   (tmp_analysis/figs_v2/subspace_behaviour.py; trained rates, accuracy
                                                     gate and top-1 along the sweep, exemplar-shared curves and spans)
        results/figures_v2/exemplar_zoo.json         (exemplar-held-out curves of the six ImageNet networks; PLAN_EXEMPLAR)
        results/figures_v2/exemplar_trunk_curves.json (exemplar-held-out curves of our trunks)
        results/figures_v2/subspace_cortex.json      (tmp_analysis/figs_v2/generality_cortex.py --run-id 2026-09-15_sub; optional)
        results/figures_v2/generality_cortex.json    (unit-split cortex; used for panels c-d until the subspace cortex fits exist)

(a) For each frozen network, the cue-conflict shape bias reachable by one gain on its own shape- and
    texture-predictive subspaces with its own head, the subspaces fitted on images sharing no object or texture
    exemplar with those scored (E1, docs/PLAN_EXEMPLAR_2026-09-21.md): thin bar, w = 0 → 1; thick segment, the
    part of the sweep within 5 points of the trained top-1; dot, the trained readout. Grey bracket below: the same
    gated range under the pre-registered estimator, whose halves share exemplars (E0).
(b) Held-out rate against w for the six ImageNet networks, with the change of top-1 along the sweep beneath.
(c) Cortex: the S − T index across the hierarchy, with the two-stream diet model's magno − parvo profile for reference.
(d) Cortex-optimal mixing weight of the lossy sum per region.
Usage: PYTHONPATH=.:tmp_analysis/figs_v2 python tmp_analysis/figs_v2/fig6_generality.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import matplotlib.pyplot as plt

import style as S

ORDER = [("concat", "two-stream, diet (streams concatenated)", S.DIET),
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
REGIONS = ["V1", "V2", "V3", "hV4", "ventral", "lateral", "dorsal"]
FAM_STYLE = {"two_stream_diet": ("two-stream, diet (magno − parvo)", S.DIET, "-", "o"),
             "single34": ("single ResNet-34, diet", S.DIETL, "-", "s"),
             "rgb_tsn": ("single ResNet-18, no diet", S.NODIET, "-", "^"),
             "resnet50": ("ImageNet ResNet-50", S.PUB, "-", "D"),
             "vit_b_16": ("ImageNet ViT-B/16", "#000000", "--", "v"),
             "convnext_base": ("ImageNet ConvNeXt-B", "#009E73", "--", "P")}
SHARED_COL = "#9A9A9A"
TRUNKS = ("concat", "single34", "single18", "rgb_tsn")
WS_GRID = np.round(np.linspace(0, 1, 21), 10)


def seed_mean(f, key="mean_curve"):
    return np.mean([s[key] for s in f["seeds"]], 0)


def held_out():
    """E1 mean curve (seed mean for our trunks) and the number printed beside it: the gated span of that curve,
    or, for the ungated two-stream concatenation, the mean over seeds of the per-repetition span, as for E0."""
    zoo = {n["tag"]: n for n in json.load(open("results/figures_v2/exemplar_zoo.json"))["networks"]}
    tr = json.load(open("results/figures_v2/exemplar_trunk_curves.json"))["families"]
    out = {}
    for key in TRUNKS:
        out[key] = dict(curve=np.mean([x["E1"]["mean_curve"] for x in tr[key]], 0),
                        span=float(np.mean([x["E1"]["span_mean"] for x in tr[key]])))
    for key in IMAGENET:
        out[key] = dict(curve=np.array(zoo[f"tv_{key}"]["spans"]["E1"]["mean_curve"]), span=None)
    return out


def gated(curve, gate):
    if gate is None:
        return None
    m = (WS_GRID >= gate[0] - 1e-9) & (WS_GRID <= gate[1] + 1e-9)
    return float(curve[m].min()), float(curve[m].max())


def load_ctx():
    """The pre-registered unit-split cortex fits (PLAN_GENERALITY P3-P5, met); the subspace-band fits
    (PLAN_SUBSPACE P-C1-P-C3, failed except P-C2) are reported in ED Table 15, not drawn here. Set
    FIG6_CORTEX=subspace to draw them instead."""
    order = (("results/figures_v2/generality_cortex.json", "unit"), ("results/figures_v2/subspace_cortex.json", "subspace"))
    if os.environ.get("FIG6_CORTEX") == "subspace":
        order = order[::-1]
    for path, mode in order:
        if os.path.exists(path):
            ctx = json.load(open(path))
            if ctx.get("families"):
                return ctx, mode
    return None, None


def main():
    sub = json.load(open("results/figures_v2/subspace_behaviour.json"))
    fam = {f["key"]: f for f in sub["families"]}
    ws = np.array(fam["resnet50"]["seeds"][0]["ws"])
    ctx, ctx_mode = load_ctx()
    ho = held_out()
    assert np.allclose(ws, WS_GRID)

    fig = plt.figure(figsize=(S.FULL, 6.0))
    ax_a = fig.add_axes([0.26, 0.585, 0.36, 0.355])
    ax_b = fig.add_axes([0.775, 0.685, 0.15, 0.255])
    ax_b2 = fig.add_axes([0.775, 0.585, 0.15, 0.085], sharex=ax_b)
    ax_c = fig.add_axes([0.08, 0.075, 0.38, 0.35])
    ax_d = fig.add_axes([0.60, 0.075, 0.38, 0.35])

    # ---------------- (a) reachable span per network
    ys = np.arange(len(ORDER))[::-1] * 1.0
    ax_a.axvspan(0.919, 0.987, color="#EFEFEF", zorder=0)
    ax_a.text(0.953, 4.6, "human observers", ha="center", va="center", fontsize=6.0, color=S.MUTED, rotation=90)
    S.refline(ax_a, x=0.5)
    for y, (key, label, col) in zip(ys, ORDER):
        if key not in fam:
            continue
        f, h = fam[key], ho[key]
        curve, gate = h["curve"], f.get("gate")
        ax_a.plot([curve.min(), curve.max()], [y, y], "-", color=col, lw=2.2, alpha=0.45, solid_capstyle="butt", zorder=2)
        g1 = gated(curve, gate)
        if g1 is not None:
            ax_a.plot(list(g1), [y, y], "-", color=col, lw=4.0, solid_capstyle="butt", zorder=3)
        ax_a.plot(f["trained"], y, "o", color=col, ms=5.5, mec="white", mew=0.9, zorder=5)
        ax_a.plot(curve[-1], y, ">", color=col, ms=4.2, mec="white", mew=0.5, zorder=4)
        ax_a.plot(curve[0], y, "<", color=col, ms=4.2, mec="white", mew=0.5, zorder=4)
        # the pre-registered estimator, halves sharing exemplars, within the same gate (grey bracket beneath)
        c0 = seed_mean(f)
        g0 = gated(c0, gate) or (float(c0.min()), float(c0.max()))
        ax_a.plot(list(g0), [y - 0.34, y - 0.34], "-", color=SHARED_COL, lw=1.6, solid_capstyle="butt", zorder=2)
        ax_a.text(-0.02, y, label, ha="right", va="center", fontsize=6.8, transform=ax_a.get_yaxis_transform(), color=S.INK)
        ho_span = g1[1] - g1[0] if g1 is not None else h["span"]
        sh_span = f["gated_span"] if f.get("gated_span") is not None else f["span"]
        ax_a.text(1.01, y, f"{ho_span:.2f}", ha="left", va="center", fontsize=6.0, color=col if col != S.PUB else S.INK, transform=ax_a.get_yaxis_transform())
        ax_a.text(1.11, y, f"{sh_span:.2f}", ha="left", va="center", fontsize=6.0, color=SHARED_COL, transform=ax_a.get_yaxis_transform())
        print(f"  (a) {key:15s} held out {ho_span:.3f} ({'gated' if g1 else 'ungated, seed-mean span'}) | shared {sh_span:.3f} | trained {f['trained']:.3f}")
    ax_a.set_xlim(0.0, 1.0); ax_a.set_ylim(-2.35, len(ORDER) - 0.35)
    ax_a.set_yticks([]); ax_a.spines["left"].set_visible(False)
    ax_a.set_xlabel("cue-conflict shape bias (exemplars held out)")
    ax_a.text(1.01, len(ORDER) - 0.4, "held\nout", ha="left", va="bottom", fontsize=6.0, color=S.INK, transform=ax_a.get_yaxis_transform(), linespacing=0.95)
    ax_a.text(1.11, len(ORDER) - 0.4, "shared", ha="left", va="bottom", fontsize=6.0, color=SHARED_COL, transform=ax_a.get_yaxis_transform())
    ax_a.plot([], [], "o", color=S.MUTED, label="trained readout (w = 0.5)")
    ax_a.plot([], [], "-", color=S.MUTED, lw=2.2, alpha=0.45, label="reachable, w = 0 → 1, head fixed")
    ax_a.plot([], [], "-", color=S.MUTED, lw=4, label="within 5 points of trained top-1")
    ax_a.plot([], [], "-", color=SHARED_COL, lw=1.6, label="the same, exemplars shared (pre-registered)")
    ax_a.legend(loc="lower center", bbox_to_anchor=(0.47, -0.01), ncol=2, fontsize=5.8, handlelength=1.6,
                labelspacing=0.25, columnspacing=1.0, frameon=False)
    S.letter(ax_a, "a", dx=-0.24, title="one gain on a network's own shape-predictive subspace moves its rate")

    # ---------------- (b) ImageNet curves and the accuracy budget
    S.refline(ax_b, x=0.5)
    ends = []
    for key in IMAGENET:
        if key not in fam:
            continue
        f = fam[key]
        curve = ho[key]["curve"]
        ax_b.plot(ws, curve, "-", color=IN_COL[key], lw=1.3)
        ax_b.plot(0.5, f["trained"], "o", color=IN_COL[key], ms=4, mec="white", mew=0.6, zorder=5)
        ends.append((float(curve[-1]), ORDER[[k for k, _, _ in ORDER].index(key)][1].replace("ImageNet ", ""), IN_COL[key]))
        acc = [s["gain_accuracy"] for s in f["seeds"] if s.get("gain_accuracy")]
        if acc:
            d = np.mean([(np.array(a["top1"]) - a["trained_top1"]) * 100 for a in acc], 0)
            ax_b2.plot(ws, d, "-", color=IN_COL[key], lw=1.0)
    ends.sort()
    ypos = [e[0] for e in ends]
    for i in range(1, len(ypos)):
        ypos[i] = max(ypos[i], ypos[i - 1] + 0.03)
    for (yv, lab, col), yp in zip(ends, ypos):
        ax_b.text(1.03, yp, lab, ha="left", va="center", fontsize=5.9, color=col, clip_on=False)
    ax_b.set_ylabel("cue-conflict shape bias\n(exemplars held out)", fontsize=6.5)
    ax_b.set_xlim(0, 1); ax_b.set_ylim(0.0, 0.6); ax_b.set_yticks([0, 0.2, 0.4, 0.6])
    plt.setp(ax_b.get_xticklabels(), visible=False)
    ax_b2.axhspan(-5, 5, color="#F4F4F4", zorder=0)
    S.refline(ax_b2, y=0)
    ax_b2.set_ylim(-6, 2); ax_b2.set_yticks([-5, 0])
    ax_b2.set_xticks([0, 0.5, 1]); ax_b2.set_xlabel("gain weight w on the shape subspace")
    ax_b2.set_ylabel("Δ top-1\n(points)", fontsize=6.5)
    ax_b2.text(0.02, -4.6, "5-point budget", fontsize=5.6, color=S.MUTED, va="bottom")
    S.letter(ax_b, "b", dx=-0.09, title="the ranking is not readout-invariant")

    # ---------------- (c) cortex index profile
    x = np.arange(len(REGIONS))
    S.refline(ax_c, y=0)
    if ctx:
        for famkey, (label, col, ls, mk) in FAM_STYLE.items():
            if famkey not in ctx["families"]:
                continue
            fo = ctx["families"][famkey]
            m = np.array([np.nanmean(fo["index"][r]) for r in REGIONS])
            se = np.array([np.nanstd(fo["index"][r], ddof=1) / np.sqrt(np.sum(np.isfinite(fo["index"][r]))) for r in REGIONS])
            ax_c.errorbar(x, m, yerr=se, fmt=ls + mk, color=col, ms=4, lw=1.2, capsize=0, mec="white", mew=0.6, label=label, zorder=4)
        if ctx_mode == "unit" and "two_stream_diet" not in ctx["families"]:
            pass
        what = "shape- and texture-predictive subspaces" if ctx_mode == "subspace" else "shape- and texture-leaning unit halves"
        leg = ax_c.legend(loc="upper right", fontsize=5.9, handlelength=1.8, labelspacing=0.25,
                          title=f"bands: {what}", title_fontsize=6.0, alignment="left")
        leg.get_title().set_color(S.MUTED)
    else:
        S.note(ax_c, "cortex results pending", x=0.5, y=0.5, ha="center", va="center")
    ax_c.set_xticks(x); ax_c.set_xticklabels(REGIONS); ax_c.set_xlim(-0.5, 6.5)
    ax_c.set_ylabel("component-preference index\n(U_S − U_T) / (U_S + U_T)")
    ax_c.set_ylim(-1.05, 1.05)
    ax_c.axvspan(3.5, 6.5, color="#F4F4F4", zorder=0)
    ax_c.text(5.0, -1.0, "extrastriate (Kastner2015 groups)", ha="center", va="bottom", fontsize=6.2, color=S.MUTED)
    S.letter(ax_c, "c", dx=-0.07, title="which component carries cortex, with and without a stream split")

    # ---------------- (d) cortex-optimal w
    S.refline(ax_d, y=0.5)
    if ctx:
        fams = [f for f in ("rgb_tsn", "single34", "resnet50", "vit_b_16", "convnext_base") if f in ctx["families"] and ctx["families"][f].get("w_star")]
        off = np.linspace(-0.3, 0.3, max(len(fams), 1)) if len(fams) > 1 else [0.0]
        for j, famkey in enumerate(fams):
            label, col, ls, mk = FAM_STYLE[famkey]
            fo = ctx["families"][famkey]
            for i, r in enumerate(REGIONS):
                v = np.array(fo["w_star"].get(r, []), float); v = v[np.isfinite(v)]
                if len(v) == 0:
                    continue
                ax_d.plot(np.full(len(v), i + off[j]), v, mk, color=col, ms=2.6, alpha=0.55, mec="none", zorder=3)
                ax_d.plot([i + off[j] - 0.08, i + off[j] + 0.08], [np.median(v)] * 2, "-", color=col, lw=2.0, zorder=5)
        S.note(ax_d, "symbols as in c", x=0.98, y=0.03, ha="right", va="bottom")
    else:
        S.note(ax_d, "cortex results pending", x=0.5, y=0.5, ha="center", va="center")
    ax_d.set_xticks(x); ax_d.set_xticklabels(REGIONS); ax_d.set_xlim(-0.5, 6.5)
    ax_d.set_ylabel("cortex-optimal w\n(weight on the shape component)")
    ax_d.set_ylim(-0.02, 1.02)
    ax_d.axvspan(3.5, 6.5, color="#F4F4F4", zorder=0)
    S.letter(ax_d, "d", dx=-0.07, title="the mixture each region prefers")

    S.save(fig, "fig6_generality")


if __name__ == "__main__":
    main()
