"""Extended Data Fig. 6: the readout account across the model zoo (PLAN_ZOO).

  MPLCONFIGDIR=... PYTHONPATH=.:tmp_analysis/figs_v2 python tmp_analysis/figs_v2/figE6_zoo.py

a  every network sorted by its trained rate: the rates reachable within the accuracy gate
   (texture end to shape end) as a bar, the trained rate as a dot; the ImageNet-trained
   networks, the recipe variant and the three Stylized-ImageNet controls in three colours.
b  gated span against trained rate (convolutional vs attention-based; controls marked).
c  probe-based representational bias against trained rate.
Reads results/figures_v2/zoo_behaviour.json; writes results/figures_v2/figE6_zoo.{pdf,png}.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import style as S  # noqa: E402

COL = {"imagenet": S.PUB, "recipe": S.MATCH, "sin": S.HIGHSF}
MARK = {"conv": "o", "attention": "^"}
HUMAN = (0.92, 0.99)


def main():
    zoo = json.load(open("results/figures_v2/zoo_behaviour.json"))
    nets = sorted(zoo["networks"], key=lambda n: n["trained"])
    st = zoo.get("ranking", {})
    n = len(nets)

    fig = plt.figure(figsize=(S.FULL, max(5.0, 0.105 * n + 1.9)))
    ax_a = fig.add_axes([0.20, 0.14, 0.40, 0.82])
    ax_b = fig.add_axes([0.72, 0.60, 0.26, 0.36])
    ax_c = fig.add_axes([0.72, 0.14, 0.26, 0.36])

    # ---- a
    ax_a.axvspan(*HUMAN, color="#E8E8E8", lw=0, zorder=0)
    for i, m in enumerate(nets):
        col = COL[m["group"]]
        ax_a.plot([m["rate_w0"], m["rate_w1"]], [i, i], color=col, lw=0.8, alpha=0.35, solid_capstyle="butt", zorder=1)
        ax_a.plot([m["texture_end"], m["shape_end"]], [i, i], color=col, lw=3.2, solid_capstyle="butt", zorder=2)
        ax_a.plot(m["trained"], i, marker=MARK[m["family"]], ms=3.6, mfc="white", mec=col, mew=0.9, zorder=3)
    ax_a.set_yticks(range(n))
    ax_a.set_yticklabels([m["label"] for m in nets], fontsize=6)
    for lab, m in zip(ax_a.get_yticklabels(), nets):
        lab.set_color(COL[m["group"]] if m["group"] != "imagenet" else S.INK)
    ax_a.set_ylim(-0.8, n - 0.2)
    ax_a.set_xlim(0, 1)
    ax_a.set_xlabel("cue-conflict shape-bias rate")
    ax_a.tick_params(axis="y", length=0)
    ax_a.spines["left"].set_visible(False)
    ax_a.text(0.955, n - 0.6, "humans", ha="center", va="top", fontsize=6, color=S.MUTED, rotation=90)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=S.PUB, lw=3.2, label="reachable within 5 pts of top-1 (subspace rule)"),
               Line2D([], [], color=S.PUB, lw=0.8, alpha=0.35, label="whole sweep, w = 0 → 1"),
               Line2D([], [], marker="o", ls="", mfc="white", mec=S.PUB, label="trained readout, convolutional"),
               Line2D([], [], marker="^", ls="", mfc="white", mec=S.PUB, label="trained readout, attention-based"),
               Line2D([], [], color=S.MATCH, lw=3.2, label="ResNet-50, V2 training recipe"),
               Line2D([], [], color=S.HIGHSF, lw=3.2, label="Stylized-ImageNet ResNet-50s (Geirhos et al., 2019)")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.004), ncol=3, fontsize=5.8, frameon=False, columnspacing=1.6)
    ax_a.text(-0.42, 1.0, "a", transform=ax_a.transAxes, fontsize=10, fontweight="bold", va="bottom")

    # ---- b
    for m in nets:
        ax_b.plot(m["trained"], m["gated_span"], marker=MARK[m["family"]], ms=4, ls="", mfc=COL[m["group"]] if m["group"] != "imagenet" else "white",
                  mec=COL[m["group"]], mew=0.9)
    ax_b.axhline(0.20, color=S.RULE, lw=0.7, ls="--", zorder=0)
    ax_b.set_xlabel("trained rate")
    ax_b.set_ylabel("gated span (subspace rule)")
    ax_b.set_xlim(0, 1); ax_b.set_ylim(0, max(0.5, max(m["gated_span"] for m in nets) + 0.05))
    if st:
        ax_b.text(0.98, 0.03, f"ImageNet-trained, n = {st['n']}\nspan {st['gated_span_min']:.2f}–{st['gated_span_max']:.2f}; ρ(span, rate) {st['rho_span_vs_trained']:+.2f}\n"
                  f"τ(trained, shape-end rank) {st['tau_trained_vs_shape_end']:.2f}\n{100 * st['frac_reversed']:.0f}% of pairs reverse order",
                  transform=ax_b.transAxes, fontsize=5.5, va="bottom", ha="right", color=S.MUTED)
    ax_b.text(-0.32, 1.0, "b", transform=ax_b.transAxes, fontsize=10, fontweight="bold", va="bottom")

    # ---- c
    pr = [m for m in nets if "probe" in m]
    for m in pr:
        ax_c.plot(m["trained"], m["probe"]["bias"], marker=MARK[m["family"]], ms=4, ls="", mfc=COL[m["group"]] if m["group"] != "imagenet" else "white",
                  mec=COL[m["group"]], mew=0.9)
    ax_c.axhline(0, color=S.RULE, lw=0.7, zorder=0)
    ax_c.set_xlabel("trained rate")
    ax_c.set_ylabel("probe bias (shape − texture accuracy)")
    ax_c.set_xlim(0, 1)
    if pr:
        lo = min(m["probe"]["bias"] for m in pr); hi = max(m["probe"]["bias"] for m in pr)
        ax_c.set_ylim(min(lo - 0.05, -0.05), max(hi + 0.05, 0.05))
    if "tau_trained_vs_probe_bias" in st:
        ax_c.text(0.98, 0.03, f"τ(trained, probe-bias rank) {st['tau_trained_vs_probe_bias']:.2f}\nn = {st['n_probe']}",
                  transform=ax_c.transAxes, fontsize=5.5, va="bottom", ha="right", color=S.MUTED)
    ax_c.text(-0.32, 1.0, "c", transform=ax_c.transAxes, fontsize=10, fontweight="bold", va="bottom")

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(S.OUT, f"figE6_zoo.{ext}"))
    print("wrote", os.path.join(S.OUT, "figE6_zoo.{pdf,png}"), f"({n} networks)")


if __name__ == "__main__":
    main()
