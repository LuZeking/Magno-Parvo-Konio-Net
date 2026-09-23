"""Extended Data Fig. 6: the readout account across the model zoo (PLAN_ZOO), on held-out exemplars (PLAN_EXEMPLAR).

  MPLCONFIGDIR=... PYTHONPATH=.:tmp_analysis/figs_v2 python tmp_analysis/figs_v2/figE6_zoo.py

a  every network sorted by its trained rate: the rates reachable within the accuracy gate with the subspaces
   fitted on exemplars the rate is not scored on (E1, texture end to shape end) as a bar, the same range with
   exemplars shared (E0) as a pale bar behind it, the trained rate as a dot; the ImageNet-trained networks, the
   recipe variant and the three Stylized-ImageNet controls in three colours.
b  held-out gated span against trained rate (convolutional vs attention-based; controls marked).
c  exemplar-balanced probe bias (Q1, leave-one-exemplar-out; 5-95% over draws) against trained rate.
Reads results/figures_v2/exemplar_zoo.json (spans, probes, ranking) and zoo_behaviour.json (architecture family);
writes results/figures_v2/figE6_zoo.{pdf,png}.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(__file__))
import style as S  # noqa: E402

COL = {"imagenet": S.PUB, "recipe": S.MATCH, "sin": S.HIGHSF}
MARK = {"conv": "o", "attention": "^"}
HUMAN = (0.92, 0.99)
SHARED = "#DADADA"


def main():
    ex = json.load(open("results/figures_v2/exemplar_zoo.json"))
    fam = {m["tag"]: m["family"] for m in json.load(open("results/figures_v2/zoo_behaviour.json"))["networks"]}
    nets = sorted(ex["networks"], key=lambda m: m["trained"])
    for m in nets:
        m["family"] = fam[m["tag"]]
    gate = {g["tag"]: g for g in json.load(open("results/figures_v2/exemplar_gate.json"))["networks"]}
    gs = json.load(open("results/figures_v2/exemplar_gate.json"))["summary"]["heldout"]
    fu = json.load(open("results/figures_v2/exemplar_floors.json"))["units"]
    for m in nets:                      # E1 under the gate measured on the same held-out bases (addendum C)
        m["hg"] = gate[m["tag"]]["heldout"]
    rk, n = ex["stats"]["ranking"], len(nets)
    im = [m for m in nets if m["group"] == "imagenet"]

    fig = plt.figure(figsize=(S.FULL, max(5.0, 0.105 * n + 1.9)))
    ax_a = fig.add_axes([0.20, 0.14, 0.40, 0.82])
    ax_b = fig.add_axes([0.72, 0.60, 0.26, 0.36])
    ax_c = fig.add_axes([0.72, 0.14, 0.26, 0.36])

    # ---- a
    ax_a.axvspan(*HUMAN, color="#E8E8E8", lw=0, zorder=0)
    for i, m in enumerate(nets):
        col, e0, e1 = COL[m["group"]], m["spans"]["E0"], m["hg"]
        ax_a.plot([e0["texture_end"], e0["shape_end"]], [i, i], color=SHARED, lw=5.0, solid_capstyle="butt", zorder=1)
        ax_a.plot([e1["texture_end"], e1["shape_end"]], [i, i], color=col, lw=2.6, solid_capstyle="butt", zorder=2)
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
    handles = [Line2D([], [], color=S.PUB, lw=2.6, label="reachable within 5 pts of top-1, exemplars held out"),
               Line2D([], [], color=SHARED, lw=5.0, label="the same, exemplars shared"),
               Line2D([], [], marker="o", ls="", mfc="white", mec=S.PUB, label="trained readout, convolutional"),
               Line2D([], [], marker="^", ls="", mfc="white", mec=S.PUB, label="trained readout, attention-based"),
               Line2D([], [], color=S.MATCH, lw=2.6, label="ResNet-50, V2 training recipe"),
               Line2D([], [], color=S.HIGHSF, lw=2.6, label="Stylized-ImageNet ResNet-50s (Geirhos et al., 2018a)")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.004), ncol=3, fontsize=5.8, frameon=False, columnspacing=1.6)
    ax_a.text(-0.42, 1.0, "a", transform=ax_a.transAxes, fontsize=10, fontweight="bold", va="bottom")

    # ---- b
    for m in nets:
        ax_b.plot(m["trained"], m["hg"]["gated_span"], marker=MARK[m["family"]], ms=4, ls="",
                  mfc=COL[m["group"]] if m["group"] != "imagenet" else "white", mec=COL[m["group"]], mew=0.9)
    ax_b.axhline(0.20, color=S.RULE, lw=0.7, ls="--", zorder=0)
    ax_b.set_xlabel("trained rate")
    ax_b.set_ylabel("gated span, exemplars held out")
    ax_b.set_xlim(0, 1); ax_b.set_ylim(0, 0.5)
    g = np.array([m["hg"]["gated_span"] for m in im]); tr = np.array([m["trained"] for m in im])
    rho = spearmanr(g, tr)[0]
    fi, ft = fu["image"], fu["texture_exemplar"]
    ax_b.text(0.98, 0.97, f"ImageNet-trained, n = {len(im)}\nspan {g.min():.2f}–{g.max():.2f} (median {np.median(g):.2f}); "
              f"ρ(span, rate) {rho:+.2f}\nτ(trained, shape-end rank) {gs['tau']:.2f}\n"
              f"{100 * gs['frac_reversed']:.0f}% of pairs reverse (floor {100 * fi['floor_all']:.1f}% images,\n"
              f"{100 * ft['floor_all']:.1f}% texture exemplars); resolved pairs:\n"
              f"{ft['observed']['heldout_gate']['n_resolved_reversed']}/{ft['n_resolved']} reverse (floor {100 * ft['floor_resolved']:.2f}%)",
              transform=ax_b.transAxes, fontsize=5.5, va="top", ha="right", color=S.MUTED)
    ax_b.text(-0.32, 1.0, "b", transform=ax_b.transAxes, fontsize=10, fontweight="bold", va="bottom")
    print(f"  (b) n={len(im)} E1 gated {g.min():.3f}-{g.max():.3f} median {np.median(g):.3f} n>=0.20 {(g >= 0.2).sum()} "
          f"rho {rho:+.3f} tau {gs['tau']:.3f} reversed {gs['frac_reversed']:.3f}")

    # ---- c
    ax_c.axhline(0, color=S.RULE, lw=0.7, zorder=0)
    for m in nets:
        q = m["probes"]["Q1"]
        ax_c.plot([m["trained"]] * 2, [q["bias_p05"], q["bias_p95"]], "-", color=COL[m["group"]], lw=0.6, alpha=0.6, zorder=1)
        ax_c.plot(m["trained"], q["bias"], marker=MARK[m["family"]], ms=4, ls="",
                  mfc=COL[m["group"]] if m["group"] != "imagenet" else "white", mec=COL[m["group"]], mew=0.9, zorder=2)
        if m["tag"] == "hub_resnet50_sin":
            ax_c.annotate("SIN", (m["trained"], q["bias"]), xytext=(-6, 0), textcoords="offset points", fontsize=5.8,
                          color=COL["sin"], ha="right", va="center")
    ax_c.set_xlabel("trained rate")
    ax_c.set_ylabel("probe bias, exemplars held out\n(shape − texture accuracy)")
    ax_c.set_xlim(0, 1)
    lo = min(m["probes"]["Q1"]["bias_p05"] for m in nets); hi = max(m["probes"]["Q1"]["bias_p95"] for m in nets)
    ax_c.set_ylim(min(lo - 0.05, -0.05), max(hi + 0.05, 0.05))
    qi = np.array([m["probes"]["Q1"]["bias"] for m in im])
    ax_c.text(0.03, 0.97, f"τ(trained, probe-bias rank) {ex['stats']['tau_trained_Q1']:.2f}, n = {len(nets)}\n"
              f"ImageNet-trained {qi.min():+.2f} to {qi.max():+.2f}".replace(" -", " −"),
              transform=ax_c.transAxes, fontsize=5.5, va="top", ha="left", color=S.MUTED)
    ax_c.text(-0.32, 1.0, "c", transform=ax_c.transAxes, fontsize=10, fontweight="bold", va="bottom")
    print(f"  (c) Q1 ImageNet {qi.min():+.3f}..{qi.max():+.3f}, below 0: {(qi < 0).sum()}/{len(qi)}; tau {ex['stats']['tau_trained_Q1']:.3f}")

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(S.OUT, f"figE6_zoo.{ext}"))
    print("wrote", os.path.join(S.OUT, "figE6_zoo.{pdf,png}"), f"({n} networks)")


if __name__ == "__main__":
    main()
