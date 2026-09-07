"""Figure 5: matching the human shape-bias rate is not matching human decisions.

PAPER_PLAN §57 (pre-registered), §59; PAPER_DRAFT §2.6.2; Extended Data §E6.3.

This is the paper's third dissociation and the only one about human behaviour
directly. A null needs three things drawn, and each gets a panel:

  (a) THE DISSOCIATION. Every one of the 189 readouts (9 runs x 21 values of
      `w`) as one point: shape-bias rate against trial-level agreement with the
      same 10 observers, as a percentage of the human-human ceiling. The x-axis
      spans 3x; the y-axis never leaves the bottom fifth. Because `w` moves the
      rate over FROZEN representations (§2.3), the horizontal spread inside a
      single arm is not a model difference -- it is one network being re-read.
      The three published CNNs are drawn on the same axes so the reader can see
      that varying architecture, data and objective lands in the same band.

  (b) THE POSITIVE CONTROL. An equivalence interval bounds a null but cannot
      show the measure would have SEEN an effect had one been there. On the
      same 1200 images and the same 10 observers, the same machinery recovers
      ~3x more structure from image difficulty than from the shape-texture
      decision. Both are plotted as a percentage of their OWN split-half
      ceiling, which is the only unit in which two different targets compare.

  (c) WHAT WE DO NOT CONCLUDE. (b) shows the diet arm above the no-diet arm on
      difficulty, which invites a claim about the developmental front end. The
      crossed 2x2 does not support it -- it is a train/test MATCHING pattern,
      not a weights effect -- so the control that kills the claim is drawn
      beside the result that invited it, rather than left out of the figure.

Deliberately NOT drawn: a regression line through (a). Fitting a slope to
points whose x-position was set by a free parameter would invite reading the
scatter as an estimated relationship; the claim is a bound, and the bound is
drawn as one.

Usage: PYTHONPATH=. python tmp_analysis/fig5_human_correspondence.py
"""
from __future__ import annotations

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SWEEP = "results/sweep_error_consistency"
PERIMAGE = ("/scratch/byrong/mpk/perimage_behavior/results/"
            "2026-09-07_geirhos_perimage/perimage_correspondence.json")
OUT = "results/figures"

DIET = "#0072B2"    # blue        -- diet arm
DIETL = "#56B4E9"   # sky blue    -- diet arm, other checkpoint policy
NODIET = "#D55E00"  # vermillion  -- no-diet arm
PUB = "#5A5A5A"     # grey        -- published CNNs (not ours)
HUMAN = "#000000"
INK = "#1A1A1A"
MUTED = "#767676"
GREY = "#5A5A5A"

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "axes.edgecolor": "#999999", "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "figure.dpi": 200, "font.family": "DejaVu Sans",
    "legend.frameon": False,
})

# Geirhos et al. (2020), Table/Fig. values for CNNs trained on ImageNet.
# These are the ONLY hardcoded numbers in this figure and they are published,
# not measured here; they are labelled as published in the panel.
PUBLISHED = [("AlexNet", 0.429, 0.032), ("GoogLeNet", 0.312, 0.025),
             ("VGG-16", 0.172, 0.017)]


def load_sweeps() -> tuple[dict[str, list[dict]], float, float]:
    """The 9 kappa sweeps, grouped by arm, plus the human anchors they assert."""
    groups = {"diet_final": "dkl_bcs_seed*_final", "diet_best": "dkl_bcs_seed*_best",
              "nodiet": "dkl_base_seed*_final"}
    out, ceils, hsbs = {}, set(), set()
    for name, pat in groups.items():
        runs = [json.load(open(f))
                for f in sorted(glob.glob(f"{SWEEP}/{pat}_kappa_sweep.json"))]
        if len(runs) != 3:
            raise SystemExit(f"{name}: expected 3 runs, found {len(runs)}")
        out[name] = runs
        ceils |= {round(r["human_human_ceiling"], 6) for r in runs}
        hsbs |= {round(r["human_shape_bias"], 6) for r in runs}
    # Every run recomputed the humans independently; if they disagree, one of
    # the nine scored a different observer set and the panel would be mixing
    # incompatible ceilings.
    if len(ceils) != 1 or len(hsbs) != 1:
        raise SystemExit(f"runs disagree on the human anchors: {ceils} {hsbs}")
    return out, ceils.pop(), hsbs.pop()


def xy(runs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    sb = np.array([[r["shape_bias"] for r in run["sweep"]] for run in runs])
    pc = np.array([[r["kappa_pct_ceiling"] for r in run["sweep"]] for run in runs])
    return sb, pc


def main():
    sweeps, ceiling, human_sb = load_sweeps()
    per = json.load(open(PERIMAGE))
    M = per["models"]

    fig = plt.figure(figsize=(7.2, 2.95))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.42, 1.0, 1.0], wspace=0.42,
                          left=0.070, right=0.995, top=0.835, bottom=0.205)

    # ---------------- (a) the dissociation -----------------------------------
    ax = fig.add_subplot(gs[0, 0])
    arms = [("diet_best", DIET, "diet, best ckpt"),
            ("diet_final", DIETL, "diet, final ckpt"),
            ("nodiet", NODIET, "no diet")]
    best_overall = -np.inf
    for key, colour, label in arms:
        sb, pc = xy(sweeps[key])
        for s in range(sb.shape[0]):
            ax.plot(sb[s], pc[s], "-", color=colour, lw=0.6, alpha=0.30, zorder=2)
        ax.plot(sb.ravel(), pc.ravel(), "o", ms=2.5, color=colour, mew=0,
                alpha=0.85, zorder=3, label=label)
        best_overall = max(best_overall, np.nanmax(pc))
        print(f"  {label:18s} rate {sb.min():.3f}-{sb.max():.3f}  "
              f"best {np.nanmax(pc):.1f}% of ceiling")

    for name, sb_p, k_p in PUBLISHED:
        ax.plot(sb_p, 100 * k_p / ceiling, "D", ms=3.6, color=PUB, mew=0, zorder=4)
        # AlexNet sits between the other two on x; lifting its label above
        # the marker is what keeps the three from running together.
        up = name == "AlexNet"
        ax.annotate(name, (sb_p, 100 * k_p / ceiling),
                    textcoords="offset points",
                    xytext=(0, 6.5 if up else -6.5), ha="center",
                    va="bottom" if up else "top", fontsize=6.0, color=PUB,
                    bbox=dict(boxstyle="square,pad=0.10", fc="white",
                              ec="none", alpha=0.82))

    ax.axhline(100, color=HUMAN, lw=1.0, zorder=5)
    ax.text(0.985, 100, "human–human ceiling  (κ = %.3f)" % ceiling,
            fontsize=6.4, va="bottom", ha="right", color=HUMAN)
    ax.axhline(best_overall, color=GREY, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.text(0.145, best_overall + 2.5, "best of all 189 readouts: %.1f%%" % best_overall,
            fontsize=6.2, va="bottom", ha="left", color=GREY)
    ax.axvline(human_sb, color=HUMAN, lw=0.7, ls=(0, (2, 2)), zorder=1)
    ax.text(human_sb - 0.012, 62, "human rate %.3f" % human_sb, fontsize=6.2,
            rotation=90, va="center", ha="right", color=HUMAN)

    ax.set_xlim(0.13, 1.0)
    ax.set_ylim(-12, 118)
    ax.set_xlabel("cue-conflict shape-bias rate")
    ax.set_ylabel("error consistency\n(% of human–human ceiling)")
    ax.set_title("a   Rate moves 3×; agreement does not", loc="left",
                 fontsize=8.2, fontweight="bold", pad=5)
    handles = [plt.Line2D([], [], color=c, lw=0, marker="o", ms=3.4, mew=0,
                          label=l) for _k, c, l in arms]
    handles.append(plt.Line2D([], [], color=PUB, lw=0, marker="D", ms=3.4,
                              mew=0, label="published CNNs (Geirhos 2020)"))
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.012, 0.86),
              fontsize=6.3, handletextpad=0.35, labelspacing=0.28,
              borderpad=0.1)

    # ---------------- (b) the positive control -------------------------------
    ax = fig.add_subplot(gs[0, 1])
    MEAS = [("shape_rate_resid__pearson", "which cue\nis picked"),
            ("valid_rate_resid__pearson", "whether either\ncue resolves")]
    ARMS = [("dkl_bcs_seed{}_final", DIET, "diet"),
            ("dkl_base_seed{}_final", NODIET, "no diet")]
    width, gap = 0.30, 0.10
    for mi, (meas, mlabel) in enumerate(MEAS):
        for ai, (tmpl, colour, alabel) in enumerate(ARMS):
            vals = np.array([M[tmpl.format(s)][meas]["oos_pct_ceiling"]
                             for s in range(3)])
            x0 = mi + (ai - 0.5) * (width + gap)
            ax.bar(x0, vals.mean(), width=width, color=colour, alpha=0.30,
                   edgecolor=colour, lw=0.9, zorder=2)
            ax.plot(np.full(3, x0), vals, "o", ms=3.0, color=colour, mew=0,
                    zorder=4, alpha=0.95)
            print(f"  {mlabel!r:28s} {alabel:8s} {vals.mean():5.1f}% "
                  f"[{vals.min():.1f}, {vals.max():.1f}]")
    ax.axhline(0, color="#BBBBBB", lw=0.8, zorder=1)
    ax.set_xticks(range(len(MEAS)))
    ax.set_xticklabels([m[1] for m in MEAS], fontsize=6.8)
    ax.set_xlim(-0.55, len(MEAS) - 0.45)
    ax.set_ylim(-2, 32)
    ax.set_ylabel("out-of-sample correspondence\n(% of split-half ceiling)", fontsize=7.4)
    ax.set_title("b   The measure does detect\n     a correspondence", loc="left",
                 fontsize=8.2, fontweight="bold", pad=5)
    h = [plt.Line2D([], [], color=c, lw=3, alpha=0.55, label=l) for _t, c, l in ARMS]
    ax.legend(handles=h, loc="upper left", fontsize=6.4, handlelength=1.1,
              handletextpad=0.4, labelspacing=0.25, borderpad=0.1)

    # ---------------- (c) the control that withdraws the reading -------------
    ax = fig.add_subplot(gs[0, 2])
    CELLS = [("dkl_bcs_seed{}_final",      1, 1, DIET,   "diet w."),
             ("dkl_bcs_seed{}_final_nofe", 1, 0, DIET,   "diet w."),
             ("dkl_base_seed{}_final_fe",  0, 1, NODIET, "no-diet w."),
             ("dkl_base_seed{}_final",     0, 0, NODIET, "no-diet w.")]
    for tmpl, wrow, fecol, colour, _lab in CELLS:
        vals = np.array([M[tmpl.format(s)]["valid_rate_resid__pearson"]["oos_pct_ceiling"]
                         for s in range(3)])
        matched = (wrow == fecol)
        x0 = fecol
        ax.plot([x0 - 0.17, x0 + 0.17], [vals.mean()] * 2, "-", color=colour,
                lw=2.0, alpha=1.0 if matched else 0.42, zorder=3,
                solid_capstyle="butt")
        ax.plot(np.full(3, x0), vals, "o", ms=3.0, color=colour, mew=0,
                alpha=0.95 if matched else 0.42, zorder=4)
        print(f"  2x2 weights={'diet' if wrow else 'no diet':8s} "
              f"front-end={'ON ' if fecol else 'OFF'}  {vals.mean():+5.1f}%"
              f"  {'matched' if matched else 'crossed'}")
    for wrow, colour in [(1, DIET), (0, NODIET)]:
        pts = []
        for tmpl, wr, fc, c, _l in CELLS:
            if wr != wrow:
                continue
            v = np.mean([M[tmpl.format(s)]["valid_rate_resid__pearson"]["oos_pct_ceiling"]
                         for s in range(3)])
            pts.append((fc, v))
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "-", color=colour,
                lw=1.0, alpha=0.55, zorder=2)
    ax.axhline(0, color="#BBBBBB", lw=0.8, zorder=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["front end\noff", "front end\non"], fontsize=6.8)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-14, 34)
    ax.set_ylabel("difficulty correspondence\n(% of ceiling)", fontsize=7.4)
    ax.set_title("c   …but the 2×2 withdraws\n     the diet reading", loc="left",
                 fontsize=8.2, fontweight="bold", pad=5)
    ax.text(0.5, -0.30, "matched cells solid, crossed pale; crossed cells\n"
                        "resolve neither cue on 70–85% of images",
            transform=ax.transAxes, ha="center", va="top", fontsize=5.8,
            color=GREY, style="italic")
    h = [plt.Line2D([], [], color=c, lw=2, label=l)
         for c, l in [(DIET, "diet weights"), (NODIET, "no-diet weights")]]
    ax.legend(handles=h, loc="upper left", fontsize=6.4, handlelength=1.1,
              handletextpad=0.4, labelspacing=0.25, borderpad=0.1)

    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        path = f"{OUT}/fig5_human_correspondence.{ext}"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        print("saved ->", path)


if __name__ == "__main__":
    main()
