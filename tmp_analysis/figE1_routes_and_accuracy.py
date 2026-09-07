"""Extended Data Figure 1: the two routes (§2.1), and the accuracy control (§2.4).

PAPER_DRAFT §2.1 Table 1 and §2.4's accuracy-matched control; Extended Data
§E1 and §E4.

Both panels are in Extended Data rather than the main text for the same reason:
each rests on ONE model per cell. The untrained references carry no training
seed at all, and the accuracy-matched control is a single retrained run. The
main-text claims they support are stated at that strength in §2.1 and §2.4, and
drawing them here keeps the figure's evidential weight matched to the text's.

  (a) THE ROUTE ASYMMETRY. Probe shape-texture bias before and after training,
      per stream. The magno stream's shape bias is essentially complete before
      any weight is updated and training adds +0.014; the parvo stream starts
      near neutral and acquires its texture preference over training (-0.395).
      The third line is the one that makes it causal rather than descriptive:
      with the transform removed, the untrained magno stream starts lower and
      training drives it ACROSS the sign flip. Ordinary training does not fail
      to build shape bias -- it destroys the shape bias the input supplied.

  (b) THE ACCURACY CONTROL. The diet arm is less accurate (§2.7), so the whole
      of §2.4 could be an accuracy effect. The fatal outcome was pre-registered
      before the run: a no-diet model AT the diet's accuracy showing a
      shape-biased magno stream would kill the paper's central claim. Both
      instruments are drawn on their own axis because they are on different
      scales, and both are drawn because they were built on different machinery
      and agree to within 0.6 percentage points on how much of the gap
      accuracy closes.

Deliberately NOT drawn: an error bar on any untrained point. Those references
have no seed, so an interval there would be manufactured.

Usage: PYTHONPATH=. python tmp_analysis/figE1_routes_and_accuracy.py
"""
from __future__ import annotations

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ST = "results/shape_texture"
ST_BEST = "results/shape_texture_bestep"
ST_MATCH = "results/shape_texture_matched"
CC = "results/cue_conflict"
CC_BEST = "results/cue_conflict_bestep"
CC_MATCH = "results/cue_conflict_matched"
OUT = "results/figures"

DIET = "#0072B2"
NODIET = "#D55E00"
MATCH = "#E69F00"   # orange -- the accuracy-matched control
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


def probe(path: str) -> tuple[float, float]:
    d = json.load(open(path))
    return d["streams"]["magno"]["bias"], d["streams"]["parvo"]["bias"]


def probe_arm(pattern: str, stream: str = "magno") -> np.ndarray:
    fs = sorted(glob.glob(pattern))
    if not fs:
        raise SystemExit(f"no files matched {pattern}")
    return np.array([json.load(open(f))["streams"][stream]["bias"] for f in fs])


def cc_arm(pattern: str, cond: str = "magno_only") -> np.ndarray:
    fs = sorted(f for f in glob.glob(pattern) if f.endswith(".json"))
    if not fs:
        raise SystemExit(f"no files matched {pattern}")
    return np.array([json.load(open(f))["conditions"][cond]["shape_bias"] for f in fs])


def main():
    fig = plt.figure(figsize=(7.2, 2.95))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.55, 1.0, 1.0], wspace=0.46,
                          left=0.070, right=0.990, top=0.825, bottom=0.165)

    # ---------------- (a) the two routes -------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    um, up = probe(f"{ST}/untrained_blur_cs.json")       # diet front end, untrained
    nm, _np_ = probe(f"{ST}/untrained_noblur.json")      # no diet front end, untrained
    tm, tp = probe(f"{ST}/dkl_bcs_seed0_trained.json")   # Table 1 uses seed 0
    bm, _bp = probe(f"{ST}/dkl_base_seed0_trained.json")

    # 3-seed trained ranges, drawn as whiskers so the single-seed lines are not
    # read as more precise than the arm actually is.
    d_m = probe_arm(f"{ST}/dkl_bcs_seed?_trained.json", "magno")
    d_p = probe_arm(f"{ST}/dkl_bcs_seed?_trained.json", "parvo")
    b_m = probe_arm(f"{ST}/dkl_base_seed?_trained.json", "magno")

    ROUTES = [("magno, with transform",    um, tm, d_m, DIET,   "-"),
              ("parvo (L−M), with transform", up, tp, d_p, DIET, "--"),
              ("magno, transform removed", nm, bm, b_m, NODIET, "-")]
    for label, y0, y1, rng, colour, ls in ROUTES:
        ax.plot([0, 1], [y0, y1], ls, color=colour, lw=1.5, zorder=3,
                marker="o", ms=4.0, mew=0, label=label)
        ax.vlines(1, rng.min(), rng.max(), color=colour, lw=3.0, alpha=0.30, zorder=2)
        ax.annotate(f"{y1 - y0:+.3f}", (1, y1), textcoords="offset points",
                    xytext=(7, 0), ha="left", va="center", fontsize=6.6, color=colour)
        print(f"  {label:28s} untrained {y0:+.4f} -> trained {y1:+.4f} "
              f"(learning {y1 - y0:+.4f}; 3-seed range {rng.max() - rng.min():.4f})")

    ax.axhline(0, color=INK, lw=0.9, zorder=1)
    ax.text(-0.045, 0.012, "sign flip", fontsize=6.3, va="bottom", ha="left", color=INK)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["untrained", "trained"])
    ax.set_xlim(-0.09, 1.30)
    ax.set_ylim(-0.72, 1.0)
    ax.set_ylabel("linear probe, shape − texture bias")
    ax.set_title("a   Shape bias is supplied by the input;\n"
                 "     texture bias is built by learning",
                 loc="left", fontsize=8.2, fontweight="bold", pad=5)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.015, 0.055), fontsize=6.4,
              handlelength=1.6, handletextpad=0.5, labelspacing=0.3,
              borderpad=0.15)
    ax.text(0.5, -0.185, "untrained references carry no training seed (n = 1); "
                         "bars are the 3-seed trained range",
            transform=ax.transAxes, ha="center", va="top", fontsize=5.8,
            color=GREY, style="italic")

    # ---------------- (b, c) the accuracy control ----------------------------
    # TWO panels, not one with twin axes. The probe (0 = sign flip) and
    # cue-conflict (0.5 = parity) scales have different zero points and
    # different units; overlaying them on a shared frame would put the
    # cue-conflict points at a probe value they do not have.
    #
    # The diet reference is the BEST-checkpoint arm, because that is the policy
    # the matched control was selected under (PLAN §68/§71.6); reading it from
    # the final-checkpoint files would silently compare across policies.
    PANELS = [
        ("b", "linear probe, magno\nshape − texture bias", 0.0, "sign flip",
         (-0.62, 1.00),
         probe_arm(f"{ST_BEST}/dkl_bcs_seed?_ep*.json", "magno"),
         probe_arm(f"{ST}/dkl_base_seed?_trained.json", "magno"),
         probe_arm(f"{ST_MATCH}/dkl_base_matched_seed0_ep010.json", "magno")),
        ("c", "cue-conflict shape bias,\nmagno stream only", 0.5, "parity",
         (0.30, 1.00),
         cc_arm(f"{CC_BEST}/dkl_bcs_seed?_ep*.json"),
         cc_arm(f"{CC}/dkl_base_seed?.json"),
         cc_arm(f"{CC_MATCH}/dkl_base_matched_seed0_ep010.json")),
    ]
    for pi, (letter, ylab, zero, zlab, ylim, diet, nod, matched) in enumerate(PANELS):
        a = fig.add_subplot(gs[0, pi + 1])
        for k, (vals, colour) in enumerate(
                [(diet, DIET), (nod, NODIET), (matched, MATCH)]):
            a.plot(np.full(len(vals), k), vals, "o", ms=4.2, color=colour,
                   mew=0, alpha=0.95, zorder=4)
            a.plot([k - 0.20, k + 0.20], [vals.mean()] * 2, "-", color=colour,
                   lw=2.0, zorder=3, solid_capstyle="butt")
        gap_closed = 100 * (matched.mean() - nod.mean()) / (diet.mean() - nod.mean())
        print(f"  {ylab.splitlines()[0]:32s} diet {diet.mean():+.4f}  "
              f"no diet {nod.mean():+.4f}  matched {matched.mean():+.4f}  "
              f"-> closes {gap_closed:.1f}% of the gap")

        # The gap the control had to close, drawn as the thing it did not close.
        a.annotate("", xy=(2.42, diet.mean()), xytext=(2.42, nod.mean()),
                   arrowprops=dict(arrowstyle="<->", color="#BBBBBB", lw=0.8))
        a.text(2.52, (diet.mean() + nod.mean()) / 2, "the gap\n§2.4 explains",
               fontsize=5.8, va="center", ha="left", color=GREY)
        a.annotate(f"closes\n{gap_closed:.1f}%", (2, matched.mean()),
                   textcoords="offset points", xytext=(2, -13), ha="center",
                   va="top", fontsize=6.2, color=MATCH)

        a.axhline(zero, color=INK, lw=0.9, zorder=1)
        a.text(-0.42, zero, zlab, fontsize=6.2, va="bottom", ha="left", color=INK)
        a.set_xticks([0, 1, 2])
        a.set_xticklabels(["diet", "no\ndiet", "no diet,\naccuracy-\nmatched"],
                          fontsize=6.4)
        a.set_xlim(-0.5, 3.35)
        a.set_ylim(*ylim)
        a.set_ylabel(ylab, fontsize=7.4)
        title = ("b   Matching accuracy does not\n     recover the shape bias"
                 if letter == "b" else
                 "c   …and a second instrument\n     agrees to 0.6 points")
        a.set_title(title, loc="left", fontsize=8.2, fontweight="bold", pad=5)
        if letter == "c":
            a.text(0.5, -0.315, "accuracy-matched arm is a single retrained run (n = 1)",
                   transform=a.transAxes, ha="center", va="top", fontsize=5.8,
                   color=GREY, style="italic")

    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        path = f"{OUT}/figE1_routes_and_accuracy.{ext}"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        print("saved ->", path)


if __name__ == "__main__":
    main()
