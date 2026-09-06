"""Figure 3: shape bias is set by the readout, over representations that never change.

PAPER_PLAN §52.3 / §20.3; PAPER_DRAFT §2.3.

The figure makes the paper's central claim as a MEASUREMENT rather than an
argument. `y1` and `y2` -- the two streams' pre-classifier features -- do not
depend on `w`, so sweeping `w` moves the behavioural readout while holding both
representations EXACTLY fixed. Nothing is retrained, refitted or reselected
between the leftmost and rightmost point of panel (a).

  (a) the sweep. 21 values of `w`, 3 seeds, both checkpoint policies. The
      w = 0.5 rule is drawn because that column is the published model and is
      reproduced here EXACTLY (§2.3) -- it is the anchor showing the sweep and
      the main pipeline are the same computation, so it is marked, not hidden.

  (b) the same y-axis, showing the field. This is the panel that carries the
      argument: the span one network covers by reweighting alone (panel a)
      against the span the literature covers by training different networks.
      Humans are drawn because they are the reference; they are also the one
      point the sweep does NOT reach, and drawing it is the honest form of the
      result (§2.3 claims no match).

Deliberately NOT drawn: a fitted `w`. No value of the weighting was estimated
from human data (§2.3), so no curve here is a fit to the human point and none
is drawn as one.

Usage: PYTHONPATH=. python tmp_analysis/fig3_readout_sweep.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SWEEP = "results/fusion_sweep"
CC_BEST = "results/cue_conflict_bestep"
CC_FINAL = "results/cue_conflict"
CC_SINGLE = "results/cue_conflict_single_bestep"
OUT = "results/figures"

# Okabe-Ito, matching fig1.py so the two figures share an identity.
BEST = "#0072B2"   # blue       -- best saved checkpoints
FINAL = "#56B4E9"  # sky blue   -- final checkpoints
NODIET = "#D55E00"  # vermillion
HUMAN = "#000000"
GREY = "#5A5A5A"
INK = "#1A1A1A"
MUTED = "#767676"

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


def load_sweep(policy: str) -> tuple[np.ndarray, np.ndarray]:
    """(w grid, seeds x w) shape bias for one checkpoint policy."""
    paths = sorted(glob.glob(os.path.join(SWEEP, f"*_{policy}_wsweep.json")))
    if not paths:
        sys.exit(f"no sweep files for policy {policy!r} in {SWEEP}")
    ws, rows = None, []
    for p in paths:
        d = json.load(open(p))
        pts = sorted(d["sweep"], key=lambda r: r["w"])
        w = np.array([r["w"] for r in pts])
        if ws is None:
            ws = w
        elif not np.allclose(ws, w):
            sys.exit(f"w grid mismatch in {p}")
        rows.append([r["shape_bias"] for r in pts])
    return ws, np.array(rows)


def cc(dirname: str, cond: str, prefix: str) -> np.ndarray:
    """Per-seed cue-conflict shape bias for one condition of one model arm.

    `prefix` is REQUIRED: results/cue_conflict holds every condition (swap,
    parvo_highsf, rgb, ...), so a bare glob silently averages across arms.
    """
    out = []
    for p in sorted(glob.glob(os.path.join(dirname, f"{prefix}*.json"))):
        d = json.load(open(p))
        if cond in d.get("conditions", {}):
            out.append(d["conditions"][cond]["shape_bias"])
    if not out:
        sys.exit(f"no {cond!r} for {prefix!r} in {dirname}")
    return np.array(out)


def humans() -> np.ndarray:
    """Ten Geirhos observers, scored by the identical statistic (§4.3)."""
    import contextlib
    import io
    sys.path.insert(0, "tmp_analysis")
    # NB: this module runs its whole analysis at import time.
    with contextlib.redirect_stdout(io.StringIO()):
        from human_shapebias import human_observers
        obs = human_observers()
    return np.array([o["shape_bias"] for o in obs])


w, sb_best = load_sweep("best")
_, sb_final = load_sweep("final")

diet_full = cc(CC_BEST, "full", "dkl_bcs_seed")
diet_magno = cc(CC_BEST, "magno_only", "dkl_bcs_seed")
# no-diet's best saved checkpoint IS epoch 150 (§2.6.2), i.e. the final one.
nodiet_full = cc(CC_FINAL, "full", "dkl_base_seed")
single = cc(CC_SINGLE, "full", "dkl_bcs_single_seed")
hum = humans()

# Published values from the Geirhos et al. (2018a) release, reproduced by our
# pipeline (§2.3). Hardcoded because they are the published reference, not our
# measurement.
PUBLISHED = [("VGG-16", 0.172), ("GoogLeNet", 0.312), ("AlexNet", 0.429)]

fig = plt.figure(figsize=(7.2, 3.1))
gs = fig.add_gridspec(1, 2, width_ratios=[2.45, 1.0], wspace=0.06)
axa = fig.add_subplot(gs[0, 0])
axb = fig.add_subplot(gs[0, 1], sharey=axa)

YLO, YHI = 0.12, 1.005

# ---- panel (a): the sweep -------------------------------------------------
axa.axhline(0.5, color=GREY, lw=0.7, ls=(0, (4, 3)), zorder=1)
axa.text(0.012, 0.5, "parity", color=GREY, fontsize=6.5, va="bottom", ha="left")
axa.axhline(hum.mean(), color=HUMAN, lw=0.9, zorder=1)
axa.text(0.012, hum.mean(), f"human observers  {hum.mean():.3f}", color=HUMAN,
         fontsize=6.5, va="bottom", ha="left")

for arr, col, lab in ((sb_final, FINAL, "final checkpoints"),
                      (sb_best, BEST, "best saved checkpoints")):
    for r in arr:
        axa.plot(w, r, color=col, lw=0.7, alpha=0.42, zorder=2)
    axa.plot(w, arr.mean(0), color=col, lw=2.0, zorder=3, label=lab,
             solid_capstyle="round")

axa.axvline(0.5, color=GREY, lw=0.7, ls=":", zorder=1)
m50 = sb_final[:, np.isclose(w, 0.5)].mean()
axa.annotate(f"w = 0.5\nthe published model\n{m50:.3f}",
             xy=(0.5, m50), xytext=(0.115, 0.80), fontsize=6.5, color=INK,
             ha="left", va="center",
             arrowprops=dict(arrowstyle="-", lw=0.7, color=GREY,
                             connectionstyle="arc3,rad=0.22"))

# Span of the MEAN best-saved curve -- the quantity §2.3 reports. NOT the
# across-seed envelope, which is wider (0.54) and a different statistic.
mb = sb_best.mean(0)
lo, hi = mb.min(), mb.max()
axa.annotate("", xy=(0.045, lo), xytext=(0.045, hi),
             arrowprops=dict(arrowstyle="<->", lw=0.9, color=INK))
axa.text(0.062, (lo + hi) / 2, f"{hi - lo:.2f}\nspanned by\nreweighting alone",
         fontsize=6.5, color=INK, va="center", ha="left")

axa.set_xlabel("$w$  —  weight on the magno stream in the readout")
axa.set_ylabel("cue-conflict shape bias")
axa.set_xlim(-0.005, 1.005)
axa.set_ylim(YLO, YHI)
axa.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
axa.legend(loc="lower right", handlelength=1.5)
axa.set_title("a   one network, representations frozen", loc="left",
              fontweight="bold", pad=6)

# ---- panel (b): the field, same y-axis ------------------------------------
axb.plot([0.10, 0.38], [0.5, 0.5], color=GREY, lw=0.7, ls=(0, (4, 3)), zorder=1)
axb.plot([0.10, 0.38], [hum.mean()] * 2, color=HUMAN, lw=0.9, zorder=1)

axb.axhspan(lo, hi, color=BEST, alpha=0.10, zorder=0)
axb.text(0.045, (lo + hi) / 2, "range reached by panel a", fontsize=6.2,
         color=BEST, ha="center", va="center", rotation=90, zorder=1)

pts = [(v, n, GREY, "o") for n, v in PUBLISHED]
pts += [(nodiet_full.mean(), "no diet", NODIET, "s"),
        (single.mean(), "single stream", NODIET, "D"),
        (diet_full.mean(), "diet, integrated", BEST, "s"),
        (diet_magno.mean(), "diet, magno stream", BEST, "^"),
        (hum.mean(), "humans", HUMAN, "*")]

# Labels collide (AlexNet 0.429 / no diet 0.416; humans 0.958 / magno 0.903),
# so push them apart to a minimum separation and draw a leader where a label
# has moved off its marker.
SEP = 0.052
pts.sort(key=lambda t: t[0])
ys = [t[0] for t in pts]
for i in range(1, len(ys)):
    if ys[i] - ys[i - 1] < SEP:
        ys[i] = ys[i - 1] + SEP
shift = (max(ys) - YHI + 0.02)
if shift > 0:
    ys = [y - shift for y in ys]

for (v, n, col, mk), y in zip(pts, ys):
    ms, z = (8, 5) if mk == "*" else (4.5, 4)
    axb.plot([0.28], [v], marker=mk, ms=ms, color=col, mec="white", mew=0.6,
             zorder=z, clip_on=False)
    if abs(y - v) > 0.004:
        axb.plot([0.315, 0.40], [v, y], color=col, lw=0.5, alpha=0.55, zorder=2)
    axb.text(0.43, y, f"{n}  {v:.3f}", fontsize=6.5, color=col, va="center",
             ha="left")

axb.set_xlim(0, 1)
axb.set_xticks([])
for sp_ in ("left", "bottom"):
    axb.spines[sp_].set_visible(False)
plt.setp(axb.get_yticklabels(), visible=False)
axb.tick_params(axis="y", length=0)
axb.set_title("b   the field, one network each", loc="left",
              fontweight="bold", pad=6)

os.makedirs(OUT, exist_ok=True)
fig.savefig(os.path.join(OUT, "fig3_readout_sweep.png"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig3_readout_sweep.pdf"), bbox_inches="tight")

print(f"w grid: {len(w)} values, {sb_best.shape[0]} seeds per policy")
print(f"best  : {sb_best[:, 0].mean():.3f} -> {sb_best[:, -1].mean():.3f}")
print(f"final : {sb_final[:, 0].mean():.3f} -> {sb_final[:, -1].mean():.3f}")
print(f"w=0.5 per-seed (final): "
      + " / ".join(f"{v:.4f}" for v in sb_final[:, np.isclose(w, 0.5)].ravel()))
print(f"span drawn: {lo:.3f}-{hi:.3f} = {hi-lo:.3f}")
print(f"humans n={len(hum)} mean={hum.mean():.4f} "
      f"range={hum.min():.3f}-{hum.max():.3f}")
print(f"diet full={diet_full.mean():.3f} magno={diet_magno.mean():.3f} "
      f"nodiet={nodiet_full.mean():.3f} single={single.mean():.3f}")
print("wrote results/figures/fig3_readout_sweep.{png,pdf}")
