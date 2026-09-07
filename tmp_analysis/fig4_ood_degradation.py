"""Figure 4: shape bias does not confer out-of-distribution robustness.

PAPER_PLAN §53 (pre-registered), §56; PAPER_DRAFT §2.6.1; Extended Data §E6.2.

§2.6.1 is a NULL that the reader is primed to disbelieve: anyone who accepts
§2.4 expects the robustness to follow, because that is what the field reports.
A table of AUCs cannot show that the null is a property of the curves rather
than of the summary statistic, so this figure draws the curves themselves.

  (a-g) one panel per corruption family, relative accuracy against severity.
        Relative, not absolute, because the diet arm starts from a lower clean
        accuracy (§2.7) and an absolute plot would confound the two (§8). This
        is EXACTLY ood_verdict.py's `rel = acc / acc[:, :1]`, so the curves and
        the AUCs in the text are the same computation.

  (h) the summary: relative AUC per family, diet against no diet, one marker
      per seed. This is where complete separation is visible as separation
      rather than asserted as a p-value -- which matters because the exact
      two-sided Mann-Whitney floor at 3 vs 3 is p = 0.100, so the p-value alone
      can never look impressive no matter how clean the result is.

Two panels are annotated because they are the honest qualifications:
  * low-pass is the ONE family where the diet wins, and it was declared
    trained-in BEFORE the experiment ran -- the diet *is* a low-pass blur.
    Drawing it unmarked would flatter the paper; hiding it would be worse.
  * phase-scrambling is the one family where the arms are indistinguishable,
    bounded by equivalence to +/-0.033. It is drawn as a null, not as a win.

Deliberately NOT drawn: significance stars. At n=3 vs 3 the floor is p=0.100
and a star would misrepresent an exact-test floor as a threshold crossing.

Usage: PYTHONPATH=. python tmp_analysis/fig4_ood_degradation.py
"""
from __future__ import annotations

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sps

OOD = "results/ood_bestep"          # best saved checkpoint per seed (PLAN §71.6)
OUT = "results/figures"

# Okabe-Ito, matching fig1.py / fig3_readout_sweep.py so the figures share an identity.
DIET = "#0072B2"    # blue
NODIET = "#D55E00"  # vermillion
SWAP = "#009E73"    # bluish green
HIGHSF = "#CC79A7"  # reddish purple
INK = "#1A1A1A"
MUTED = "#767676"
GREY = "#5A5A5A"

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "axes.edgecolor": "#999999", "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "figure.dpi": 200, "font.family": "DejaVu Sans",
    "legend.frameon": False,
})

# (label, filename prefix, colour, is-shape-biased) -- CONDS as in ood_verdict.py
CONDS = [("diet",         "dkl_bcs_seed",              DIET,   True),
         ("no diet",      "dkl_base_seed",             NODIET, False),
         ("swap",         "dkl_bcs_swap_seed",         SWAP,   False),
         ("parvo_highsf", "dkl_bcs_parvo_highsf_seed", HIGHSF, False)]

# Pre-registered ladder order (§53). Kept explicit so a family added to the
# JSONs later cannot silently reorder the panels of a published figure.
FAMILIES = ["uniform-noise", "high-pass", "false-colour", "phase-scrambling",
            "rotation", "contrast", "low-pass"]

# Severity tick labels, per family, read from the JSON but shortened for print.
def sev_labels(sev: list[str], family: str) -> list[str]:
    if family == "false-colour":
        return ["clean", "false\ncolour"]
    if family == "high-pass":
        return ["clean"] + [s.rstrip("0").rstrip(".") for s in sev[1:]]
    return ["clean"] + [s.rstrip("0").rstrip(".") if "." in s else s for s in sev[1:]]


def load() -> dict[str, list[dict]]:
    out = {}
    for label, prefix, _c, _sb in CONDS:
        runs = [json.load(open(f)) for f in sorted(glob.glob(f"{OOD}/{prefix}*.json"))]
        if not runs:
            raise SystemExit(f"no runs matched {OOD}/{prefix}*.json")
        out[label] = runs
    return out


def relative(runs: list[dict], family: str) -> np.ndarray:
    """(seeds x severities) accuracy relative to each seed's own clean score.

    Identical to ood_verdict.py's `rel = acc / acc[:, :1]`; the AUC quoted in
    §2.6.1 is this matrix's row mean, so the curve and the number cannot drift.
    """
    acc = np.array([r["families"][family]["accuracy"] for r in runs], float)
    return acc / acc[:, :1]


def main():
    data = load()
    n_seeds = {k: len(v) for k, v in data.items()}
    print("runs loaded:", n_seeds)
    if set(n_seeds.values()) != {3}:
        raise SystemExit(f"expected 3 seeds per condition, got {n_seeds}")

    fig = plt.figure(figsize=(7.2, 4.6))
    gs = fig.add_gridspec(2, 4, hspace=0.78, wspace=0.50,
                          left=0.062, right=0.995, top=0.865, bottom=0.135)

    for k, fam in enumerate(FAMILIES):
        ax = fig.add_subplot(gs[k // 4, k % 4])
        sev = data["diet"][0]["families"][fam]["severities"]
        x = np.arange(len(sev))
        for label, _p, colour, _sb in CONDS:
            rel = relative(data[label], fam)
            for row in rel:
                ax.plot(x, row, color=colour, lw=0.5, alpha=0.32, zorder=2)
            ax.plot(x, rel.mean(0), color=colour, lw=1.6, zorder=3,
                    marker="o", ms=2.6, mew=0)
        ax.set_ylim(-0.03, 1.06)
        ax.set_xlim(-0.28, len(sev) - 0.72)
        ax.set_xticks(x)
        rot = 30 if len(sev) >= 7 else 0
        ax.set_xticklabels(sev_labels(sev, fam), fontsize=5.4,
                           rotation=rot,
                           ha="right" if rot else "center")
        ax.axhline(1.0, color="#CCCCCC", lw=0.6, ls=(0, (3, 3)), zorder=1)
        title = fam.replace("-", "-​")
        ax.set_title(title, pad=3.5, fontsize=8)
        if k % 4 == 0:
            ax.set_ylabel("accuracy /\nclean accuracy", fontsize=7.5)
        ax.set_yticks([0, 0.5, 1.0])

        # The two honest qualifications, marked on the panels they belong to.
        if fam == "low-pass":
            ax.text(0.5, 0.955, "trained in", transform=ax.transAxes,
                    ha="center", va="top", fontsize=6.2, color=DIET,
                    style="italic")
        if fam == "contrast":
            ax.text(0.5, 0.955, "trained in", transform=ax.transAxes,
                    ha="center", va="top", fontsize=6.2, color=GREY,
                    style="italic")
        if fam == "phase-scrambling":
            ax.text(0.5, 0.955, "equivalent, ±0.033", transform=ax.transAxes,
                    ha="center", va="top", fontsize=6.2, color=GREY, style="italic")

    # ---- (h) the summary: relative AUC per family, one marker per seed -------
    ax = fig.add_subplot(gs[1, 3])
    order = FAMILIES
    ys = np.arange(len(order))[::-1]
    for yi, fam in zip(ys, order):
        for label, _p, colour, _sb in CONDS[:2]:          # diet and no diet only
            auc = relative(data[label], fam).mean(1)
            ax.plot(auc, np.full_like(auc, yi, dtype=float), "o", ms=3.0,
                    color=colour, mew=0, alpha=0.9, zorder=3)
        a = relative(data["diet"], fam).mean(1)
        b = relative(data["no diet"], fam).mean(1)
        complete = a.min() > b.max() or b.min() > a.max()
        p = sps.mannwhitneyu(a, b, alternative="two-sided").pvalue
        ax.plot([min(a.min(), b.min()), max(a.max(), b.max())], [yi, yi],
                color="#DDDDDD", lw=0.7, zorder=1)
        print(f"  {fam:18s} diet {a.mean():.4f}  no diet {b.mean():.4f}  "
              f"{'complete sep' if complete else 'overlap':13s} p={p:.3f}")
        if complete:
            ax.text(0.985, yi, "✱", transform=ax.get_yaxis_transform(),
                    ha="right", va="center", fontsize=6.0, color=GREY)
    ax.set_yticks(ys)
    ax.set_yticklabels([f.replace("-", "-​") for f in order], fontsize=6.0)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel("relative-accuracy AUC", fontsize=7.5)
    ax.set_title("summary", pad=3.5, fontsize=8)
    ax.tick_params(axis="y", length=0)
    ax.text(1.0, -0.345, "✱ complete seed separation\n(p = 0.100, the exact 3v3 floor)",
            transform=ax.transAxes, ha="right", va="top", fontsize=5.6, color=GREY)

    handles = [plt.Line2D([], [], color=c, lw=1.6, marker="o", ms=3, mew=0, label=l)
               for l, _p, c, _sb in CONDS]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.058, 0.945),
               ncol=4, fontsize=7, handlelength=1.4, columnspacing=1.6)
    fig.text(0.058, 0.985, "Shape bias does not confer out-of-distribution robustness",
             fontsize=9.2, fontweight="bold", ha="left", va="center", color=INK)

    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        path = f"{OUT}/fig4_ood_degradation.{ext}"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        print("saved ->", path)


if __name__ == "__main__":
    main()
