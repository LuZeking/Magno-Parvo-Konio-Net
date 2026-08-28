"""Figure: where each input-defined stream uniquely explains cortical variance.

Panel A  stream preference across eccentricity (the headline)
Panel B  stream preference across retinotopic areas V1 -> hV4
Panel C  the pre-declared dorsal-vs-ventral interaction, one dot per subject

Every panel plots all 8 subjects individually, because the subject is the
unit of inference and group means alone would hide whether the effect is
carried by everyone or by two outliers.

Usage:
  python tmp_analysis/fig_streams.py [--model untrained_seed0]
                                     [--compare untrained_seed0_noblur]
The --compare overlay is for the blur control: if the eccentricity gradient
is a restatement of the magno stream's low-pass front end, it flattens
there, and the two sets of lines will separate in panel A.
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sps

from src.analysis.encoding.stats import ECC_LABELS, load_results

VIS = {1: "V1v", 2: "V1d", 3: "V2v", 4: "V2d", 5: "V3v", 6: "V3d", 7: "hV4"}
STREAMS = {1: "early", 2: "midventral", 3: "midlateral", 4: "midparietal",
           5: "ventral", 6: "lateral", 7: "parietal"}
BAND_CENTRE = {1: 0.25, 2: 0.75, 3: 1.5, 4: 3.0}
NC_MIN = 0.1

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="untrained_seed0")
ap.add_argument("--compare", default=None)
ap.add_argument("--out", default="results/figures/fig_streams.png")
args = ap.parse_args()


def spi_table(res, roi_key, labels):
    """Median SPI per (subject, ROI)."""
    subs = sorted(res)
    out = np.full((len(subs), len(labels)), np.nan)
    for i, s in enumerate(subs):
        z = res[s]
        roi, nc = z[roi_key], z["noise_ceiling"]
        um, up = z["unique_magno"], z["unique_parvo"]
        for j, lab in enumerate(sorted(labels)):
            v = (roi == lab) & (nc > NC_MIN) & ((um + up) > 1e-3)
            if v.sum() >= 20:
                r = (um[v] - up[v]) / (um[v] + up[v])
                out[i, j] = np.median(r[np.isfinite(r)])
    return subs, out


def diff_table(res, roi_key, labels):
    """Median (Um - Up) per (subject, ROI): no ratio degeneracy."""
    subs = sorted(res)
    out = np.full((len(subs), len(labels)), np.nan)
    for i, s in enumerate(subs):
        z = res[s]
        roi, nc = z[roi_key], z["noise_ceiling"]
        d = z["unique_magno"] - z["unique_parvo"]
        for j, lab in enumerate(sorted(labels)):
            v = (roi == lab) & (nc > NC_MIN)
            if v.sum() >= 20:
                out[i, j] = np.median(d[v][np.isfinite(d[v])])
    return subs, out


res = load_results("results/encoding", args.model)
if not res:
    raise SystemExit(f"no results for {args.model}")
res_c = load_results("results/encoding", args.compare) if args.compare else {}

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1))
C_MAIN, C_CMP = "#1f4e79", "#c0504d"

# ---- Panel A: eccentricity ---------------------------------------------
ax = axes[0]
x = np.log([BAND_CENTRE[b] for b in sorted(ECC_LABELS)])
subs, tab = spi_table(res, "roi_prf-eccrois", ECC_LABELS)
for row in tab:
    ax.plot(x, row, color=C_MAIN, alpha=0.28, lw=1)
ax.plot(x, np.nanmean(tab, 0), color=C_MAIN, lw=2.6, marker="o",
        label=f"{args.model} (n={len(subs)})", zorder=5)
if res_c:
    _, tabc = spi_table(res_c, "roi_prf-eccrois", ECC_LABELS)
    for row in tabc:
        ax.plot(x, row, color=C_CMP, alpha=0.28, lw=1)
    ax.plot(x, np.nanmean(tabc, 0), color=C_CMP, lw=2.6, marker="s",
            label=f"{args.compare}", zorder=5)
ax.axhline(0, color="0.4", lw=0.8, ls="--")
ax.set_xticks(x)
ax.set_xticklabels([ECC_LABELS[b].replace("ecc_", "") + "°"
                    for b in sorted(ECC_LABELS)])
ax.set_xlabel("pRF eccentricity band")
ax.set_ylabel("stream preference  (U$_m$−U$_p$)/(U$_m$+U$_p$)")
ax.set_title("A  fovea → periphery", loc="left", fontweight="bold")
ax.legend(fontsize=7, frameon=False, loc="lower right")
ax.text(0.03, 0.95, "magno", transform=ax.transAxes, fontsize=8, color="0.3",
        va="top")
ax.text(0.03, 0.05, "parvo", transform=ax.transAxes, fontsize=8, color="0.3",
        va="bottom")

# ---- Panel B: V1 -> hV4 -------------------------------------------------
ax = axes[1]
subs, tabv = spi_table(res, "roi_prf-visualrois", VIS)
xs = np.arange(len(VIS))
for row in tabv:
    ax.plot(xs, row, color=C_MAIN, alpha=0.28, lw=1)
ax.plot(xs, np.nanmean(tabv, 0), color=C_MAIN, lw=2.6, marker="o", zorder=5)
if res_c:
    _, tabvc = spi_table(res_c, "roi_prf-visualrois", VIS)
    for row in tabvc:
        ax.plot(xs, row, color=C_CMP, alpha=0.28, lw=1)
    ax.plot(xs, np.nanmean(tabvc, 0), color=C_CMP, lw=2.6, marker="s",
            zorder=5)
ax.axhline(0, color="0.4", lw=0.8, ls="--")
ax.set_xticks(xs)
ax.set_xticklabels([VIS[k] for k in sorted(VIS)], rotation=45)
ax.set_xlabel("retinotopic area")
ax.set_title("B  V1 → hV4", loc="left", fontweight="bold")

# ---- Panel C: the pre-declared interaction ------------------------------
ax = axes[2]
names = list(STREAMS.values())
subs, dtab = diff_table(res, "roi_streams", STREAMS)
ia = [names.index(n) for n in ("midparietal", "parietal")]
ib = [names.index(n) for n in ("midventral", "ventral")]
dor, ven = np.nanmean(dtab[:, ia], 1), np.nanmean(dtab[:, ib], 1)
for a, b in zip(dor, ven):
    ax.plot([0, 1], [b, a], color="0.55", lw=1, marker="o", ms=3.5,
            alpha=0.75)
ax.plot([0, 1], [np.nanmean(ven), np.nanmean(dor)], color=C_MAIN, lw=2.8,
        marker="o", ms=7, zorder=5)
d = dor - ven
d = d[np.isfinite(d)]
W, p = sps.wilcoxon(d)
ax.axhline(0, color="0.4", lw=0.8, ls="--")
ax.set_xticks([0, 1])
ax.set_xticklabels(["ventral\n(midventral,\nventral)",
                    "dorsal\n(midparietal,\nparietal)"])
ax.set_ylabel("U$_m$ − U$_p$   (noise-ceiling normalized)")
ax.set_title("C  stream × axis interaction", loc="left", fontweight="bold")
ax.text(0.5, 0.03,
        f"Δ={d.mean():+.4f}, p={p:.3f}\n{int((d > 0).sum())}/{len(d)} subjects",
        transform=ax.transAxes, ha="center", fontsize=8)

for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
os.makedirs(os.path.dirname(args.out), exist_ok=True)
fig.savefig(args.out, dpi=200)
fig.savefig(args.out.replace(".png", ".pdf"))
print("wrote", args.out)
