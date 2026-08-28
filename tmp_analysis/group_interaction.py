"""Pre-declared group test (n=8): magno/parvo x cortical-axis dissociation.

Unit of inference is the SUBJECT. Per subject and ROI we take the median
noise-ceiling-normalized unique variance for each stream. Then:
  1. the INTERACTION: (U_m - U_p) on dorsal-leaning ROIs minus the same on
     ventral-leaning ROIs, tested with a sign-flip permutation across
     subjects;
  2. the two SIMPLE EFFECTS: magno vs parvo within each ROI set.
Nieuwenhuis et al. (2011): a double-dissociation claim needs the
interaction, not just two significant simple effects in opposite directions.
"""

from __future__ import annotations

import numpy as np

from src.analysis.encoding.stats import (
    STREAMS_LABELS, collect_group_table, fdr, interaction, load_results,
    simple_effect,
)

DORSAL = ["midparietal", "parietal"]
VENTRAL = ["midventral", "ventral"]

res = load_results("results/encoding", "untrained_seed0")
subjects, names, um, up = collect_group_table(
    res, "roi_streams", STREAMS_LABELS, nc_min=0.1)
print(f"n subjects = {len(subjects)}  {subjects}\n")

# ---- per-ROI simple effects -------------------------------------------
print("Per-ROI magno-minus-parvo unique variance (median over voxels, "
      "then across subjects)")
print(f"{'ROI':<13}{'mean(Um-Up)':>13}{'median':>10}{'W':>7}{'p':>9}"
      f"{'p_FDR<.05':>11}")
pvals, rows = [], []
for j, name in enumerate(names):
    se = simple_effect(um[:, j], up[:, j])
    d = um[:, j] - up[:, j]
    rows.append((name, float(np.mean(d)), se))
    pvals.append(se["p"])
rej = fdr(np.array(pvals), q=0.05)
for (name, mean_d, se), r in zip(rows, rej):
    print(f"{name:<13}{mean_d:>+13.4f}{se['median_diff']:>+10.4f}"
          f"{se['W']:>7.0f}{se['p']:>9.4f}{('yes' if r else 'no'):>11}")

# ---- the interaction ---------------------------------------------------
ia = np.array([names.index(n) for n in DORSAL])
ib = np.array([names.index(n) for n in VENTRAL])
res_ia = interaction(um, up, ia, ib, n_perm=100000, seed=0)
print(f"\nINTERACTION  dorsal{DORSAL} minus ventral{VENTRAL}")
print(f"  mean difference-of-differences = {res_ia['mean_dod']:+.4f}")
print(f"  sign-flip permutation p        = {res_ia['p_perm']:.5f}"
      f"   (2^8 = 256 distinct sign patterns -> p floor = {1/256:.4f})")
d = res_ia["per_subject"]
print(f"  per-subject: " + "  ".join(f"{v:+.4f}" for v in d))
print(f"  subjects with positive effect  = {int((d > 0).sum())}/{len(d)}")
w = simple_effect(np.nanmean(um[:, ia], 1) - np.nanmean(up[:, ia], 1),
                  np.nanmean(um[:, ib], 1) - np.nanmean(up[:, ib], 1))
print(f"  paired Wilcoxon on the same contrast: W={w['W']:.0f}, "
      f"p={w['p']:.4f}")
# Cohen's dz for the within-subject contrast
print(f"  Cohen's dz = {np.mean(d) / np.std(d, ddof=1):+.2f}")

# ---- simple effects on the two ROI sets --------------------------------
print("\nSimple effects on the pooled ROI sets (required for a double "
      "dissociation)")
for label, idx in (("dorsal", ia), ("ventral", ib)):
    se = simple_effect(np.nanmean(um[:, idx], 1), np.nanmean(up[:, idx], 1))
    sign = "magno>parvo" if se["median_diff"] > 0 else "parvo>magno"
    print(f"  {label:<8} median(Um-Up)={se['median_diff']:+.4f}  "
          f"W={se['W']:.0f}  p={se['p']:.4f}   {sign}")
