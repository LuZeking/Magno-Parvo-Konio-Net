"""Effect sizes for every headline paired contrast, because p is uninformative.

THE PROBLEM. With n=8 subjects the two-sided exact paired Wilcoxon has a floor
of p=.0078, reached whenever all 8 pairs agree in sign. Almost every contrast
in this project reaches it. PAPER_PLAN currently reports p=.0078 for the
cortical gradient, the PR set point, the in-silico cue-conflict dissociation,
the front-end orderings and the leg-(c) controls alike -- so the reported
statistic cannot distinguish an enormous effect from a merely consistent one,
and a reader has no way to tell which contrasts are load-bearing.

Note the matched-pairs rank-biserial correlation does NOT fix this: it is
+/-1.0 exactly when all pairs agree, so it saturates on the same contrasts the
p-value does. The quantity that separates them is the standardized paired
difference, Cohen's d_z = mean(diff) / sd(diff), together with an interval on
the raw difference in its own units.

WHAT THIS PRINTS. For every headline contrast: n, the raw mean difference with
a t-based 95% CI, d_z with a bootstrap CI over subjects, the Wilcoxon p, and
the sign count. Sorted by |d_z| so the ordering the p-values hide is visible.

DISCLOSURES.
  * n=8 makes the bootstrap CI on d_z coarse; it is reported to show the
    ordering is not fragile, not as a precise interval. Percentile bootstrap,
    10k resamples of SUBJECTS, seed fixed.
  * d_z is a within-subject standardized difference and is not comparable
    across instruments measured in different units -- compare d_z only within
    a block below, never across blocks.
  * For the PR block, n=8 is 8 image samples from 8 subjects' NSD stimuli, not
    8 independent brains (§47.5, §49.4). The same caveat that applies to the
    p-values applies to these intervals.
"""
from __future__ import annotations
import glob
import os

import numpy as np
from scipy import stats as sps

from src.analysis.encoding.stats import load_results

NC_MIN, MIN_VOX = 0.1, 20
VIS = [1, 2, 3, 4, 5, 6, 7]                      # V1v V1d V2v V2d V3v V3d hV4
HIER = np.array([1, 1, 2, 2, 3, 3, 4])
RNG = np.random.default_rng(0)
NBOOT = 10000


# --------------------------------------------------------------- statistics
def dz_ci(d: np.ndarray, nboot: int = NBOOT) -> tuple[float, float, float]:
    """Cohen's d_z with a percentile bootstrap CI over subjects."""
    dz = d.mean() / d.std(ddof=1)
    idx = RNG.integers(0, len(d), size=(nboot, len(d)))
    b = d[idx]
    sd = b.std(ddof=1, axis=1)
    keep = sd > 0
    bz = b.mean(axis=1)[keep] / sd[keep]
    return dz, float(np.percentile(bz, 2.5)), float(np.percentile(bz, 97.5))


def contrast(name: str, a: np.ndarray, b: np.ndarray) -> dict:
    """a and b are paired per-subject vectors; the difference is a - b."""
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    d = a - b
    n = len(d)
    dz, lo, hi = dz_ci(d)
    se = d.std(ddof=1) / np.sqrt(n)
    t = sps.t.ppf(0.975, n - 1)
    p = sps.wilcoxon(d)[1] if n > 0 else np.nan
    return dict(name=name, n=n, mean=d.mean(), sd=d.std(ddof=1),
                ci=(d.mean() - t * se, d.mean() + t * se),
                dz=dz, dz_ci=(lo, hi), p=p,
                agree=int((d > 0).sum()) if d.mean() > 0 else int((d < 0).sum()))


def show(block: str, rows: list[dict], unit: str) -> None:
    print("\n" + "=" * 108)
    print(f"{block}   (difference in {unit}; d_z comparable WITHIN this block only)")
    print("=" * 108)
    print(f"{'contrast':44s} {'n':>2s} {'mean diff':>10s} {'95% CI':>19s} "
          f"{'d_z':>7s} {'d_z 95% CI':>17s} {'p':>7s} {'sign':>5s}")
    for r in sorted(rows, key=lambda r: -abs(r["dz"])):
        print(f"{r['name']:44s} {r['n']:2d} {r['mean']:+10.4f} "
              f"[{r['ci'][0]:+8.4f},{r['ci'][1]:+8.4f}] {r['dz']:+7.2f} "
              f"[{r['dz_ci'][0]:+7.2f},{r['dz_ci'][1]:+7.2f}] {r['p']:7.4f} "
              f"{r['agree']:2d}/{r['n']}")


# --------------------------------------------------- leg (c): cortical profile
def spi_profile(model: str, stat: str = "norm") -> tuple[np.ndarray, np.ndarray]:
    """Per-subject stream-preference profile across V1v..hV4.

    TWO statistics exist in this codebase and they are NOT interchangeable:

      stat="norm"  median(Um - Up) / median(Um + Up)   -- a ratio of medians,
                   guarded at den > 1e-4. This is what §48 and the manuscript
                   report, implemented in tmp_analysis/apply_repair.py.
      stat="pvr"   median over voxels of (Um - Up)/(Um + Up)  -- a median of
                   per-voxel ratios, with a per-voxel (Um+Up) > 1e-3 filter.
                   This is tmp_analysis/visualrois.py's `spi`.

    They agree closely on the diet arm and diverge on the degenerate control
    arms, where one stream contributes almost no unique variance so per-voxel
    ratios saturate at +/-1 (§48.2). "norm" is primary here because it is the
    published statistic; 55.3 reports the other as a robustness check.
    """
    res = load_results("results/encoding", model)
    subs = sorted(res)
    prof = np.full((len(subs), len(VIS)), np.nan)
    for i, s in enumerate(subs):
        z = res[s]
        roi, nc = z["roi_prf-visualrois"], z["noise_ceiling"]
        um, up = z["unique_magno"], z["unique_parvo"]
        rel = (nc > NC_MIN) & np.isfinite(um) & np.isfinite(up)
        for j, lab in enumerate(VIS):
            if stat == "norm":
                sel = (roi == lab) & rel
                if sel.sum() >= MIN_VOX:
                    den = np.median(um[sel] + up[sel])
                    if den > 1e-4:
                        prof[i, j] = np.median(um[sel] - up[sel]) / den
            else:
                v = (roi == lab) & (nc > NC_MIN) & ((um + up) > 1e-3)
                if v.sum() >= MIN_VOX:
                    r = (um[v] - up[v]) / (um[v] + up[v])
                    prof[i, j] = np.median(r[np.isfinite(r)])
    return np.array(subs), prof


def grad_and_rho(prof: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    grad = prof[:, 6] - prof[:, 0]                       # hV4 - V1v
    rho = np.array([sps.spearmanr(HIER, p)[0] for p in prof])
    return grad, rho


CORTEX = {
    "diet":         "dkl_bcs_seed0_trained",
    "no diet":      "dkl_base_seed0_trained",
    "swap":         "dkl_bcs_swap_seed0_trained",
    "parvo_highsf": "dkl_bcs_parvo_highsf_seed0_trained",
}
G, R, G2, R2 = {}, {}, {}, {}
for k, m in CORTEX.items():
    _, prof = spi_profile(m, "norm")
    G[k], R[k] = grad_and_rho(prof)
    _, prof2 = spi_profile(m, "pvr")
    G2[k], R2[k] = grad_and_rho(prof2)

print("=" * 108)
print("REPRODUCTION CHECK against §48, `norm` statistic")
print("=" * 108)
print(f"{'arm':14s} {'rho':>9s} {'§48':>9s} {'hV4-V1v':>10s} {'§48':>10s}")
REF = {"diet": (-0.964, -0.8618), "no diet": (-0.502, -0.1013),
       "swap": (-0.259, -0.0265), "parvo_highsf": (-0.780, -0.1186)}
for k in CORTEX:
    print(f"{k:14s} {R[k].mean():+9.4f} {REF[k][0]:+9.3f} "
          f"{G[k].mean():+10.4f} {REF[k][1]:+10.4f}")

rows = [contrast(f"hV4-V1v: diet vs {k}", G["diet"], G[k])
        for k in ("no diet", "swap", "parvo_highsf")]
show("LEG (c) CORTICAL GRADIENT, hV4-V1v  (`norm`, the published statistic)",
     rows, "norm units")

rows = [contrast(f"rho: diet vs {k}", R["diet"], R[k])
        for k in ("no diet", "swap", "parvo_highsf")]
show("LEG (c) HIERARCHY RANK CORRELATION, Spearman rho  (`norm`)", rows, "rho units")

rows = [contrast(f"rho: diet vs {k}", R2["diet"], R2[k])
        for k in ("no diet", "swap", "parvo_highsf")]
show("ROBUSTNESS: the same rho contrasts under the per-voxel-ratio statistic",
     rows, "rho units")

# ------------------------------------------------------------ PR trained ratio
STORE = os.path.join(os.environ.get("CLAUDE_JOB_DIR", "/tmp"), "tmp", "pr_store.npz")
_pr = dict(np.load(STORE))


def pr_ratio(model: str) -> np.ndarray:
    d = {int(k.partition("|subj")[2]): v[0] / v[1]
         for k, v in _pr.items() if k.partition("|subj")[0] == model}
    return np.array([d[s] for s in sorted(d)])


PRM = {
    "diet/lum":    "dkl_bcs_seed0_trained",
    "diet/highsf": "dkl_bcs_parvo_highsf_seed0_trained",
    "diet/rg":     "dkl_bcs_swap_seed0_trained",
    "nodiet/s0":   "dkl_base_seed0_trained",
    "nodiet/s1":   "dkl_base_seed1_trained",
}
P = {k: pr_ratio(m) for k, m in PRM.items()}
rows = []
for a in ("diet/lum", "diet/highsf", "diet/rg"):
    for b in ("nodiet/s0", "nodiet/s1"):
        rows.append(contrast(f"PR trained: {a} vs {b}", P[a], P[b]))
rows.append(contrast("PR trained: diet/rg vs diet/lum", P["diet/rg"], P["diet/lum"]))
rows.append(contrast("PR trained: diet/rg vs diet/highsf", P["diet/rg"], P["diet/highsf"]))
rows.append(contrast("PR trained: diet/highsf vs diet/lum", P["diet/highsf"], P["diet/lum"]))
show("PR TRAINED magno/parvo RATIO", rows, "ratio units")

# -------------------------------------- in-silico cue conflict and the r2 cost
z1 = np.load("results/figures/fig1_data.npz")
rows = []
if "b__diet__magno" in z1.files:
    dcc = z1["b__diet__magno"] - z1["b__diet__parvo"]
    ncc = z1["b__nodiet__magno"] - z1["b__nodiet__parvo"]
    rows.append(contrast("in-silico VSB magno-parvo: diet vs no diet", dcc, ncc))
    show("IN-SILICO CUE CONFLICT THROUGH THE ENCODING MODEL", rows, "VSB units")
rows = []
if "d__diet__r2nc" in z1.files:
    rows.append(contrast("r2/NC: diet vs no diet",
                         z1["d__diet__r2nc"], z1["d__nodiet__r2nc"]))
    show("THE VARIANCE COST", rows, "r2/NC units")

# ----------------------------------------------------------------- the summary
print("\n" + "=" * 108)
print("WHY THIS SECTION EXISTS")
print("=" * 108)
allrows = []
for k in ("no diet", "swap", "parvo_highsf"):
    allrows.append(contrast(f"hV4-V1v diet vs {k}", G["diet"], G[k]))
for a in ("diet/lum", "diet/rg"):
    allrows.append(contrast(f"PR {a} vs nodiet/s0", P[a], P["nodiet/s0"]))
floor = [r for r in allrows if abs(r["p"] - 0.0078125) < 1e-6]
if floor:
    dzs = [abs(r["dz"]) for r in floor]
    print(f"  {len(floor)} of these {len(allrows)} contrasts report the IDENTICAL p = .0078,")
    print(f"  the exact two-sided floor for n=8. Their |d_z| ranges from "
          f"{min(dzs):.2f} to {max(dzs):.2f}")
    print(f"  -- a factor of {max(dzs)/min(dzs):.1f}. The p-value is constant across that range")
    print("  by construction and therefore carries no information about magnitude.")
print("\n  RECOMMENDATION for the manuscript: report the raw difference with its")
print("  95% CI as the primary quantity, d_z as the standardized effect, and give")
print("  the p-value once with a footnote that .0078 is the n=8 exact floor and")
print("  that many contrasts attain it. Do not rank claims by p, and do not use")
print("  rank-biserial r either -- it saturates at 1.0 on the same contrasts.")
