"""§77: equivalence bounds for the paper's three dissociations.

Answers the NHB referee's major comment 4 -- "state, for each, what difference
the design could have detected and what interval the data exclude" -- with the
bounds fixed in §77.3 and committed (03ca8fc) BEFORE any interval here was
computed.

Every SESOI is anchored to a different already-published quantity, so knowing a
point estimate cannot move it. The bounds are final: §77.5 forbids rewidening
one after seeing its interval.

    PYTHONPATH=. python tmp_analysis/equivalence_tests.py
"""
from __future__ import annotations

import csv
import glob
import itertools
import os
import re

import numpy as np
import scipy.stats as sps

from src.analysis.behavior.cue_conflict import (
    GEIRHOS_CLASSES, load_class_index, list_cue_conflict_images,
)

# ---- §77.3 bounds, fixed before computing. DO NOT EDIT AFTER SEEING OUTPUT --
SESOI_PHASE = 0.041     # rotation, smallest OOD effect the paper claims
SESOI_KAPPA = 0.0234    # 10% of the 0.234 human-human ceiling
SESOI_SLOPE = 0.1625    # half the ceiling over the observed 0.720 span
SESOI_VO = 0.0195       # hMT+, smallest ROI effect this measure claims
ALPHA = 0.05

HUMAN_DIR = "/scratch/byrong/VST_data/geirhos_human_data/cue_conflict"
CC = "results/cue_conflict"
OOD = "results/ood_bestep"
STEM = re.compile(r"([a-z]+)\d+-([a-z]+)\d+$")


# ------------------------------------------------------------- TOST engine --
def tost_ind(a, b, sesoi):
    """Welch TOST for two independent groups. Returns dict."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    diff = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
    df = se**4 / (
        (a.var(ddof=1) / a.size) ** 2 / (a.size - 1)
        + (b.var(ddof=1) / b.size) ** 2 / (b.size - 1)
    )
    return _tost_from(diff, se, df, sesoi)


def tost_paired(d, sesoi):
    """Paired TOST on the vector of differences."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    se = d.std(ddof=1) / np.sqrt(d.size)
    return _tost_from(d.mean(), se, d.size - 1, sesoi)


def _tost_from(diff, se, df, sesoi):
    # THIS INTERVAL IS PARAMETRIC; THE PAPER'S DECLARED TEST IS NOT.
    # PAPER_DRAFT.md:1351 declares the paired exact Wilcoxon signed-rank test
    # at n=8. Use this to BOUND a null -- how large an effect could the data
    # have hidden -- which is existing practice (§2.6.1, §2.6.2) and fine.
    # Do NOT use it to overturn a Wilcoxon verdict: at n=8 the two disagree
    # around alpha (§103 found t-p ~.075 against Wilcoxon .1094 on the same
    # data, which briefly turned two accepted nulls into "significant"
    # findings). For an interval that can speak to the verdict, use
    # Hodges-Lehmann with the Walsh-average CI -- exact at n=8 and consistent
    # with Wilcoxon by construction (tmp_analysis/prf_hodges_lehmann.py).
    # 90% CI is the interval whose containment in +-sesoi <=> both one-sided
    # tests reject at alpha=.05.
    tcrit = sps.t.ppf(1 - ALPHA, df)
    lo, hi = diff - tcrit * se, diff + tcrit * se
    p_lower = sps.t.sf((diff + sesoi) / se, df)      # H0: diff <= -sesoi
    p_upper = sps.t.cdf((diff - sesoi) / se, df)     # H0: diff >= +sesoi
    p_tost = max(p_lower, p_upper)
    return dict(diff=diff, se=se, df=df, ci=(lo, hi), p_tost=p_tost,
                equivalent=(lo > -sesoi and hi < sesoi), sesoi=sesoi,
                ci_halfwidth=tcrit * se)


def mde_ind(a, b, power=0.80):
    """Smallest raw difference detectable at 80% power, two-sample t."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = a.size
    sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    df = 2 * n - 2
    tcrit = sps.t.ppf(1 - ALPHA / 2, df)
    for d in np.arange(0.01, 30, 0.01):
        ncp = d * np.sqrt(n / 2)
        pw = sps.nct.sf(tcrit, df, ncp) + sps.nct.cdf(-tcrit, df, ncp)
        if pw >= power:
            return d * sd, d
    return np.nan, np.nan


def mde_paired(d, power=0.80):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    n, sd, df = d.size, d.std(ddof=1), d.size - 1
    tcrit = sps.t.ppf(1 - ALPHA / 2, df)
    for dz in np.arange(0.01, 30, 0.01):
        ncp = dz * np.sqrt(n)
        pw = sps.nct.sf(tcrit, df, ncp) + sps.nct.cdf(-tcrit, df, ncp)
        if pw >= power:
            return dz * sd, dz
    return np.nan, np.nan


def show(name, r, mde, unit=""):
    lo, hi = r["ci"]
    verdict = ("EQUIVALENT" if r["equivalent"]
               else "NOT equivalent -- interval exceeds the bound")
    print(f"\n  {name}")
    print(f"    difference      {r['diff']:+.4f}{unit}")
    print(f"    90% CI          [{lo:+.4f}, {hi:+.4f}]  (half-width "
          f"{r['ci_halfwidth']:.4f}, df={r['df']:.1f})")
    print(f"    SESOI           +-{r['sesoi']:.4f}   (§77.3, fixed in 03ca8fc)")
    print(f"    TOST p          {r['p_tost']:.4f}")
    print(f"    detectable@80%  {mde:.4f}  (the effect this design could find)")
    print(f"    -> {verdict}")
    if not r["equivalent"]:
        ratio = r["ci_halfwidth"] / r["sesoi"]
        print(f"       the CI is {ratio:.1f}x the bound; the data cannot "
              f"exclude an effect as")
        print(f"       large as the smallest one the paper claims.")


# ------------------------------------------------------ 1. phase-scrambling --
def rel_auc(prefix, family):
    out = []
    for f in sorted(glob.glob(f"{OOD}/{prefix}*.json")):
        import json
        acc = np.array(json.load(open(f))["families"][family]["accuracy"],
                       float)
        out.append((acc / acc[0]).mean())
    return np.array(out)


# ---------------------------------------------------------- 2/3. kappa ------
def kappa(a, b):
    m = np.array([x is not None and y is not None for x, y in zip(a, b)])
    if m.sum() < 30:
        return np.nan
    x = np.array([v == "S" for v in np.asarray(a, dtype=object)[m]])
    y = np.array([v == "S" for v in np.asarray(b, dtype=object)[m]])
    c_obs = float((x == y).mean())
    px, py = float(x.mean()), float(y.mean())
    c_exp = px * py + (1 - px) * (1 - py)
    if c_exp >= 1.0:
        return np.nan
    return (c_obs - c_exp) / (1 - c_exp)


def main() -> None:
    print("=" * 78)
    print("§77 EQUIVALENCE TESTS -- bounds fixed in 03ca8fc before computing")
    print("=" * 78)

    # ---------------------------------------------------------- test 1 -----
    print("\n" + "-" * 78)
    print("TEST 1  §2.5 phase-scrambling: diet vs no diet, relative AUC")
    print("-" * 78)
    d = rel_auc("dkl_bcs_seed", "phase-scrambling")
    nd = rel_auc("dkl_base_seed", "phase-scrambling")
    print(f"  diet    {np.round(d, 4).tolist()}  mean {d.mean():.4f}")
    print(f"  no diet {np.round(nd, 4).tolist()}  mean {nd.mean():.4f}")
    r = tost_ind(d, nd, SESOI_PHASE)
    m, _ = mde_ind(d, nd)
    show("phase-scrambling, diet - no diet", r, m)
    print("\n    NOTE (§77.4): at 3v3 the exact rank-sum floors at p=0.10, so no")
    print("    NONPARAMETRIC equivalence result is possible at any bound. The")
    print("    interval above is t-based on 3 seeds per arm and is fragile.")

    # ---------------------------------------------------- kappa pipeline ---
    images = list_cue_conflict_images()
    stems = [os.path.splitext(os.path.basename(p))[0] for p, _, _ in images]
    shapes = [s for _, s, _ in images]
    textures = [t for _, _, t in images]
    index = {st: i for i, st in enumerate(stems)}

    humans = {}
    for f in sorted(glob.glob(f"{HUMAN_DIR}/*subject-*.csv")):
        subj = re.search(r"(subject-\d+)", os.path.basename(f)).group(1)
        choice = [None] * len(stems)
        with open(f) as fh:
            for row in csv.DictReader(fh):
                mm = STEM.search(os.path.splitext(row["imagename"])[0])
                if not mm:
                    continue
                i = index.get(mm.group(0))
                if i is None:
                    continue
                rr = row["object_response"]
                if rr == shapes[i]:
                    choice[i] = "S"
                elif rr == textures[i]:
                    choice[i] = "T"
        humans[subj] = choice

    hh = np.array([kappa(humans[a], humans[b])
                   for a, b in itertools.combinations(sorted(humans), 2)],
                  dtype=float)
    ceil = float(np.nanmean(hh))
    print("\n" + "-" * 78)
    print("TEST 2  §2.6 error consistency: diet vs no diet, whole model")
    print("-" * 78)
    print(f"  human-human ceiling kappa = {ceil:.3f} "
          f"({np.isfinite(hh).sum()} pairs)   [draft says 0.234]")
    assert abs(ceil - 0.234) < 0.002, f"ceiling {ceil} != published 0.234"

    cls = load_class_index()
    cols = [cls[c] for c in GEIRHOS_CLASSES]
    sh_a = np.asarray(shapes, dtype=object)
    tx_a = np.asarray(textures, dtype=object)

    def choices(model, readout):
        z = np.load(f"{CC}/{model}_logits.npz", allow_pickle=True)
        if readout not in z.files:
            return None
        dec = np.asarray(GEIRHOS_CLASSES, dtype=object)[
            z[readout][:, cols].argmax(1)]
        out = np.full(len(stems), None, dtype=object)
        out[dec == sh_a] = "S"
        out[dec == tx_a] = "T"
        return list(out)

    def k_and_sb(model, readout):
        ch = choices(model, readout)
        if ch is None:
            return np.nan, np.nan
        k = float(np.nanmean([kappa(ch, humans[h]) for h in sorted(humans)]))
        nS = sum(c == "S" for c in ch)
        nT = sum(c == "T" for c in ch)
        return k, (nS / (nS + nT) if (nS + nT) else np.nan)

    diet = [f"dkl_bcs_seed{i}" for i in (0, 1, 2)]
    base = [f"dkl_base_seed{i}" for i in (0, 1, 2)]
    kd = np.array([k_and_sb(m, "full")[0] for m in diet])
    kb = np.array([k_and_sb(m, "full")[0] for m in base])
    print(f"  diet    kappa {np.round(kd, 4).tolist()}  mean {kd.mean():.4f}"
          f"  ({100*kd.mean()/ceil:.1f}% of ceiling)")
    print(f"  no diet kappa {np.round(kb, 4).tolist()}  mean {kb.mean():.4f}"
          f"  ({100*kb.mean()/ceil:.1f}% of ceiling)")
    r2 = tost_ind(kd, kb, SESOI_KAPPA)
    m2, _ = mde_ind(kd, kb)
    show("kappa, diet - no diet (whole model)", r2, m2)

    # ---------------------------------------------------------- test 3 -----
    print("\n" + "-" * 78)
    print("TEST 3  §2.6 does shape bias track kappa? slope across models")
    print("-" * 78)
    pts = []
    for lab, models, readout in [
        ("diet/full", diet, "full"), ("nodiet/full", base, "full"),
        ("diet/magno", diet, "magno_only"),
        ("nodiet/magno", base, "magno_only"),
    ]:
        ks, sbs = zip(*[k_and_sb(m, readout) for m in models])
        pts.append((lab, float(np.mean(sbs)), float(np.mean(ks))))
    # Published Geirhos models (§2.6 table) -- shape bias, kappa vs human.
    for lab, sb, k in [("AlexNet", 0.429, 0.032), ("GoogLeNet", 0.312, 0.025),
                       ("VGG", 0.172, 0.017)]:
        pts.append((lab, sb, k))
    print(f"  {'model':14s} {'shape bias':>11s} {'kappa':>8s} {'% ceil':>8s}")
    for lab, sb, k in pts:
        print(f"  {lab:14s} {sb:11.3f} {k:8.4f} {100*k/ceil:7.1f}%")
    sb_v = np.array([p[1] for p in pts])
    k_v = np.array([p[2] for p in pts])
    n = len(pts)
    slope, icept, rr, pp, se_s = sps.linregress(sb_v, k_v)
    tcrit = sps.t.ppf(1 - ALPHA, n - 2)
    lo, hi = slope - tcrit * se_s, slope + tcrit * se_s
    span = sb_v.max() - sb_v.min()
    print(f"\n  n={n} models, shape-bias span {sb_v.min():.3f}-{sb_v.max():.3f}"
          f" ({span:.3f})")
    print(f"  OLS slope       {slope:+.4f} kappa per unit shape bias "
          f"(r={rr:+.3f}, p={pp:.3f})")
    print(f"  90% CI          [{lo:+.4f}, {hi:+.4f}]")
    print(f"  SESOI           +-{SESOI_SLOPE:.4f}  (§77.3)")
    equiv = lo > -SESOI_SLOPE and hi < SESOI_SLOPE
    print(f"  -> {'EQUIVALENT: slope bounded inside the SESOI' if equiv else 'NOT equivalent'}")
    print(f"\n  Translated to the referee's question: across the observed "
          f"{span:.3f} shape-bias")
    print(f"  span the data exclude a kappa change larger than "
          f"{max(abs(lo), abs(hi))*span:+.4f},")
    print(f"  i.e. {100*max(abs(lo), abs(hi))*span/ceil:.1f}% of the "
          f"human-human ceiling.")

    # ---------------------------------------------------------- test 4 -----
    print("\n" + "-" * 78)
    print("TEST 4  §2.3 VO median(U_m - U_p), n=8 subjects")
    print("-" * 78)
    vo = vo_per_subject()
    print(f"  per-subject {np.round(vo, 4).tolist()}")
    print(f"  mean {vo.mean():+.4f}   {int((vo > 0).sum())}/{vo.size} positive")
    W, pw = sps.wilcoxon(vo)
    print(f"  Wilcoxon p={pw:.4f}   [draft reports p=.46]")
    r4 = tost_paired(vo, SESOI_VO)
    m4, _ = mde_paired(vo)
    show("VO median(U_m - U_p) vs 0", r4, m4)

    # Nonparametric companion: BCa-free percentile bootstrap.
    rng = np.random.default_rng(0)
    bs = np.array([rng.choice(vo, vo.size, replace=True).mean()
                   for _ in range(20000)])
    blo, bhi = np.percentile(bs, [5, 95])
    print(f"\n    bootstrap 90% CI [{blo:+.4f}, {bhi:+.4f}] (20k resamples) -> "
          f"{'inside' if blo > -SESOI_VO and bhi < SESOI_VO else 'NOT inside'} "
          f"the bound")

    print("\n" + "=" * 78)


def vo_per_subject() -> np.ndarray:
    """median(U_m - U_p) in VO1+VO2, exactly as tmp_analysis/kastner_rois.py.

    The constants are inlined rather than imported: kastner_rois.py runs its
    whole analysis at module level, so importing it would re-execute that.
    """
    from src.analysis.encoding import nsd_data
    from src.analysis.encoding.stats import load_results

    KASTNER = {1: "V1v", 2: "V1d", 3: "V2v", 4: "V2d", 5: "V3v", 6: "V3d",
               7: "hV4", 8: "VO1", 9: "VO2", 10: "PHC1", 11: "PHC2",
               12: "TO2", 13: "TO1", 14: "LO2", 15: "LO1", 16: "V3B",
               17: "V3A", 18: "IPS0", 19: "IPS1", 20: "IPS2", 21: "IPS3",
               22: "IPS4", 23: "IPS5", 24: "SPL1", 25: "FEF"}
    NC_MIN, MIN_VOX = 0.1, 20

    res = load_results("results/encoding", "untrained_seed0")
    out = []
    for s in sorted(res):
        z = res[s]
        mask = nsd_data.load_mask(s, "nsdgeneral")
        kl = nsd_data.load_roi_labels(s, "Kastner2015", mask)
        nc = z["noise_ceiling"]
        d = z["unique_magno"] - z["unique_parvo"]
        ok = (nc > NC_MIN) & np.isfinite(d)
        vals = []
        for lab in ("VO1", "VO2"):
            j = [k for k in sorted(KASTNER) if KASTNER[k] == lab][0]
            v = ok & (kl == j)
            vals.append(np.median(d[v]) if v.sum() >= MIN_VOX else np.nan)
        out.append(np.nanmean(vals))
    return np.array(out, float)


if __name__ == "__main__":
    main()
