"""Exact permutation null for §2.2(b)'s hierarchical-rank monotonicity.

§2.2(b) reports a per-subject Spearman rho of the V1v->hV4 stream-preference
profile against hierarchical rank (HIER = [1,1,2,2,3,3,4]), -0.964 with the
diet against -0.536 without. §2.2 also discloses that this gradient is the
survivor of seven regional hypotheses. A null cannot undo post-hoc selection --
that disclosure stays exactly where it is -- but it does answer a different and
answerable question: how far does the observed rho exceed what an arbitrary
labelling of the same seven ROIs would give?

The test is EXACT, not sampled. HIER has ties (1,1,2,2,3,3,4), so the seven
labels admit 7!/(2!2!2!1!) = 630 distinct assignments. All 630 are enumerated.
The same permuted labelling is applied to every subject, because the ROI labels
are shared across subjects: the question is whether the true labelling is
special, not whether one subject's profile is.

Both statistics are reported (§8's two-statistics rule).

Usage: PYTHONPATH=.:tmp_analysis python tmp_analysis/hierarchy_permutation_null.py
"""
from __future__ import annotations

import itertools, json

import numpy as np
import scipy.stats as sps

from src.analysis.encoding.stats import load_results

VIS = [1, 2, 3, 4, 5, 6, 7]                    # V1v V1d V2v V2d V3v V3d hV4
NAMES = ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4"]
HIER = np.array([1, 1, 2, 2, 3, 3, 4])
NC_MIN, MIN_VOX = 0.1, 20
SRC = "results/encoding_cc"


def spi_profile(model: str, stat: str, src: str = SRC):
    """Identical construction to tmp_analysis/effect_sizes.py:spi_profile."""
    res = load_results(src, model)
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


def distinct_labelings(hier: np.ndarray) -> list[tuple]:
    return sorted(set(itertools.permutations(hier.tolist())))


def mean_rho(prof: np.ndarray, labels) -> float:
    lab = np.asarray(labels, dtype=float)
    return float(np.nanmean([sps.spearmanr(lab, p)[0] for p in prof]))


def run(model: str, stat: str) -> dict | None:
    subs, prof = spi_profile(model, stat)
    if prof.size == 0 or np.all(np.isnan(prof)):
        return None
    perms = distinct_labelings(HIER)
    obs = mean_rho(prof, HIER)
    null = np.array([mean_rho(prof, p) for p in perms])
    # one-sided: the pre-stated direction is a DECLINE with hierarchical rank
    n_le = int((null <= obs + 1e-12).sum())
    p = n_le / len(perms)
    return {
        "model": model, "stat": stat, "n_subjects": int(len(subs)),
        "observed_mean_rho": obs, "n_permutations": len(perms),
        "p_exact_one_sided": p,
        "null_min": float(null.min()), "null_max": float(null.max()),
        "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)),
        "null_q025": float(np.quantile(null, 0.025)),
        "rank_of_observed": n_le,
        "is_most_extreme": bool(n_le == 1),
    }


def main():
    models = {
        "diet":    ["dkl_bcs_seed0_trained", "dkl_bcs_seed1_trained",
                    "dkl_bcs_seed2_trained"],
        "no diet": ["dkl_base_seed0_trained", "dkl_base_seed1_trained",
                    "dkl_base_seed2_trained"],
    }
    out = []
    print(f"Exact permutation null over {len(distinct_labelings(HIER))} distinct "
          f"labelings of HIER={HIER.tolist()}  (source: {SRC})\n")
    for arm, names in models.items():
        for m in names:
            for stat in ("norm", "raw"):
                r = run(m, stat)
                if r is None:
                    print(f"  {m:<28} {stat:<5} -- no fits found")
                    continue
                r["arm"] = arm
                out.append(r)
                star = "  <-- most extreme of all 630" if r["is_most_extreme"] else ""
                print(f"  {arm:<8} {m:<28} {stat:<5} "
                      f"rho={r['observed_mean_rho']:+.4f}  "
                      f"p_exact={r['p_exact_one_sided']:.4f}  "
                      f"(null mean {r['null_mean']:+.3f}, "
                      f"2.5th pct {r['null_q025']:+.3f}){star}")
    with open("results/hierarchy_permutation_null.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nsaved -> results/hierarchy_permutation_null.json")
    print("\nNOTE: this calibrates the gradient against arbitrary labelling of the\n"
          "same ROIs. It does NOT address the post-hoc selection of the gradient\n"
          "from seven regional hypotheses; §2.2's family-size disclosure stands.")


if __name__ == "__main__":
    main()
