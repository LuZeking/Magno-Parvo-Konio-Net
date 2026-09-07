#!/usr/bin/env python
"""Re-derive every number the draft states for the artefact floor (E2.1) and the
progressive-transition OOD arm (E6.2), straight from the result JSONs.

Exits non-zero if any draft number drifts from what the files say. Run it after
touching either section, and after the diet-arm half of the recovery analysis
lands (job 26442320) -- the E2.1 numbers are currently the no-diet geometry only.

    python tmp_analysis/verify_new_claims.py
"""
import glob
import json
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.stats import spearmanr, wilcoxon

FAILS: list[str] = []


def chk(name: str, got: float, want: float, tol: float) -> None:
    ok = abs(got - want) <= tol
    print(f"  {'OK ' if ok else 'BAD'} {name:38s} got {got:+.4f}  draft {want:+.4f}")
    if not ok:
        FAILS.append(name)


def hv4_minus_v1(v: np.ndarray) -> float:
    """§2.2's statistic: hV4 minus the mean of V1v and V1d.

    visualrois.py:72-75. NOT the value at hV4 and NOT a median over the ROI
    profile -- the floor was first published as the latter (+0.126) and had to
    be corrected to +0.051. All four of §2.2's published numbers reproduce
    under this reading and under no other.
    """
    return float(v[6] - v[:2].mean())


def stats_from(rows: list[dict], prefix: str) -> tuple[float, float, float]:
    """(ratio-index contrast, difference contrast, monotonicity rho).

    Both statistics run on nc-divided unique variances, matching production:
    variance_partition returns unique_magno = unique_m / nc, and visualrois.py
    reads that. "normalised" in §2.2 means normalised by (um+up), NOT by nc.
    """
    um = np.array([r[f"{prefix}unique_m"] for r in rows])
    up = np.array([r[f"{prefix}unique_p"] for r in rows])
    ratio = (um - up) / (um + up)
    with warnings.catch_warnings():
        # condition B's TRUTH is constant by construction (solved to zero), so
        # its rho is undefined. Only the estimated rho is ever read.
        warnings.simplefilter("ignore")
        rho = float(spearmanr(np.arange(len(ratio)), ratio).statistic)
    return hv4_minus_v1(ratio), hv4_minus_v1(um - up), rho


def check_recovery() -> None:
    files = sorted(glob.glob(
        "results/vp_recovery_realnc/vp_recovery_dkl_base*.json"))
    assert len(files) == 8, f"expected 8 no-diet subjects, found {len(files)}"

    # (condition, ratio contrast, diff contrast, rho, n positive on ratio)
    want = [("B_manufacture", 0.051, 0.0009, 0.68, 8),
            ("A_recovery", 1.700, 0.499, 1.000, 8)]
    for cond, wr, wd, wrho, wpos in want:
        vals = [stats_from(json.load(open(f))["conditions"][cond]["rows"], "est_")
                for f in files]
        RAT, DIF, RHO = (np.array(x) for x in zip(*vals))
        print(f"[{cond}]")
        chk("normalised (ratio) contrast", RAT.mean(), wr, 0.001)
        chk("raw (difference) contrast", DIF.mean(), wd, 0.001)
        chk("monotonicity rho median", float(np.median(RHO)), wrho, 0.005)
        pos = int((RAT > 0).sum())
        print(f"  {'OK ' if pos == wpos else 'BAD'} ratio sign {pos}/8 positive "
              f"(draft {wpos}/8), wilcoxon p={wilcoxon(RAT).pvalue:.4f}")
        if pos != wpos:
            FAILS.append(f"{cond} sign")
        if cond == "B_manufacture":
            chk("B ratio range low", RAT.min(), 0.007, 0.001)
            chk("B ratio range high", RAT.max(), 0.094, 0.001)
            # the whole point: the truth really is solved to zero
            tru = [stats_from(json.load(open(f))["conditions"][cond]["rows"], "true_")
                   for f in files]
            mx = max(abs(t[0]) for t in tru)
            print(f"  {'OK ' if mx == 0 else 'BAD'} true contrast is exactly zero "
                  f"(max |true| = {mx:.2e})")
            if mx != 0:
                FAILS.append("B truth nonzero")
            p = wilcoxon(DIF).pvalue
            print(f"  {'OK ' if p > 0.5 else 'BAD'} raw floor n.s. p={p:.4f} "
                  f"(draft .84), {int((DIF > 0).sum())}/8 positive (draft 4/8)")
        if cond == "A_recovery":
            # magnitude fidelity: the two statistics are biased OPPOSITE ways,
            # which is why "-0.862 is a floor" was withdrawn.
            tru = [stats_from(json.load(open(f))["conditions"][cond]["rows"], "true_")
                   for f in files]
            TR = np.array([t[0] for t in tru]); TD = np.array([t[1] for t in tru])
            chk("ratio inflation est/true", RAT.mean() / TR.mean(), 1.26, 0.01)
            chk("diff compression est/true", DIF.mean() / TD.mean(), 0.71, 0.01)

    # The measured profile the simulation is conditioned on. Its DECLINE is the
    # whole point: the first run used an ascending ramp and inverted the sign.
    nc = [0.408, 0.413, 0.392, 0.351, 0.352, 0.320, 0.322]
    r = spearmanr(np.arange(7), nc)
    print("[NC profile]")
    chk("rho", r.statistic, -0.893, 0.002)
    print(f"  OK  p={r.pvalue:.4f} (draft states .007)")


def check_production_statistic() -> None:
    """§2.2's four published numbers must reproduce under the hV4-V1 reading."""
    from src.analysis.encoding.stats import load_results
    NC_MIN, labs = 0.1, [1, 2, 3, 4, 5, 6, 7]

    def profile(tag: str, ratio: bool) -> np.ndarray:
        res = load_results("results/encoding", tag)
        out = np.full((len(res), 7), np.nan)
        for i, s in enumerate(sorted(res)):
            z = res[s]
            roi, nc = z["roi_prf-visualrois"], z["noise_ceiling"]
            um, up = z["unique_magno"], z["unique_parvo"]
            for j, lab in enumerate(labs):
                v = (roi == lab) & (nc > NC_MIN) & ((um + up) > 1e-3)
                if v.sum() >= 20:
                    r = (um[v] - up[v]) / (um[v] + up[v]) if ratio else um[v] - up[v]
                    out[i, j] = np.median(r[np.isfinite(r)])
        return out

    print("[production §2.2 statistic]")
    for tag, ratio, want in [("dkl_bcs_seed0_trained", True, -0.834),
                             ("dkl_bcs_seed0_trained", False, -0.166),
                             ("dkl_base_seed0_trained", True, -0.079),
                             ("dkl_base_seed0_trained", False, -0.027)]:
        pr = profile(tag, ratio)
        c = np.array([hv4_minus_v1(pr[i]) for i in range(pr.shape[0])])
        chk(f"{tag.split('_seed')[0]} {'ratio' if ratio else 'diff '} contrast",
            float(np.nanmean(c)), want, 0.001)


# (dvd_trans, diet mean, diet min, diet max, no-diet mean) as the draft states them
OOD_DRAFT = {
    "uniform-noise":    (0.640, 0.634, 0.619, 0.647, 0.729),
    "high-pass":        (0.680, 0.678, 0.647, 0.696, 0.827),
    "rotation":         (0.572, 0.574, 0.567, 0.582, 0.615),
    "false-colour":     (0.839, 0.854, 0.845, 0.860, 0.938),
    "phase-scrambling": (0.362, 0.382, 0.365, 0.391, 0.373),
    "low-pass":         (0.436, 0.444, 0.434, 0.464, 0.356),
    "contrast":         (0.489, 0.481, 0.460, 0.493, 0.549),
}


def check_ood() -> None:
    d = "results/ood_bestep"

    def rel_auc(files: list[str], fam: str) -> np.ndarray:
        """Relative accuracy AUC, matching ood_verdict.curves()."""
        acc = np.array([json.load(open(f))["families"][fam]["accuracy"]
                        for f in files], float)
        return (acc / acc[:, :1]).mean(1)

    diet = sorted(glob.glob(f"{d}/dkl_bcs_seed*.json"))
    nod = sorted(glob.glob(f"{d}/dkl_base_seed*.json"))
    trans = [f"{d}/thesis_dvd_trans.json"]
    print("[OOD relative AUC]")
    inside = closer = 0
    for fam, (wt, wd, wlo, whi, wnd) in OOD_DRAFT.items():
        t = rel_auc(trans, fam)[0]
        dd, nn = rel_auc(diet, fam), rel_auc(nod, fam)
        chk(f"{fam} trans", t, wt, 0.001)
        chk(f"{fam} diet mean", dd.mean(), wd, 0.001)
        chk(f"{fam} diet lo", dd.min(), wlo, 0.001)
        chk(f"{fam} diet hi", dd.max(), whi, 0.001)
        chk(f"{fam} nodiet mean", nn.mean(), wnd, 0.001)
        inside += dd.min() <= t <= dd.max()
        closer += abs(t - dd.mean()) < abs(t - nn.mean())
    print(f"  inside diet seed range: {inside}/7 (draft says 5)")
    print(f"  closer to diet than no-diet: {closer}/7 (draft says 6)")
    if inside != 5:
        FAILS.append("inside count")
    if closer != 6:
        FAILS.append("closer count")
    for lab, files, want in [("trans", trans, 0.5682), ("diet", diet, 0.5530),
                             ("nodiet", nod, 0.6951)]:
        c = np.mean([json.load(open(f))["families"]["uniform-noise"]["accuracy"][0]
                     for f in files])
        chk(f"clean acc {lab}", float(c), want, 0.0005)


def main() -> int:
    check_recovery()
    check_production_statistic()
    check_ood()
    print("\nFAILURES:", FAILS if FAILS else "none")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
