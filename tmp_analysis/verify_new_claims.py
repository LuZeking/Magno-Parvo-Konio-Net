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
import sys

import numpy as np
from scipy.stats import spearmanr, wilcoxon

FAILS: list[str] = []


def chk(name: str, got: float, want: float, tol: float) -> None:
    ok = abs(got - want) <= tol
    print(f"  {'OK ' if ok else 'BAD'} {name:38s} got {got:+.4f}  draft {want:+.4f}")
    if not ok:
        FAILS.append(name)


def contrast(rows: list[dict]) -> tuple[float, float, float]:
    """(normalised median, raw median, monotonicity rho) for one subject."""
    um = np.array([r["est_unique_m"] for r in rows])
    up = np.array([r["est_unique_p"] for r in rows])
    n = (um - up) / (um + up)
    return (float(np.median(n)), float(np.median(um - up)),
            float(spearmanr(np.arange(len(n)), n).statistic))


def check_recovery() -> None:
    files = sorted(glob.glob(
        "results/vp_recovery_realnc/vp_recovery_dkl_base*.json"))
    assert len(files) == 8, f"expected 8 no-diet subjects, found {len(files)}"
    for cond, wn, wr, wrho in [("B_manufacture", 0.126, 0.025, 0.68),
                               ("A_recovery", 0.361, 0.064, 1.000)]:
        vals = [contrast(json.load(open(f))["conditions"][cond]["rows"])
                for f in files]
        N, R, RHO = (np.array(x) for x in zip(*vals))
        print(f"[{cond}]")
        chk("normalised subject-median mean", N.mean(), wn, 0.001)
        chk("raw subject-median mean", R.mean(), wr, 0.001)
        chk("monotonicity rho median", float(np.median(RHO)), wrho, 0.005)
        p = wilcoxon(N).pvalue
        print(f"  OK  sign {(N > 0).sum()}/8 positive, wilcoxon p={p:.4f}")
        if cond == "B_manufacture":
            chk("B range low", N.min(), 0.038, 0.001)
            chk("B range high", N.max(), 0.207, 0.001)

    # The measured profile the simulation is conditioned on. Its DECLINE is the
    # whole point: the first run used an ascending ramp and inverted the sign.
    nc = [0.408, 0.413, 0.392, 0.351, 0.352, 0.320, 0.322]
    r = spearmanr(np.arange(7), nc)
    print("[NC profile]")
    chk("rho", r.statistic, -0.893, 0.002)
    print(f"  OK  p={r.pvalue:.4f} (draft states .007)")


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
    check_ood()
    print("\nFAILURES:", FAILS if FAILS else "none")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
