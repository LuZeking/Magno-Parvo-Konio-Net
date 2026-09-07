"""Bound the two nulls §101 stated without bounding them.

950c6df is right that both are equivalence claims read off non-significant
differences, and that this paper bounds all four of its other nulls with TOST
(§2.6.1, §2.6.2, VO, phase-scrambling) rather than asserting them:

  N1  diet: trained - own init, ecc|size      "preserved"
  N2  trained no-diet - input null (noblur)   "no more than a Gabor bank"

THE SESOI PROBLEM, stated rather than solved by picking one. §77.5 forbids
re-widening a bound after seeing an interval, and its bounds were fixed before
computing for the paper's three dissociations -- none covers these two. Picking
a bound now, having already seen the point estimates, would be exactly the move
§77 exists to prevent.

So this does NOT declare equivalence. It reports what the data can bound:
  * the 90% CI (the TOST interval), and
  * the MDE at 80% power -- the smallest effect this n could have detected,
    which is a property of the design, not of the result;
and then asks one question that needs no chosen threshold: does the interval
exclude an INDEPENDENTLY MEASURED reference effect -- the same contrast in the
other arm? That is a comparison, not a bound, and it is the strongest thing
available without a pre-registered SESOI.
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, "/home/byrong/projects/TSN/Magno-Parvo-Konio-Net/"
                   ".claude/worktrees/mpk-dev")

from src.analysis.encoding import nsd_data  # noqa: E402
from tmp_analysis.equivalence_tests import mde_paired, tost_paired  # noqa: E402

SRC = ("/home/byrong/projects/TSN/Magno-Parvo-Konio-Net/.claude/worktrees/"
       "mpk-dev/results/encoding")
_c = {}


def z(v):
    return (v - v.mean()) / v.std()


def prf(s):
    if s not in _c:
        m = nsd_data.load_mask(s, "nsdgeneral")
        _c[s] = (nsd_data.load_prf(s, "eccentricity", m),
                 nsd_data.load_prf(s, "size", m),
                 nsd_data.load_prf(s, "R2", m))
    return _c[s]


def betas(model, stat, r2_min=5.0):
    out = {}
    for f in sorted(glob.glob(f"{SRC}/{model}_subj*.npz")):
        s = int(os.path.basename(f).split("subj")[-1].split(".")[0])
        d = np.load(f, allow_pickle=True)
        um, up, nc = d["unique_magno"], d["unique_parvo"], d["noise_ceiling"]
        ecc, size, r2 = prf(s)
        ok = ((nc > 0.1) & ((um + up) > 1e-3) & (r2 > r2_min)
              & np.isfinite(ecc) & np.isfinite(size) & (size > 0))
        if ok.sum() < 200:
            continue
        le, ls = z(np.log(ecc[ok] + 0.5)), z(np.log(size[ok]))
        y = z(((um[ok] - up[ok]) / (um[ok] + up[ok])) if stat == "norm"
              else (um[ok] - up[ok]))
        n = int(ok.sum())
        m2, *_ = np.linalg.lstsq(np.column_stack([np.ones(n), le, ls]), y,
                                 rcond=None)
        out[s] = m2[1]
    return out


def paired(a, b, stat):
    A, B = betas(a, stat), betas(b, stat)
    subs = sorted(set(A) & set(B))
    return np.array([A[s] - B[s] for s in subs])


NULLS = [
    ("N1 diet: trained - own init",
     "dkl_bcs_seed0_trained", "untrained_seed0",
     "dkl_base_seed0_trained", "untrained_seed0_noblur",
     "the same contrast in the NO-DIET arm"),
    ("N2 trained no-diet - input null (noblur)",
     "dkl_base_seed0_trained", "gabor_noblur",
     "dkl_bcs_seed0_trained", "gabor_blur_cs",
     "the same contrast in the DIET arm"),
]

for label, a, b, ra, rb, refdesc in NULLS:
    print(f"\n{'=' * 74}\n{label}\n{'=' * 74}")
    for stat in ("raw", "norm"):
        d = paired(a, b, stat)
        ref = paired(ra, rb, stat).mean()
        # sesoi is set to |reference effect| purely to obtain the TOST
        # p-value against an INDEPENDENTLY MEASURED quantity; it is NOT a
        # pre-registered smallest-effect-of-interest and must not be read as
        # one (§77.5).
        r = tost_paired(d, abs(ref))
        mde, dz = mde_paired(d)
        lo, hi = r["ci"]
        print(f"  {stat:4s} diff {r['diff']:+.3f}  90% CI "
              f"[{lo:+.3f}, {hi:+.3f}]  MDE(80%) {mde:+.3f} (dz {dz:.2f})")
        print(f"       reference effect = {ref:+.3f} ({refdesc})")
        excl = (lo > ref) or (hi < ref)
        print(f"       interval {'EXCLUDES' if excl else 'CONTAINS'} the "
              f"reference  -> "
              f"{'smaller than it' if excl else 'cannot be distinguished from it'}")
        print(f"       TOST vs |reference| as bound: p={r['p_tost']:.4f} "
              f"{'(inside)' if r['equivalent'] else '(not inside)'}")

print("\nNo equivalence to ZERO is claimed anywhere above: these data have no")
print("pre-registered SESOI for this contrast, and choosing one now would be")
print("the move §77.5 forbids. What is reportable is the interval, the MDE,")
print("and whether the reference effect is excluded.")
