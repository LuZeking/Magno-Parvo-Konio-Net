"""Does the arm difference live in the MARGINAL eccentricity effect or only the PARTIAL?

This decides how §98 relates to the draft. The draft's live sentence (line
1890, "Claims deliberately not made") reports the arms as NOT differing on
eccentricity -- but it reports a MARGINAL slope (raw +0.047 vs +0.041,
difference n.s.). My interaction test used the PARTIAL, controlling pRF size.
Those are different quantities, so the two results may not conflict at all.

If the marginal interaction is n.s. and the partial one is not, the finding is
sharper than "the draft was wrong": the size confound was hiding the ARM
DIFFERENCE as well as attenuating the within-arm effect, and the draft's
sentence is correct about the quantity it names.
"""
import glob
import os
import sys

import numpy as np
from scipy import stats as sps

sys.path.insert(0, "/home/byrong/projects/TSN/Magno-Parvo-Konio-Net/"
                   ".claude/worktrees/mpk-dev")

from src.analysis.encoding import nsd_data  # noqa: E402

R = ("/home/byrong/projects/TSN/Magno-Parvo-Konio-Net/.claude/worktrees/"
     "mpk-dev/results/encoding_cc")
_c = {}


def z(v):
    return (v - v.mean()) / v.std()


def prf(s):
    if s not in _c:
        m = nsd_data.load_mask(s, "nsdgeneral")
        _c[s] = (m, nsd_data.load_prf(s, "eccentricity", m),
                 nsd_data.load_prf(s, "size", m), nsd_data.load_prf(s, "R2", m))
    return _c[s]


def betas(model, stat, r2_min=5.0):
    """{subj: (ecc_marginal, ecc_partial)}"""
    out = {}
    for f in sorted(glob.glob(f"{R}/{model}_subj*.npz")):
        s = int(os.path.basename(f).split("subj")[-1].split(".")[0])
        d = np.load(f, allow_pickle=True)
        um, up, nc = d["unique_magno"], d["unique_parvo"], d["noise_ceiling"]
        _, ecc, size, r2 = prf(s)
        ok = ((nc > 0.1) & ((um + up) > 1e-3) & (r2 > r2_min)
              & np.isfinite(ecc) & np.isfinite(size) & (size > 0))
        le = z(np.log(ecc[ok] + 0.5))
        ls = z(np.log(size[ok]))
        y = z(((um[ok] - up[ok]) / (um[ok] + up[ok])) if stat == "norm"
              else (um[ok] - up[ok]))
        n = int(ok.sum())
        m1, *_ = np.linalg.lstsq(np.column_stack([np.ones(n), le]), y,
                                 rcond=None)
        m2, *_ = np.linalg.lstsq(np.column_stack([np.ones(n), le, ls]), y,
                                 rcond=None)
        out[s] = (m1[1], m2[1])
    return out


print(f"{'stat':5s} {'term':10s} {'diet':>8s} {'no diet':>8s} "
      f"{'diff':>8s} {'pos':>5s} {'p':>8s}")
for stat in ("norm", "raw"):
    A = [betas(f"dkl_bcs_seed{s}_trained", stat) for s in (0, 1, 2)]
    B = [betas(f"dkl_base_seed{s}_trained", stat) for s in (0, 1, 2)]
    subs = sorted(set(A[0]) & set(B[0]))
    for idx, term in ((0, "marginal"), (1, "partial")):
        a = np.array([np.mean([x[s][idx] for x in A]) for s in subs])
        b = np.array([np.mean([x[s][idx] for x in B]) for s in subs])
        d = a - b
        p = sps.wilcoxon(d).pvalue
        print(f"{stat:5s} {term:10s} {a.mean():+8.3f} {b.mean():+8.3f} "
              f"{d.mean():+8.3f} {int((d > 0).sum()):3d}/{len(d)} {p:8.4f}")

print("\nIf marginal is n.s. and partial is significant, the draft's sentence")
print("is right about the quantity it names, and the size confound was hiding")
print("the arm difference rather than the draft having got it wrong.")
