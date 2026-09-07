"""Does the ecc/size dissociation exist for §8.2's UNTRAINED front-end pair?

§99.3 claimed my trained diet-vs-no-diet result "reconciled" with the draft's
line 1890. It does not: `blur_verdict.py:24` pins that sentence's contrast as
MAIN, CTRL = "untrained_seed0", "untrained_seed0_noblur" -- one UNTRAINED
network, diet front end against no-blur front end. Same verdict (n.s.),
different models, different contrast. §99.3 is withdrawn.

This runs the same marginal-vs-partial decomposition on the pair line 1890
actually names. The pRF maps are subject-level, so they apply to any model's
voxelwise fits and nothing needs refitting.

Two outcomes, both informative and stated before running:
  * if the partial separates the two front ends the way it separates the
    trained arms, line 1890 gains a companion on its own contrast;
  * if it does not, the size confound is something TRAINING introduces, which
    is a better story than either version -- and it would mean the trained
    result cannot be read as a front-end effect at all.

Note these two models live only in results/encoding/ (§91): neither is in
encoding_cc, so the source directory differs from §98's and is stated here.
"""
import glob
import os
import sys

import numpy as np
from scipy import stats as sps

sys.path.insert(0, "/home/byrong/projects/TSN/Magno-Parvo-Konio-Net/"
                   ".claude/worktrees/mpk-dev")

from src.analysis.encoding import nsd_data  # noqa: E402

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
        m1, *_ = np.linalg.lstsq(np.column_stack([np.ones(n), le]), y,
                                 rcond=None)
        m2, *_ = np.linalg.lstsq(np.column_stack([np.ones(n), le, ls]), y,
                                 rcond=None)
        out[s] = (m1[1], m2[1], m2[2])   # ecc marginal, ecc|size, size|ecc
    return out


MAIN, CTRL = "untrained_seed0", "untrained_seed0_noblur"
print(f"§8.2's pair: {MAIN}  vs  {CTRL}   (source: results/encoding)\n")

for stat in ("raw", "norm"):
    A, B = betas(MAIN, stat), betas(CTRL, stat)
    subs = sorted(set(A) & set(B))
    print(f"--- statistic = {stat}, n = {len(subs)} subjects ---")
    print(f"{'term':12s} {'diet fe':>9s} {'noblur fe':>10s} {'diff':>9s} "
          f"{'pos':>5s} {'p':>8s}")
    for idx, term in ((0, "ecc marginal"), (1, "ecc|size"), (2, "size|ecc")):
        a = np.array([A[s][idx] for s in subs])
        b = np.array([B[s][idx] for s in subs])
        d = a - b
        p = sps.wilcoxon(d).pvalue
        print(f"{term:12s} {a.mean():+9.3f} {b.mean():+10.3f} {d.mean():+9.3f} "
              f"{int((d > 0).sum()):3d}/{len(d)} {p:8.4f}")
    # within-model, is there an ecc effect at all once size is controlled?
    for name, M in ((MAIN, A), (CTRL, B)):
        v = np.array([M[s][1] for s in subs])
        w = np.array([M[s][2] for s in subs])
        print(f"  {name:24s} ecc|size {v.mean():+.3f} "
              f"{int((v > 0).sum())}/{len(v)} p={sps.wilcoxon(v).pvalue:.4f}"
              f"   size|ecc {w.mean():+.3f} "
              f"{int((w < 0).sum())}/{len(w)} neg "
              f"p={sps.wilcoxon(w).pvalue:.4f}")
    print()

print("Trained comparison for reference (§99.1): ecc|size difference")
print("  +0.134 norm / +0.305 raw, 8/8, at the n=8 floor.")
print("If the untrained pair shows no such separation, the dissociation is")
print("something training introduces, not a property of the front end.")
