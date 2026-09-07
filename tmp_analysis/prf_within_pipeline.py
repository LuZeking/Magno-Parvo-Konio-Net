"""§100 corrected: every model from results/encoding, each arm against its OWN init.

Three defects in §100.3, all raised by 950c6df and all real:

1. The no-diet arm was contrasted against `untrained_seed0` (the DIET front
   end). Its own initialisation is `untrained_seed0_noblur`. The published
   -0.374 therefore confounds front end with training.
2. "Untrained already exceeds trained diet" read a direction off a null
   (-0.077, 2/8, p=.1094) -- the §98.4 fallacy one notch quieter, and not
   needed: preservation is the claim, and "essentially unchanged" is both what
   the data support and the better fit to a critical-period account.
3. Four of five contrasts crossed encoding_cc (trained) against encoding
   (untrained/Gabor). All six models exist in `encoding`, so the boundary is
   removable rather than merely disclosable.

Cost of the fix, stated rather than absorbed: dkl_bcs_seed1/2 and
dkl_base_seed1/2 are encoding_cc-only (§91.2), so a within-pipeline table is
SEED 0 BY CONSTRUCTION. Seed stability has to be checked separately in
encoding_cc, where all three live, and the two cannot be combined in one table.
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


def betas(model, stat="raw", r2_min=5.0):
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
        out[s] = (m2[1], m2[2])
    return out


MODELS = ["gabor_blur_cs", "gabor_noblur", "untrained_seed0",
          "untrained_seed0_noblur", "dkl_bcs_seed0_trained",
          "dkl_base_seed0_trained"]

for stat in ("raw", "norm"):
    print(f"\n{'=' * 76}\nstatistic = {stat}   (ALL fits from results/encoding)"
          f"\n{'=' * 76}")
    store = {}
    print(f"{'model':26s} {'n':>2s} {'ecc|size':>21s} {'size|ecc':>21s}")
    for m in MODELS:
        b = betas(m, stat)
        if not b:
            print(f"{m:26s}  -- not found in encoding/")
            continue
        store[m] = b
        subs = sorted(b)
        e = np.array([b[s][0] for s in subs])
        zz = np.array([b[s][1] for s in subs])
        print(f"{m:26s} {len(subs):2d} "
              f"{f'{e.mean():+.3f} {int((e>0).sum())}/{len(e)} p={sps.wilcoxon(e).pvalue:.4f}':>21s} "
              f"{f'{zz.mean():+.3f} {int((zz<0).sum())}/{len(zz)}neg p={sps.wilcoxon(zz).pvalue:.4f}':>21s}")

    print(f"\n  {'MATCHED contrast (ecc|size)':46s} {'diff':>8s} {'sign':>6s} "
          f"{'p':>8s}")
    pairs = [
        ("dkl_bcs_seed0_trained", "untrained_seed0",
         "diet:    trained - its OWN init (diet FE)"),
        ("dkl_base_seed0_trained", "untrained_seed0_noblur",
         "no diet: trained - its OWN init (noblur FE)"),
        ("untrained_seed0", "gabor_blur_cs",
         "untrained diet   - input null diet"),
        ("untrained_seed0_noblur", "gabor_noblur",
         "untrained noblur - input null noblur"),
        ("dkl_bcs_seed0_trained", "gabor_blur_cs",
         "trained diet     - input null diet"),
        ("dkl_base_seed0_trained", "gabor_noblur",
         "trained no-diet  - input null noblur"),
    ]
    for a, b_, lab in pairs:
        if a not in store or b_ not in store:
            continue
        subs = sorted(set(store[a]) & set(store[b_]))
        d = np.array([store[a][s][0] - store[b_][s][0] for s in subs])
        print(f"  {lab:46s} {d.mean():+8.3f} "
              f"{int((d > 0).sum()):3d}/{len(d)} {sps.wilcoxon(d).pvalue:8.4f}")

    # the size term, within arm, since its sign flip is part of the claim
    print(f"\n  {'MATCHED contrast (size|ecc)':46s} {'diff':>8s} {'sign':>6s} "
          f"{'p':>8s}")
    for a, b_, lab in pairs[:2]:
        if a not in store or b_ not in store:
            continue
        subs = sorted(set(store[a]) & set(store[b_]))
        d = np.array([store[a][s][1] - store[b_][s][1] for s in subs])
        print(f"  {lab:46s} {d.mean():+8.3f} "
              f"{int((d > 0).sum()):3d}/{len(d)} {sps.wilcoxon(d).pvalue:8.4f}")
