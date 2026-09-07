"""Do the §98 partial betas survive the choices that produced them?

The headline is a suppression pattern: controlling pRF size roughly DOUBLES the
eccentricity beta in the diet arm (+0.066 -> +0.119 norm), because the two
predictors correlate at +0.61 and act with opposite signs. That is
mathematically expected rather than suspicious, but it is also exactly the
shape of result that an arbitrary implementation choice can manufacture -- and
§96 established that a statistic which moves on such a choice is not measuring
the data.

Two choices are unstated conventions with no scientific content:
  * the prf_R2 threshold (5 was picked because it is conventional)
  * the log transform on both predictors
Vary both. If the partials keep their sign and rough magnitude across all of
them, the dissociation is a property of the data; if they do not, it is a
property of the filter and must be reported as such.
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


def z(v):
    return (v - v.mean()) / v.std()


def partials(model, r2_min, use_log, stat="norm"):
    out_e, out_s = [], []
    for f in sorted(glob.glob(f"{R}/{model}_subj*.npz")):
        s = int(os.path.basename(f).split("subj")[-1].split(".")[0])
        d = np.load(f, allow_pickle=True)
        um, up, nc = d["unique_magno"], d["unique_parvo"], d["noise_ceiling"]
        mask = nsd_data.load_mask(s, "nsdgeneral")
        ecc = nsd_data.load_prf(s, "eccentricity", mask)
        size = nsd_data.load_prf(s, "size", mask)
        r2 = nsd_data.load_prf(s, "R2", mask)
        ok = ((nc > 0.1) & ((um + up) > 1e-3) & (r2 > r2_min)
              & np.isfinite(ecc) & np.isfinite(size) & (size > 0))
        if ok.sum() < 200:
            continue
        e, sz = ecc[ok], size[ok]
        le = z(np.log(e + 0.5)) if use_log else z(e)
        ls = z(np.log(sz)) if use_log else z(sz)
        y = ((um[ok] - up[ok]) / (um[ok] + up[ok])) if stat == "norm" \
            else (um[ok] - up[ok])
        A = np.column_stack([np.ones(ok.sum()), le, ls])
        coef, *_ = np.linalg.lstsq(A, z(y), rcond=None)
        out_e.append(coef[1]); out_s.append(coef[2])
    return np.array(out_e), np.array(out_s)


ARMS = {"diet": "dkl_bcs_seed0_trained", "no diet": "dkl_base_seed0_trained"}
print("Partial betas (statistic = norm, seed 0), across the two arbitrary "
      "choices\n")
print(f"{'arm':8s} {'R2>':>4s} {'pred':>7s} "
      f"{'ecc|size':>20s} {'size|ecc':>20s}")
for arm, model in ARMS.items():
    for use_log in (True, False):
        for r2_min in (0.0, 5.0, 10.0, 20.0):
            be, bs = partials(model, r2_min, use_log)
            pe, ps = sps.wilcoxon(be).pvalue, sps.wilcoxon(bs).pvalue
            print(f"{arm:8s} {r2_min:4.0f} {'log' if use_log else 'linear':>7s} "
                  f"{f'{be.mean():+.3f} {int((be<0).sum())}/8 p={pe:.4f}':>20s} "
                  f"{f'{bs.mean():+.3f} {int((bs<0).sum())}/8 p={ps:.4f}':>20s}")
    print()

print("The dissociation is a property of the data only if, in EVERY row:")
print("  diet     ecc|size > 0 and size|ecc < 0")
print("  no diet  ecc|size ~ 0 and size|ecc > 0")
print("Any row that breaks it is the filter talking, not the cortex.")
