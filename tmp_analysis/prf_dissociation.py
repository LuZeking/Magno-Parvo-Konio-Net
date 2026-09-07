"""§7.2 item 2: is stream preference about ECCENTRICITY or about pRF SIZE?

The eccentricity bands confound the two entirely, so "magno explains peripheral
voxels" and "magno explains large-pRF voxels" are currently the same sentence.
They are different claims and only one is about retinotopy. NSD's continuous
pRF maps separate them: prf_characterise.py measured r(log ecc, log size) =
+0.611 within subject (VIF 1.66), which is low enough for partial betas to mean
something.

CONVENTIONS, stated because this project has repeatedly paid for leaving them
implicit (§86, §92):
  * voxel filter   nc > 0.1, (u_m + u_p) > 1e-3, prf_R2 > 5, finite ecc/size
  * predictors     z-scored log(ecc + 0.5) and z-scored log(size), within
                   subject; log because cortical magnification is logarithmic,
                   +0.5 because foveal ecc reaches 0
  * both statistics reported: SPI = (u_m - u_p)/(u_m + u_p) per voxel, and the
    difference u_m - u_p. §2.2's "norm" and "raw", same meanings as there
  * estimator      standardised OLS beta within subject; aggregated across the
                   8 subjects with subject as the resampling unit; Wilcoxon
                   signed-rank, which floors at p = .0078 at n = 8 (§80)
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
NC_MIN, DENOM_MIN, R2_MIN = 0.1, 1e-3, 5.0


def z(v):
    return (v - v.mean()) / v.std()


def betas(y, X):
    """Standardised OLS betas for y ~ X (X already z-scored), with intercept."""
    A = np.column_stack([np.ones(len(y)), X])
    coef, *_ = np.linalg.lstsq(A, z(y), rcond=None)
    return coef[1:]


def analyse(model):
    rows = []
    for f in sorted(glob.glob(f"{R}/{model}_subj*.npz")):
        s = int(os.path.basename(f).split("subj")[-1].split(".")[0])
        d = np.load(f, allow_pickle=True)
        um, up, nc = d["unique_magno"], d["unique_parvo"], d["noise_ceiling"]
        mask = nsd_data.load_mask(s, "nsdgeneral")
        ecc = nsd_data.load_prf(s, "eccentricity", mask)
        size = nsd_data.load_prf(s, "size", mask)
        r2 = nsd_data.load_prf(s, "R2", mask)
        assert len(ecc) == len(um), (
            f"subj{s}: pRF has {len(ecc)} voxels, fit has {len(um)}"
        )
        ok = ((nc > NC_MIN) & ((um + up) > DENOM_MIN) & (r2 > R2_MIN)
              & np.isfinite(ecc) & np.isfinite(size) & (size > 0))
        if ok.sum() < 200:
            continue
        le, ls = z(np.log(ecc[ok] + 0.5)), z(np.log(size[ok]))
        X2 = np.column_stack([le, ls])
        for stat, y in (("norm", (um[ok] - up[ok]) / (um[ok] + up[ok])),
                        ("raw", um[ok] - up[ok])):
            b_e_alone = betas(y, le[:, None])[0]
            b_s_alone = betas(y, ls[:, None])[0]
            b_e, b_s = betas(y, X2)
            rows.append(dict(subject=s, stat=stat, n=int(ok.sum()),
                             ecc_alone=b_e_alone, size_alone=b_s_alone,
                             ecc_partial=b_e, size_partial=b_s))
    return rows


def report(rows, stat, label):
    r = [x for x in rows if x["stat"] == stat]
    if not r:
        return
    print(f"\n  --- {label}, statistic = {stat}, n = {len(r)} subjects "
          f"({np.median([x['n'] for x in r]):.0f} voxels median) ---")
    print(f"  {'term':16s} {'mean beta':>10s} {'range':>18s} "
          f"{'neg':>5s} {'p':>7s}")
    for key, name in (("ecc_alone", "ecc alone"),
                      ("size_alone", "size alone"),
                      ("ecc_partial", "ecc | size"),
                      ("size_partial", "size | ecc")):
        v = np.array([x[key] for x in r])
        w = sps.wilcoxon(v)
        print(f"  {name:16s} {v.mean():+10.3f} "
              f"{f'{v.min():+.3f}..{v.max():+.3f}':>18s} "
              f"{int((v < 0).sum()):3d}/{len(v)} {w.pvalue:7.4f}")


MODELS = [(f"dkl_{a}_seed{s}_trained", f"{'diet' if a == 'bcs' else 'no diet'}"
           f" seed{s}") for a in ("bcs", "base") for s in (0, 1, 2)]

print("§7.2 item 2 -- eccentricity vs pRF size as predictors of stream "
      "preference")
print(f"filters: nc>{NC_MIN}, (um+up)>{DENOM_MIN}, prf_R2>{R2_MIN}")
for model, label in MODELS:
    rows = analyse(model)
    if not rows:
        print(f"\n{model}: no fits found")
        continue
    print(f"\n{'=' * 70}\n{model}  ({label})\n{'=' * 70}")
    for stat in ("norm", "raw"):
        report(rows, stat, label)
