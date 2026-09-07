"""Characterise NSD's continuous pRF maps before designing the §7.2-item-2 model.

Three things have to be settled from the data rather than assumed:

1. ALIGNMENT. The pRF volumes are indexed with the same nsdgeneral mask the
   fits used, but that is a claim until it is tested. `prf-eccrois` is an
   ordinal band label attached by the same mask, so median pRF eccentricity
   must rise monotonically across those bands. If it does not, the indexing is
   wrong and nothing downstream means anything.
2. THE R2 FILTER. NSD leaves unfittable voxels at R2 = 0. Whatever threshold is
   used has to be stated and its cost in voxels known.
3. COLLINEARITY. Eccentricity and pRF size both grow with cortical distance
   from the fovea. If they are collinear at r ~ 0.9 within subject, "controls
   for size" is not something the data can support and the partial betas will
   be unstable. This decides whether §7.2 item 2 is answerable at all.
"""
import sys

import numpy as np
from scipy import stats as sps

sys.path.insert(0, "/home/byrong/projects/TSN/Magno-Parvo-Konio-Net/"
                   ".claude/worktrees/mpk-dev")

from src.analysis.encoding import nsd_data  # noqa: E402

ECC_BANDS = {1: "0-0.5deg", 2: "0.5-1", 3: "1-2", 4: "2-4", 5: "4+"}

print(f"{'subj':>4s} {'voxels':>7s} {'R2>0':>7s} {'R2>5':>7s} {'R2>10':>7s} "
      f"{'ecc med':>8s} {'size med':>9s}")
per_subj = []
for s in range(1, 9):
    mask = nsd_data.load_mask(s, "nsdgeneral")
    ecc = nsd_data.load_prf(s, "eccentricity", mask)
    size = nsd_data.load_prf(s, "size", mask)
    r2 = nsd_data.load_prf(s, "R2", mask)
    n = mask.sum()
    ok = r2 > 5
    print(f"{s:4d} {n:7d} {int((r2 > 0).sum()):7d} {int((r2 > 5).sum()):7d} "
          f"{int((r2 > 10).sum()):7d} {np.median(ecc[ok]):8.2f} "
          f"{np.median(size[ok]):9.2f}")
    per_subj.append((s, mask, ecc, size, r2))

print("\n=== 1. ALIGNMENT: median pRF ecc by the existing eccroi band ===")
print(f"{'subj':>4s} " + " ".join(f"{v:>9s}" for v in ECC_BANDS.values())
      + "   monotonic?")
n_mono = 0
for s, mask, ecc, size, r2 in per_subj:
    band = nsd_data.load_roi_labels(s, "prf-eccrois", mask)
    meds = []
    for b in ECC_BANDS:
        m = (band == b) & (r2 > 5)
        meds.append(np.median(ecc[m]) if m.sum() >= 20 else np.nan)
    v = np.array(meds)
    mono = np.all(np.diff(v[np.isfinite(v)]) > 0)
    n_mono += bool(mono)
    print(f"{s:4d} " + " ".join(f"{x:9.2f}" for x in v)
          + f"   {'YES' if mono else 'NO'}")
print(f"\nmonotonic in {n_mono}/8 subjects "
      f"-> {'ALIGNMENT CONFIRMED' if n_mono == 8 else 'CHECK INDEXING'}")

print("\n=== 2/3. COLLINEARITY of the two predictors, within subject ===")
print(f"{'subj':>4s} {'n':>7s} {'r(ecc,size)':>12s} {'r(log,log)':>11s} "
      f"{'VIF':>6s}")
rs = []
for s, mask, ecc, size, r2 in per_subj:
    ok = (r2 > 5) & np.isfinite(ecc) & np.isfinite(size) & (size > 0)
    e, z = ecc[ok], size[ok]
    r_lin = np.corrcoef(e, z)[0, 1]
    r_log = np.corrcoef(np.log(e + 0.5), np.log(z))[0, 1]
    rs.append(r_log)
    print(f"{s:4d} {int(ok.sum()):7d} {r_lin:12.3f} {r_log:11.3f} "
          f"{1 / (1 - r_log ** 2):6.2f}")
rs = np.array(rs)
print(f"\nmean r(log ecc, log size) = {rs.mean():+.3f}  "
      f"range {rs.min():+.3f}..{rs.max():+.3f}")
print(f"mean VIF = {np.mean(1 / (1 - rs ** 2)):.2f}")
print("\nVIF < 5 -> the partial betas are interpretable and §7.2 item 2 is")
print("answerable; VIF > 10 -> 'controlling for size' is not something these")
print("data can support and that must be reported instead of a partial beta.")
