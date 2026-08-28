"""Kastner2015 ROIs: hMT+ (TO1/TO2) and colour-biased ventral (VO1/VO2).

The plan deferred the dorsal/MT test as a limitation and had no colour ROI
beyond hV4. Both exist in the Kastner2015 atlas, which is on disk for all 8
subjects, and the encoding fits used a deterministic `nsdgeneral` mask -- so
these labels can be attached to the EXISTING npz results with no refitting.

Two pre-declared contrasts, each with the eccentricity confound handled:
  TO1/TO2 (hMT+)   -> magno, the textbook M-pathway target
  VO1/VO2          -> parvo, the colour-biased ventral target
hMT+ is peripheral-biased and VO is foveal-biased, so the raw contrast is
partly a restatement of the eccentricity gradient. The residual version
regresses the paired statistic on log eccentricity ACROSS VOXELS within
subject and re-tests the ROI difference on the residuals; only that version
licenses an ROI claim independent of retinotopy.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sps

from src.analysis.encoding import nsd_data
from src.analysis.encoding.stats import load_results

KASTNER = {1: "V1v", 2: "V1d", 3: "V2v", 4: "V2d", 5: "V3v", 6: "V3d",
           7: "hV4", 8: "VO1", 9: "VO2", 10: "PHC1", 11: "PHC2",
           12: "TO2", 13: "TO1", 14: "LO2", 15: "LO1", 16: "V3B",
           17: "V3A", 18: "IPS0", 19: "IPS1", 20: "IPS2", 21: "IPS3",
           22: "IPS4", 23: "IPS5", 24: "SPL1", 25: "FEF"}
MT = ["TO1", "TO2"]
VO = ["VO1", "VO2"]
BAND_CENTRE = {1: 0.25, 2: 0.75, 3: 1.5, 4: 3.0, 5: 6.0}
NC_MIN = 0.1
MIN_VOX = 20

res = load_results("results/encoding", "untrained_seed0")
subs = sorted(res)
names = [KASTNER[k] for k in sorted(KASTNER)]

tab = np.full((len(subs), len(names)), np.nan)      # raw median(Um-Up)
res_tab = np.full((len(subs), len(names)), np.nan)  # ecc-residualised
nvox = np.zeros((len(subs), len(names)), int)

for i, s in enumerate(subs):
    z = res[s]
    mask = nsd_data.load_mask(s, "nsdgeneral")
    kl = nsd_data.load_roi_labels(s, "Kastner2015", mask)
    if kl.size != z["noise_ceiling"].size:
        raise SystemExit(f"subj{s}: Kastner {kl.size} != npz "
                         f"{z['noise_ceiling'].size} -- mask mismatch")

    nc = z["noise_ceiling"]
    d = z["unique_magno"] - z["unique_parvo"]
    ecc = z["roi_prf-eccrois"]
    ok = (nc > NC_MIN) & np.isfinite(d)

    # Residualise on log eccentricity using every voxel that has a band.
    fit = ok & (ecc > 0)
    x = np.log([BAND_CENTRE[b] for b in ecc[fit]])
    b1, b0 = np.polyfit(x, d[fit], 1)
    resid = np.full_like(d, np.nan)
    resid[fit] = d[fit] - (b0 + b1 * x)

    for j, lab in enumerate(sorted(KASTNER)):
        v = ok & (kl == lab)
        nvox[i, j] = v.sum()
        if v.sum() >= MIN_VOX:
            tab[i, j] = np.median(d[v])
        vr = v & fit
        if vr.sum() >= MIN_VOX:
            res_tab[i, j] = np.median(resid[vr])


def report(x, label):
    x = x[np.isfinite(x)]
    if len(x) < 5:
        print(f"  {label:<30} n={len(x)} too few subjects")
        return
    W, p = sps.wilcoxon(x)
    print(f"  {label:<30} mean={x.mean():+.4f}  p={p:.4f}  "
          f"{int((x > 0).sum())}/{len(x)} pos  "
          f"dz={x.mean() / x.std(ddof=1):+.2f}")


def pool(t, labs):
    idx = [names.index(n) for n in labs]
    return np.nanmean(t[:, idx], 1)


print(f"Kastner2015 ROIs, n={len(subs)} subjects, NC>{NC_MIN}, "
      f">={MIN_VOX} voxels\n")
print("Per-ROI median(U_m - U_p), raw | median voxels/subject")
order = np.argsort(-np.nanmean(tab, 0))
for j in order:
    m = np.nanmean(tab[:, j])
    if not np.isfinite(m):
        continue
    npos = int(np.nansum(tab[:, j] > 0))
    nn = int(np.sum(np.isfinite(tab[:, j])))
    print(f"  {names[j]:<6} {m:+.4f}   {npos}/{nn} subj   "
          f"med n_vox={int(np.median(nvox[:, j]))}")

print("\nPre-declared ROI contrasts (raw)")
report(pool(tab, MT), "hMT+ (TO1,TO2)")
report(pool(tab, VO), "VO (VO1,VO2)")
report(pool(tab, MT) - pool(tab, VO), "INTERACTION hMT+ - VO")

print("\nSame, residualised on log eccentricity within subject")
report(pool(res_tab, MT), "hMT+ (TO1,TO2) | ecc")
report(pool(res_tab, VO), "VO (VO1,VO2) | ecc")
report(pool(res_tab, MT) - pool(res_tab, VO), "INTERACTION hMT+ - VO | ecc")

# Leave-one-subject-out consistency of the whole 25-ROI profile.
print("\nLeave-one-subject-out profile consistency (Spearman)")
rs = []
for i in range(len(subs)):
    others = np.nanmean(np.delete(tab, i, axis=0), 0)
    v = np.isfinite(others) & np.isfinite(tab[i])
    if v.sum() >= 8:
        r = sps.spearmanr(tab[i][v], others[v]).statistic
        rs.append(r)
        print(f"  subj{subs[i]:02d}  rho={r:+.3f}  ({int(v.sum())} ROIs)")
print(f"  mean rho = {np.mean(rs):+.3f}")
