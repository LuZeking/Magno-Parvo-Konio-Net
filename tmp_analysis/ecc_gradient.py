"""A3.1 -- eccentricity gradient of the stream-preference index (n=8).

Biological prediction: the parvocellular pathway is fovea-weighted (small,
high-acuity, chromatically opponent RFs) and the magnocellular pathway is
periphery-weighted (large, achromatic, low-SF RFs). So the stream-preference
index SPI = (Um - Up) / (Um + Up) should INCREASE with eccentricity.

Test: per subject, SPI averaged in each of NSD's four `prf-eccrois` bands,
then a per-subject linear slope across bands; group inference is a paired
test on the 8 slopes (sign-flip permutation + Wilcoxon). This is a trend
test, so it does not depend on any single band being individually reliable.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sps

from src.analysis.encoding.stats import ECC_LABELS, load_results

# NSD prf-eccrois bands are nested rings with these outer edges in degrees;
# use the ring centres (log-spaced) as the regressor.
BAND_CENTRE = {1: 0.25, 2: 0.75, 3: 1.5, 4: 3.0}
NC_MIN = 0.1

res = load_results("results/encoding", "untrained_seed0")
subjects = sorted(res)
bands = sorted(ECC_LABELS)

spi_tab = np.full((len(subjects), len(bands)), np.nan)
nvox = np.zeros((len(subjects), len(bands)), dtype=int)
for i, s in enumerate(subjects):
    z = res[s]
    roi, nc = z["roi_prf-eccrois"], z["noise_ceiling"]
    um, up = z["unique_magno"], z["unique_parvo"]
    for j, b in enumerate(bands):
        v = (roi == b) & (nc > NC_MIN)
        # SPI is only meaningful where the joint model explains something,
        # and the ratio is unstable when the denominator is near zero.
        denom = um + up
        v &= denom > 1e-3
        spi = (um[v] - up[v]) / denom[v]
        spi = spi[np.isfinite(spi)]
        nvox[i, j] = spi.size
        if spi.size >= 20:
            spi_tab[i, j] = np.median(spi)

print(f"Voxels per eccentricity band (nc>{NC_MIN})")
print(f"{'subj':6s}" + "".join(f"{ECC_LABELS[b]:>10s}" for b in bands))
for i, s in enumerate(subjects):
    print(f"{s:<6d}" + "".join(f"{nvox[i, j]:>10d}" for j in range(len(bands))))

print(f"\nMedian SPI per band  (positive = magno-preferring)")
print(f"{'subj':6s}" + "".join(f"{ECC_LABELS[b]:>10s}" for b in bands)
      + f"{'slope':>10s}")
slopes = np.full(len(subjects), np.nan)
x = np.log(np.array([BAND_CENTRE[b] for b in bands]))
for i, s in enumerate(subjects):
    y = spi_tab[i]
    ok = np.isfinite(y)
    slopes[i] = np.polyfit(x[ok], y[ok], 1)[0] if ok.sum() >= 3 else np.nan
    print(f"{s:<6d}" + "".join(f"{v:>+10.4f}" for v in y)
          + f"{slopes[i]:>+10.4f}")
print(f"{'mean':6s}" + "".join(
    f"{np.nanmean(spi_tab[:, j]):>+10.4f}" for j in range(len(bands))))

d = slopes[np.isfinite(slopes)]
rng = np.random.default_rng(0)
null = (rng.choice([-1.0, 1.0], size=(100000, len(d))) * d[None, :]).mean(1)
p_perm = (np.sum(np.abs(null) >= abs(d.mean())) + 1) / 100001
W, p_w = sps.wilcoxon(d)
print(f"\nTREND TEST on per-subject SPI-vs-log-eccentricity slope (n={len(d)})")
print(f"  mean slope = {d.mean():+.4f}   (positive = magno preference grows "
      f"toward the periphery)")
print(f"  sign-flip permutation p = {p_perm:.5f}")
print(f"  Wilcoxon W={W:.0f}, p={p_w:.4f}")
print(f"  Cohen's dz = {d.mean() / d.std(ddof=1):+.2f}")
print(f"  subjects with positive slope = {int((d > 0).sum())}/{len(d)}")

# Direct fovea-vs-periphery contrast as a simpler, pre-declared companion
fov, per = spi_tab[:, 0], spi_tab[:, -1]
ok = np.isfinite(fov) & np.isfinite(per)
W2, p2 = sps.wilcoxon(per[ok] - fov[ok])
print(f"\n  companion contrast  SPI(ecc_4) - SPI(ecc_0.5) = "
      f"{np.mean(per[ok] - fov[ok]):+.4f}, W={W2:.0f}, p={p2:.4f}, "
      f"{int((per[ok] > fov[ok]).sum())}/{ok.sum()} subjects positive")
