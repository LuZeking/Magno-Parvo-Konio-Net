"""Robustness of the eccentricity gradient to analysis choices.

Three things the headline must survive:
 1. The NC threshold (plan asks for threshold robustness explicitly).
 2. The summary statistic. SPI is a RATIO, and in the developmental-EEG
    analysis the analogous ratio index degenerated to +-1 whenever the two
    terms had opposite signs. Unique variances should be >=0, but noise can
    push them negative, so check how often the denominator misbehaves and
    re-run the trend on the plain difference (Um - Up), which has no such
    failure mode.
 3. The central tendency (median vs mean over voxels).
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sps

from src.analysis.encoding.stats import ECC_LABELS, load_results

BAND_CENTRE = {1: 0.25, 2: 0.75, 3: 1.5, 4: 3.0}
res = load_results("results/encoding", "untrained_seed0")
subjects = sorted(res)
bands = sorted(ECC_LABELS)
x = np.log(np.array([BAND_CENTRE[b] for b in bands]))


def trend(statistic: str, nc_min: float, stat=np.median):
    slopes = []
    for s in subjects:
        z = res[s]
        roi, nc = z["roi_prf-eccrois"], z["noise_ceiling"]
        um, up = z["unique_magno"], z["unique_parvo"]
        y = []
        for b in bands:
            v = (roi == b) & (nc > nc_min)
            if statistic == "spi":
                v = v & ((um + up) > 1e-3)
                val = (um[v] - up[v]) / (um[v] + up[v])
            else:
                val = um[v] - up[v]
            val = val[np.isfinite(val)]
            y.append(stat(val) if val.size >= 20 else np.nan)
        y = np.array(y)
        ok = np.isfinite(y)
        slopes.append(np.polyfit(x[ok], y[ok], 1)[0] if ok.sum() >= 3
                      else np.nan)
    d = np.array(slopes)
    d = d[np.isfinite(d)]
    W, p = sps.wilcoxon(d)
    return d.mean(), p, int((d > 0).sum()), len(d), d.mean() / d.std(ddof=1)


print("How often is the SPI denominator unstable? (Um+Up near zero or "
      "negative, over voxels with NC>0.1)")
for s in subjects:
    z = res[s]
    nc = z["noise_ceiling"]
    um, up = z["unique_magno"], z["unique_parvo"]
    v = nc > 0.1
    den = (um + up)[v]
    print(f"  subj{s:02d}: {100 * np.mean(den <= 1e-3):5.1f}% of voxels have "
          f"Um+Up <= 1e-3; {100 * np.mean(den < 0):5.1f}% strictly negative; "
          f"min Um={um[v].min():+.3f} min Up={up[v].min():+.3f}")

print("\nTrend test under varying choices "
      "(slope of statistic on log-eccentricity, n=8 subjects)")
print(f"{'statistic':<22}{'NC>':>6}{'central':>9}{'mean slope':>12}"
      f"{'p':>8}{'pos':>7}{'dz':>7}")
for statname, label in (("spi", "SPI (ratio)"),
                        ("diff", "Um-Up (difference)")):
    for nc_min in (0.05, 0.1, 0.2, 0.3):
        for stat, sname in ((np.median, "median"), (np.mean, "mean")):
            m, p, pos, n, dz = trend(statname, nc_min, stat)
            print(f"{label:<22}{nc_min:>6.2f}{sname:>9}{m:>+12.4f}"
                  f"{p:>8.4f}{f'{pos}/{n}':>7}{dz:>+7.2f}")
