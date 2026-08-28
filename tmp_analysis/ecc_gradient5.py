"""The eccentricity gradient with ALL FIVE prf-eccrois bands.

`stats.py` defines ECC_LABELS with four bands (0.5/1/2/4). The atlas has a
fifth, `ecc4+`, covering everything beyond 4 deg -- the most peripheral and
therefore the most magno-diagnostic voxels in the mask. Every eccentricity
number reported so far silently dropped it. This re-runs the gradient on all
five bands and prints both versions side by side so the change is auditable.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sps

from src.analysis.encoding.stats import load_results

# NSD stimuli span 8.4 deg, so the open band 4+ is 4-8.4; use its midpoint.
CENTRE = {1: 0.25, 2: 0.75, 3: 1.5, 4: 3.0, 5: 6.0}
NAME = {1: "0.5", 2: "1", 3: "2", 4: "4", 5: "4+"}
NC_MIN = 0.1
MIN_VOX = 20

res = load_results("results/encoding", "untrained_seed0")
subs = sorted(res)


def gradient(bands):
    """Per-subject median SPI per band, and the slope on log eccentricity."""
    spi = np.full((len(subs), len(bands)), np.nan)
    nv = np.zeros((len(subs), len(bands)), int)
    for i, s in enumerate(subs):
        z = res[s]
        ecc, nc = z["roi_prf-eccrois"], z["noise_ceiling"]
        um, up = z["unique_magno"], z["unique_parvo"]
        for j, b in enumerate(bands):
            v = (ecc == b) & (nc > NC_MIN) & ((um + up) > 1e-3)
            nv[i, j] = v.sum()
            if v.sum() >= MIN_VOX:
                r = (um[v] - up[v]) / (um[v] + up[v])
                spi[i, j] = np.median(r[np.isfinite(r)])
    x = np.log([CENTRE[b] for b in bands])
    slope = np.array([np.polyfit(x[np.isfinite(row)], row[np.isfinite(row)], 1)[0]
                      if np.isfinite(row).sum() >= 3 else np.nan for row in spi])
    return spi, slope, nv


for label, bands in (("4 bands (as reported)", [1, 2, 3, 4]),
                     ("5 bands (with ecc4+)", [1, 2, 3, 4, 5])):
    spi, slope, nv = gradient(bands)
    print(f"\n=== {label} ===")
    print("  band     mean SPI   med n_vox")
    for j, b in enumerate(bands):
        print(f"  {NAME[b]:<6} {np.nanmean(spi[:, j]):+.4f}     "
              f"{int(np.median(nv[:, j]))}")
    sl = slope[np.isfinite(slope)]
    W, p = sps.wilcoxon(sl)
    rng = np.random.default_rng(0)
    null = (rng.choice([-1.0, 1.0], size=(100000, len(sl))) * sl[None, :]).mean(1)
    pp = (np.sum(np.abs(null) >= abs(sl.mean())) + 1) / 100001
    print(f"  slope on log-ecc: mean={sl.mean():+.4f}  p={p:.4f}  "
          f"p_perm={pp:.5f}  {int((sl > 0).sum())}/{len(sl)} pos  "
          f"dz={sl.mean() / sl.std(ddof=1):+.2f}")
    a, b_ = spi[:, 0], spi[:, -1]
    d = b_ - a
    d = d[np.isfinite(d)]
    print(f"  SPI(outer) - SPI(0.5deg): mean={d.mean():+.4f}  "
          f"{int((d > 0).sum())}/{len(d)} pos  p={sps.wilcoxon(d)[1]:.4f}")
