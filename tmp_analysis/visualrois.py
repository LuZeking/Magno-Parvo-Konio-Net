"""A3.3 -- stream preference across retinotopic areas V1-hV4 (n=8).

Prediction: parvocellular input is chromatically opponent and feeds the
colour/form pathway, so parvo-unique variance should rise (SPI should fall)
from V1 toward hV4.

Caveat recorded up front: hV4 is more foveally biased than V1-V3, so this
partly re-expresses the eccentricity gradient rather than testing something
independent of it. The V1v-vs-V1d style ventral/dorsal split within a single
area is the cleaner contrast and is reported alongside.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sps

from src.analysis.encoding.stats import load_results

VIS = {1: "V1v", 2: "V1d", 3: "V2v", 4: "V2d", 5: "V3v", 6: "V3d", 7: "hV4"}
NC_MIN = 0.1

res = load_results("results/encoding", "untrained_seed0")
subjects = sorted(res)
labs = sorted(VIS)

spi = np.full((len(subjects), len(labs)), np.nan)
um_t = np.full_like(spi, np.nan)
up_t = np.full_like(spi, np.nan)
nvox = np.zeros_like(spi, dtype=int)
for i, s in enumerate(subjects):
    z = res[s]
    roi, nc = z["roi_prf-visualrois"], z["noise_ceiling"]
    um, up = z["unique_magno"], z["unique_parvo"]
    for j, lab in enumerate(labs):
        v = (roi == lab) & (nc > NC_MIN) & ((um + up) > 1e-3)
        nvox[i, j] = int(v.sum())
        if v.sum() >= 20:
            r = ((um[v] - up[v]) / (um[v] + up[v]))
            spi[i, j] = np.median(r[np.isfinite(r)])
            um_t[i, j] = np.median(um[v])
            up_t[i, j] = np.median(up[v])

print(f"Median voxels per area (nc>{NC_MIN}): " + ", ".join(
    f"{VIS[l]}={int(np.median(nvox[:, j]))}" for j, l in enumerate(labs)))

print(f"\nMedian SPI (positive = magno-preferring)")
print(f"{'subj':6s}" + "".join(f"{VIS[l]:>9s}" for l in labs))
for i, s in enumerate(subjects):
    print(f"{s:<6d}" + "".join(f"{v:>+9.3f}" for v in spi[i]))
print(f"{'mean':6s}" + "".join(
    f"{np.nanmean(spi[:, j]):>+9.3f}" for j in range(len(labs))))

print(f"\nMedian unique variance / NC")
print(f"{'':6s}" + "".join(f"{VIS[l]:>9s}" for l in labs))
print(f"{'magno':6s}" + "".join(
    f"{np.nanmean(um_t[:, j]):>9.4f}" for j in range(len(labs))))
print(f"{'parvo':6s}" + "".join(
    f"{np.nanmean(up_t[:, j]):>9.4f}" for j in range(len(labs))))


def paired(a, b, name):
    ok = np.isfinite(a) & np.isfinite(b)
    d = (a - b)[ok]
    W, p = sps.wilcoxon(d)
    print(f"  {name:<26} mean diff={d.mean():+.4f}  W={W:.0f}  p={p:.4f}  "
          f"{int((d > 0).sum())}/{ok.sum()} positive")


k = {VIS[l]: j for j, l in enumerate(labs)}
print("\nPaired contrasts across subjects (n=8)")
v1 = np.nanmean(spi[:, [k["V1v"], k["V1d"]]], axis=1)
v123 = np.nanmean(spi[:, [k[n] for n in ("V1v", "V1d", "V2v", "V2d",
                                          "V3v", "V3d")]], axis=1)
paired(spi[:, k["hV4"]], v1, "SPI hV4 - V1")
paired(spi[:, k["hV4"]], v123, "SPI hV4 - V1/V2/V3")
# ventral vs dorsal banks of the SAME areas: matched eccentricity, so this
# is the contrast that is NOT explained by the foveal bias of hV4
ven = np.nanmean(spi[:, [k[n] for n in ("V1v", "V2v", "V3v")]], axis=1)
dor = np.nanmean(spi[:, [k[n] for n in ("V1d", "V2d", "V3d")]], axis=1)
paired(dor, ven, "SPI dorsal - ventral bank")
