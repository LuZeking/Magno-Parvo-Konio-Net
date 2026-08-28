"""Group test using the paired within-voxel statistic, for consistency.

stats.py summarizes each ROI as median(U_m) - median(U_p) (a difference of
medians). The eccentricity/robustness analyses and the figure instead use
median(U_m - U_p) (the median of the paired per-voxel difference). Both are
defensible and they agree here, but reporting must not mix them.

This re-runs the pre-declared interaction and the simple effects with the
paired statistic throughout, which is the one the figure draws and the more
natural summary for a within-voxel contrast.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sps

from src.analysis.encoding.stats import load_results

STREAMS = {1: "early", 2: "midventral", 3: "midlateral", 4: "midparietal",
           5: "ventral", 6: "lateral", 7: "parietal"}
DORSAL, VENTRAL = ["midparietal", "parietal"], ["midventral", "ventral"]
NC_MIN = 0.1

res = load_results("results/encoding", "untrained_seed0")
subs = sorted(res)
names = [STREAMS[k] for k in sorted(STREAMS)]

tab = np.full((len(subs), len(names)), np.nan)
for i, s in enumerate(subs):
    z = res[s]
    roi, nc = z["roi_streams"], z["noise_ceiling"]
    d = z["unique_magno"] - z["unique_parvo"]
    for j, lab in enumerate(sorted(STREAMS)):
        v = (roi == lab) & (nc > NC_MIN)
        dv = d[v][np.isfinite(d[v])]
        if dv.size >= 20:
            tab[i, j] = np.median(dv)


def report(d, label):
    d = d[np.isfinite(d)]
    W, p = sps.wilcoxon(d)
    rng = np.random.default_rng(0)
    null = (rng.choice([-1.0, 1.0], size=(100000, len(d))) * d[None, :]).mean(1)
    pp = (np.sum(np.abs(null) >= abs(d.mean())) + 1) / 100001
    print(f"  {label:<34} mean={d.mean():+.4f}  W={W:.0f}  p={p:.4f}  "
          f"p_perm={pp:.5f}  {int((d > 0).sum())}/{len(d)}  "
          f"dz={d.mean() / d.std(ddof=1):+.2f}")


print("Per-ROI paired statistic  median(U_m - U_p), across subjects (n=8)")
for j, n in enumerate(names):
    report(tab[:, j], n)

ia = [names.index(n) for n in DORSAL]
ib = [names.index(n) for n in VENTRAL]
print("\nPre-declared interaction and its two simple effects")
report(np.nanmean(tab[:, ia], 1) - np.nanmean(tab[:, ib], 1),
       "INTERACTION dorsal - ventral")
report(np.nanmean(tab[:, ia], 1), "simple effect: dorsal")
report(np.nanmean(tab[:, ib], 1), "simple effect: ventral")
