"""Quantify the magno/parvo x ROI crossover in the completed subjects.

Statistic: per-voxel unique-variance difference (magno - parvo), summarized
per ROI, then the difference-of-differences between a dorsal-leaning ROI set
and a ventral-leaning one. With only 2 subjects this is descriptive only --
no group inference is possible -- but the direction and its consistency
across independently fitted subjects is informative.
"""

import numpy as np
from scipy import stats as sps

STREAMS = {1: "early", 2: "midventral", 3: "midlateral", 4: "midparietal",
           5: "ventral", 6: "lateral", 7: "parietal"}
DORSAL = ["midparietal", "parietal"]
VENTRAL = ["midventral", "ventral"]
LABEL_OF = {v: k for k, v in STREAMS.items()}

rows = {}
for subj in (4, 8):
    z = np.load(f"results/encoding/untrained_seed0_subj{subj:02d}.npz",
                allow_pickle=True)
    nc, roi = z["noise_ceiling"], z["roi_streams"]
    um, up = z["unique_magno"], z["unique_parvo"]
    rel = nc > 0.1
    per_roi = {}
    for name, lab in LABEL_OF.items():
        v = (roi == lab) & rel
        if v.sum() < 20:
            continue
        d = um[v] - up[v]
        d = d[np.isfinite(d)]
        # Wilcoxon across voxels is anti-conservative (voxels are not
        # independent); reported only as a within-subject descriptor.
        p = float(sps.wilcoxon(d).pvalue) if len(d) > 20 else np.nan
        per_roi[name] = {"n": int(v.sum()), "median_diff": float(np.median(d)),
                         "p_voxelwise": p}
    rows[subj] = per_roi

print(f"{'ROI':<13}" + "".join(f"{'subj%02d' % s:>22}" for s in rows))
print(f"{'':13}" + "".join(f"{'median(uM-uP)   p':>22}" for _ in rows))
for name in STREAMS.values():
    if not all(name in rows[s] for s in rows):
        continue
    line = f"{name:<13}"
    for s in rows:
        r = rows[name and s][name]
        line += f"{r['median_diff']:>+13.4f}{r['p_voxelwise']:>9.1e}"
    print(line)

print("\ndifference-of-differences (dorsal minus ventral), per subject:")
for s in rows:
    dor = np.mean([rows[s][n]["median_diff"] for n in DORSAL if n in rows[s]])
    ven = np.mean([rows[s][n]["median_diff"] for n in VENTRAL if n in rows[s]])
    print(f"  subj{s:02d}: dorsal {dor:+.4f} - ventral {ven:+.4f} "
          f"= {dor - ven:+.4f}")
print("\n(positive = magno relatively favours dorsal-leaning ROIs, "
      "parvo favours ventral-leaning ROIs)")
print("n=2 subjects: descriptive only, no group inference.")
