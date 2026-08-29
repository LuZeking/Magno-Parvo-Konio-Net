"""Is the surviving V1->hV4 effect just eccentricity in disguise?

hV4 is foveally biased and V1v is not, so a hierarchical claim needs the two
compared at matched eccentricity. This restricts both ROIs to the SAME
prf-eccrois band and re-tests within band, in both models.

The control model matters most here: it is the one in which the eccentricity
gradient's sign reversal does not exist, so an effect that holds there is
already partly protected. But the within-band test is the direct answer.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sps

from src.analysis.encoding.stats import load_results

MAIN, CTRL = "untrained_seed0", "untrained_seed0_noblur"
BANDS = {1: "0.5", 2: "1", 3: "2", 4: "4", 5: "4+"}
V1V, HV4 = 1, 7          # prf-visualrois labels
NC_MIN, MIN_VOX = 0.1, 20

res = {m: load_results("results/encoding", m) for m in (MAIN, CTRL)}
subs = sorted(set(res[MAIN]) & set(res[CTRL]))


def stat(z, sel, kind):
    um, up = z["unique_magno"], z["unique_parvo"]
    v = sel & (z["noise_ceiling"] > NC_MIN) & np.isfinite(um) & np.isfinite(up)
    if v.sum() < MIN_VOX:
        return np.nan
    a, b = um[v], up[v]
    if kind == "raw":
        return np.median(a - b)
    den = np.median(a + b)
    return np.median(a - b) / den if den > 1e-4 else np.nan


def rep(x, label):
    x = x[np.isfinite(x)]
    if len(x) < 5:
        print(f"    {label:<26} n={len(x)} too few subjects")
        return
    print(f"    {label:<26} mean={x.mean():+.4f}  p={sps.wilcoxon(x)[1]:.4f}  "
          f"{int((x > 0).sum())}/{len(x)}  "
          f"dz={x.mean() / x.std(ddof=1):+.2f}")


for kind in ("raw", "norm"):
    print(f"\n{'=' * 70}\nSTATISTIC: {kind}\n{'=' * 70}")
    for m in (MAIN, CTRL):
        lab = "blur (main)" if m == MAIN else "NO blur (control)"
        print(f"\n  [{lab}]")
        R = res[m]
        # how many voxels does each ROI have per band?
        counts = np.zeros((2, len(BANDS)), int)
        for s in subs:
            z = R[s]
            vis, ecc = z["roi_prf-visualrois"], z["roi_prf-eccrois"]
            ok = z["noise_ceiling"] > NC_MIN
            for j, b in enumerate(sorted(BANDS)):
                counts[0, j] += int(((vis == V1V) & (ecc == b) & ok).sum())
                counts[1, j] += int(((vis == HV4) & (ecc == b) & ok).sum())
        print("    band        " + "  ".join(f"{BANDS[b]:>7}"
                                             for b in sorted(BANDS)))
        print("    V1v n       " + "  ".join(f"{c:7d}" for c in counts[0]))
        print("    hV4 n       " + "  ".join(f"{c:7d}" for c in counts[1]))

        for j, b in enumerate(sorted(BANDS)):
            d = np.array([
                stat(R[s], (R[s]["roi_prf-visualrois"] == HV4)
                     & (R[s]["roi_prf-eccrois"] == b), kind)
                - stat(R[s], (R[s]["roi_prf-visualrois"] == V1V)
                       & (R[s]["roi_prf-eccrois"] == b), kind)
                for s in subs])
            rep(d, f"hV4 - V1v within {BANDS[b]} deg")

        # Pooled within-band contrast: average the within-band differences
        # that exist for each subject, so eccentricity cannot drive it.
        pooled = []
        for s in subs:
            z = R[s]
            vals = []
            for b in sorted(BANDS):
                a = stat(z, (z["roi_prf-visualrois"] == HV4)
                         & (z["roi_prf-eccrois"] == b), kind)
                c = stat(z, (z["roi_prf-visualrois"] == V1V)
                         & (z["roi_prf-eccrois"] == b), kind)
                if np.isfinite(a) and np.isfinite(c):
                    vals.append(a - c)
            pooled.append(np.mean(vals) if vals else np.nan)
        rep(np.array(pooled), "POOLED within-band")
