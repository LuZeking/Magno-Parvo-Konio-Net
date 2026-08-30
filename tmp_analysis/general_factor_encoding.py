"""Is hV4-V1v a real regional preference, or a global stream bias in disguise?

This is the confound that killed the thesis's RSA (§0) and reappeared in the
blur control (§8). It has to be asked again of the TRAINED model, and there it
is sharper than before: training flips global dominance from magno to parvo, so
the hV4-V1v effect (hV4 more parvo than V1v) now points in the SAME direction
as the global bias. That is exactly the situation in which a general factor can
masquerade as regional specialisation.

Three things distinguish the two accounts:

1. CROSSOVER. If V1v is magno-preferring and hV4 parvo-preferring, the ROIs sit
   on opposite sides of zero and no single global scaling can produce that. If
   both are parvo-preferring and hV4 merely more so, the effect is graded and a
   general factor remains a live alternative.

2. RANK vs MAGNITUDE. Subtracting each subject's own across-ROI mean removes any
   global additive stream bias. Whatever survives is regional by construction.

3. MONOTONICITY along the hierarchy. A general factor predicts no particular
   ordering of V1->V2->V3->hV4; a hierarchical account predicts a gradient.

Usage: PYTHONPATH=. python tmp_analysis/general_factor_encoding.py [MODEL ...]
"""

from __future__ import annotations

import sys

import numpy as np
from scipy import stats as sps

from src.analysis.encoding.stats import load_results

MODELS = sys.argv[1:] or ["untrained_seed0", "thesis_dvd_trans"]
VIS = {1: "V1v", 2: "V1d", 3: "V2v", 4: "V2d", 5: "V3v", 6: "V3d", 7: "hV4"}
ORDER = [1, 2, 3, 4, 5, 6, 7]
NC_MIN, MIN_VOX = 0.1, 20


def stat(z, sel, kind="norm"):
    um, up = z["unique_magno"], z["unique_parvo"]
    v = sel & (z["noise_ceiling"] > NC_MIN) & np.isfinite(um) & np.isfinite(up)
    if v.sum() < MIN_VOX:
        return np.nan
    a, b = um[v], up[v]
    if kind == "raw":
        return np.median(a - b)
    den = np.median(a + b)
    return np.median(a - b) / den if den > 1e-4 else np.nan


for model in MODELS:
    R = load_results("results/encoding", model)
    subs = sorted(R)
    print(f"\n{'=' * 78}\n{model}   (n={len(subs)})\n{'=' * 78}")

    for kind in ("raw", "norm"):
        prof = np.array([[stat(R[s], R[s]["roi_prf-visualrois"] == k, kind)
                          for k in ORDER] for s in subs])
        gmean = np.nanmean(prof, axis=1, keepdims=True)   # per-subject global
        centred = prof - gmean

        print(f"\n  [{kind}]  per-subject global stream bias: "
              f"mean {np.nanmean(gmean):+.4f}  "
              f"({int((gmean > 0).sum())}/{len(subs)} magno-dominant)")
        print(f"  {'ROI':<6}{'value':>9}{'p':>8}{'sign':>7}   "
              f"{'centred':>9}{'p':>8}{'sign':>7}")
        for j, k in enumerate(ORDER):
            v, c = prof[:, j], centred[:, j]
            v, c = v[np.isfinite(v)], c[np.isfinite(c)]
            pv = sps.wilcoxon(v)[1] if len(v) >= 5 else np.nan
            pc = sps.wilcoxon(c)[1] if len(c) >= 5 else np.nan
            print(f"  {VIS[k]:<6}{v.mean():>+9.4f}{pv:>8.3f}"
                  f"{int((v > 0).sum()):>4}/{len(v):<2}"
                  f"{c.mean():>+9.4f}{pc:>8.3f}"
                  f"{int((c > 0).sum()):>4}/{len(c):<2}")

        # 1. crossover
        v1, hv4 = prof[:, 0], prof[:, 6]
        ok = np.isfinite(v1) & np.isfinite(hv4)
        cross = int(((v1[ok] > 0) & (hv4[ok] < 0)).sum())
        print(f"    crossover (V1v magno AND hV4 parvo): "
              f"{cross}/{ok.sum()} subjects")

        # 2. does the contrast survive global centring?
        d = (centred[:, 6] - centred[:, 0])
        d = d[np.isfinite(d)]
        print(f"    hV4-V1v after removing each subject's global bias: "
              f"{d.mean():+.4f}  p={sps.wilcoxon(d)[1]:.4f}  "
              f"{int((d > 0).sum())}/{len(d)}  "
              f"dz={d.mean() / d.std(ddof=1):+.2f}")

        # 3. monotonicity: rank correlation with hierarchical position
        pos = np.array([1, 1, 2, 2, 3, 3, 4], float)   # V1,V1,V2,V2,V3,V3,hV4
        rhos = []
        for r in prof:
            ok = np.isfinite(r)
            if ok.sum() >= 5:
                rhos.append(sps.spearmanr(pos[ok], r[ok]).statistic)
        rhos = np.array(rhos)
        print(f"    Spearman rho(hierarchy, value) per subject: "
              f"mean {rhos.mean():+.3f}  "
              f"{int((rhos < 0).sum())}/{len(rhos)} negative")
