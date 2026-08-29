"""Is the blur-control verdict an artefact of SPI's bounded range?

In the no-blur control the magno (raw luminance) stream dominates unique
variance nearly everywhere, so SPI = (Um-Up)/(Um+Up) sits at +0.44..+0.80 --
well into the compressive tail of the ratio. A real eccentricity gradient
could be present there and still look flat in SPI.

This re-tests the gradient with statistics that do not compress:
  raw      median(Um - Up)          unbounded, but scales with overall fit
  norm     median(Um - Up) / median(Um + Up)   per band, scale-free, unbounded
  logratio median(log((Um+eps)/(Up+eps)))      symmetric, unbounded

If the gradient is absent under ALL of these, the collapse is real and not a
ceiling effect.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sps

from src.analysis.encoding.stats import load_results

MAIN, CTRL = "untrained_seed0", "untrained_seed0_noblur"
CENTRE = {1: 0.25, 2: 0.75, 3: 1.5, 4: 3.0, 5: 6.0}
NC_MIN, MIN_VOX, EPS = 0.1, 20, 1e-4

res = {m: load_results("results/encoding", m) for m in (MAIN, CTRL)}
subs = sorted(set(res[MAIN]) & set(res[CTRL]))


def band_stat(z, band, kind):
    um, up = z["unique_magno"], z["unique_parvo"]
    v = ((z["roi_prf-eccrois"] == band) & (z["noise_ceiling"] > NC_MIN)
         & np.isfinite(um) & np.isfinite(up))
    if v.sum() < MIN_VOX:
        return np.nan
    a, b = um[v], up[v]
    if kind == "raw":
        return np.median(a - b)
    if kind == "norm":
        den = np.median(a + b)
        return np.median(a - b) / den if den > EPS else np.nan
    if kind == "logratio":
        return np.median(np.log((np.clip(a, 0, None) + EPS)
                                / (np.clip(b, 0, None) + EPS)))
    if kind == "spi":
        w = v & ((um + up) > 1e-3)
        if w.sum() < MIN_VOX:
            return np.nan
        r = (um[w] - up[w]) / (um[w] + up[w])
        return np.median(r[np.isfinite(r)])
    raise ValueError(kind)


x = np.log([CENTRE[b] for b in sorted(CENTRE)])

for kind in ("spi", "raw", "norm", "logratio"):
    print(f"\n{'=' * 70}\nstatistic: {kind}\n{'=' * 70}")
    sl = {}
    for m in (MAIN, CTRL):
        prof = np.array([[band_stat(res[m][s], b, kind)
                          for b in sorted(CENTRE)] for s in subs])
        s_ = np.array([np.polyfit(x[np.isfinite(r)], r[np.isfinite(r)], 1)[0]
                       if np.isfinite(r).sum() >= 3 else np.nan for r in prof])
        sl[m] = s_
        lab = "blur (main)" if m == MAIN else "NO blur (control)"
        print(f"  [{lab}]")
        print("    band  " + "  ".join(f"{b:>8}" for b in
                                       ("0.5", "1", "2", "4", "4+")))
        print("    val   " + "  ".join(f"{v:+8.4f}"
                                       for v in np.nanmean(prof, 0)))
        g = s_[np.isfinite(s_)]
        W, p = sps.wilcoxon(g)
        print(f"    slope on log-ecc: mean={g.mean():+.4f}  p={p:.4f}  "
              f"{int((g > 0).sum())}/{len(g)}  "
              f"dz={g.mean() / g.std(ddof=1):+.2f}")
    d = (sl[MAIN] - sl[CTRL])
    d = d[np.isfinite(d)]
    print(f"  --> slope reduction without blur: {d.mean():+.4f}  "
          f"p={sps.wilcoxon(d)[1]:.4f}  {int((d > 0).sum())}/{len(d)}")
    keep = np.nanmean(sl[CTRL]) / np.nanmean(sl[MAIN]) * 100
    print(f"  --> fraction of the gradient that survives: {keep:.0f}%")
