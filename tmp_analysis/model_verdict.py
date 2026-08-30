"""Headline encoding contrasts for any two models, both statistics.

Generalises tmp_analysis/blur_verdict.py, which hardcoded the blur control.
The question there was "does removing the low-pass front end kill the effect?";
here it is "does the effect look the same in a TRAINED network as in an
untrained one?" -- the same table answers both.

  raw   = median(Um - Up)                unbounded; absolute advantage
  norm  = median(Um - Up)/median(Um+Up)  scale-free; relative preference

Both are reported because the bounded ratio compresses when one stream
dominates, which is exactly what happens once magno is unhandicapped (§8).
A contrast is only claimed where the two agree.

Usage:  PYTHONPATH=. python tmp_analysis/model_verdict.py MODEL_A MODEL_B
"""

from __future__ import annotations

import sys

import numpy as np
from scipy import stats as sps

from src.analysis.encoding import nsd_data
from src.analysis.encoding.stats import load_results

A = sys.argv[1] if len(sys.argv) > 1 else "untrained_seed0"
B = sys.argv[2] if len(sys.argv) > 2 else "thesis_dvd_trans"

CENTRE = {1: 0.25, 2: 0.75, 3: 1.5, 4: 3.0, 5: 6.0}
VIS = {1: "V1v", 2: "V1d", 3: "V2v", 4: "V2d", 5: "V3v", 6: "V3d", 7: "hV4"}
STREAMS = {1: "early", 2: "midventral", 3: "midlateral", 4: "midparietal",
           5: "ventral", 6: "lateral", 7: "parietal"}
NC_MIN, MIN_VOX = 0.1, 20

res = {m: load_results("results/encoding", m) for m in (A, B)}
subs = sorted(set(res[A]) & set(res[B]))
print(f"A = {A}   (n={len(res[A])})")
print(f"B = {B}   (n={len(res[B])})")
print(f"paired subjects: {subs}\n")
if len(subs) < 5:
    print("fewer than 5 paired subjects -- Wilcoxon not run")
    sys.exit(0)

kast = {s: nsd_data.load_roi_labels(s, "Kastner2015",
                                    nsd_data.load_mask(s, "nsdgeneral"))
        for s in subs}


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


def wtest(x):
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return np.nan, np.nan, 0, len(x)
    return x.mean(), sps.wilcoxon(x)[1], int((x > 0).sum()), len(x)


def line(name, a, b):
    ma, pa, na, N = wtest(a)
    mb, pb, nb, _ = wtest(b)
    d = (a - b)[np.isfinite(a - b)]
    pd_ = sps.wilcoxon(d)[1] if len(d) >= 5 else np.nan
    if np.isnan(pb):
        tag = "n/a"
    elif pb >= 0.05:
        tag = "absent in B"
    elif np.sign(ma) != np.sign(mb):
        tag = "SIGN FLIP"
    elif pd_ > 0.05:
        tag = "same in both"
    else:
        tag = "larger in B" if abs(mb) > abs(ma) else "smaller in B"
    print(f"  {name:<30} A {ma:+.4f} (p={pa:.3f}, {na}/{N})   "
          f"B {mb:+.4f} (p={pb:.3f}, {nb}/{N})   A-B p={pd_:.3f}   {tag}")


for kind in ("raw", "norm"):
    print(f"\n{'=' * 110}\nSTATISTIC: {kind}\n{'=' * 110}")
    eff = {}
    for m in (A, B):
        R = res[m]
        x = np.log([CENTRE[b] for b in sorted(CENTRE)])
        prof = np.array([[stat(R[s], R[s]["roi_prf-eccrois"] == b, kind)
                          for b in sorted(CENTRE)] for s in subs])
        sl = np.array([np.polyfit(x[np.isfinite(r)], r[np.isfinite(r)], 1)[0]
                       if np.isfinite(r).sum() >= 3 else np.nan for r in prof])
        vp = np.array([[stat(R[s], R[s]["roi_prf-visualrois"] == k, kind)
                        for k in sorted(VIS)] for s in subs])
        names = [STREAMS[k] for k in sorted(STREAMS)]
        st = np.array([[stat(R[s], R[s]["roi_streams"] == k, kind)
                        for k in sorted(STREAMS)] for s in subs])
        ia = [names.index(n) for n in ("midparietal", "parietal")]
        ib = [names.index(n) for n in ("midventral", "ventral")]
        mt = np.array([stat(R[s], np.isin(kast[s], [12, 13]), kind)
                       for s in subs])
        vo = np.array([stat(R[s], np.isin(kast[s], [8, 9]), kind)
                       for s in subs])
        eff[m] = dict(slope=sl, fovea=prof[:, 0], cross=vp[:, 6] - vp[:, 0],
                      inter=np.nanmean(st[:, ia], 1) - np.nanmean(st[:, ib], 1),
                      mt=mt, mtvo=mt - vo)
    for key, label in (("slope", "ecc slope (log-ecc)"),
                       ("fovea", "foveal band (<0 = parvo)"),
                       ("cross", "hV4 - V1v  [HEADLINE]"),
                       ("inter", "interaction dorsal-ventral"),
                       ("mt", "hMT+ (TO1,TO2)"),
                       ("mtvo", "hMT+ - VO")):
        line(label, eff[A][key], eff[B][key])

print("\nA-B p is a paired Wilcoxon on the per-subject difference between "
      "models.\n'same in both' = significant in B and not reliably different "
      "from A.")
