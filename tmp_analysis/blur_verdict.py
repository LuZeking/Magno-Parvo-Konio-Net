"""Final blur-control verdict table: every headline contrast, both statistics.

The bounded SPI ratio and the unbounded paired difference disagree about the
eccentricity gradient, and they disagree for a reason that matters:

  raw   = median(Um - Up)                unbounded; "does magno's absolute
                                         advantage grow?"
  norm  = median(Um - Up)/median(Um+Up)  scale-free; "does the relative
                                         stream preference shift?"

In the no-blur control magno dominates almost everywhere, so SPI sits near
its ceiling and compresses. Any contrast is therefore reported under BOTH,
and a result is only claimed where the two agree.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sps

from src.analysis.encoding import nsd_data
from src.analysis.encoding.stats import load_results

MAIN, CTRL = "untrained_seed0", "untrained_seed0_noblur"
CENTRE = {1: 0.25, 2: 0.75, 3: 1.5, 4: 3.0, 5: 6.0}
VIS = {1: "V1v", 2: "V1d", 3: "V2v", 4: "V2d", 5: "V3v", 6: "V3d", 7: "hV4"}
STREAMS = {1: "early", 2: "midventral", 3: "midlateral", 4: "midparietal",
           5: "ventral", 6: "lateral", 7: "parietal"}
NC_MIN, MIN_VOX = 0.1, 20

res = {m: load_results("results/encoding", m) for m in (MAIN, CTRL)}
subs = sorted(set(res[MAIN]) & set(res[CTRL]))
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
    """a = effect under blur, b = effect without blur, per subject."""
    ma, pa, na, N = wtest(a)
    mb, pb, nb, _ = wtest(b)
    # Compare MAGNITUDES: several contrasts are negative by construction, so
    # a raw a-b difference would call an effect that GREW without blur
    # "weakened". The deflationary question is only ever about |effect|.
    d = (np.abs(a) - np.abs(b))[np.isfinite(a - b)]
    pd_ = sps.wilcoxon(d)[1] if len(d) >= 5 else np.nan
    same_sign = np.sign(ma) == np.sign(mb)
    if pb >= 0.05 or not same_sign:
        surv = "COLLAPSES"
    elif pd_ > 0.05:
        surv = "SURVIVES"
    elif d.mean() > 0:
        surv = "weakened"
    else:
        surv = "SURVIVES (larger)"
    print(f"  {name:<30} blur {ma:+.4f} (p={pa:.3f}, {na}/{N})   "
          f"noblur {mb:+.4f} (p={pb:.3f}, {nb}/{N})   "
          f"diff p={pd_:.3f}   {surv}")


for kind in ("raw", "norm"):
    print(f"\n{'=' * 108}\nSTATISTIC: {kind}\n{'=' * 108}")
    eff = {}
    for m in (MAIN, CTRL):
        R = res[m]
        # eccentricity slope
        x = np.log([CENTRE[b] for b in sorted(CENTRE)])
        prof = np.array([[stat(R[s], R[s]["roi_prf-eccrois"] == b, kind)
                          for b in sorted(CENTRE)] for s in subs])
        sl = np.array([np.polyfit(x[np.isfinite(r)], r[np.isfinite(r)], 1)[0]
                       if np.isfinite(r).sum() >= 3 else np.nan for r in prof])
        # sign reversal: is the foveal band parvo-preferring at all?
        fovea = prof[:, 0]
        # V1v -> hV4
        vp = np.array([[stat(R[s], R[s]["roi_prf-visualrois"] == k, kind)
                        for k in sorted(VIS)] for s in subs])
        cross = vp[:, 6] - vp[:, 0]
        # dorsal - ventral interaction
        names = [STREAMS[k] for k in sorted(STREAMS)]
        st = np.array([[stat(R[s], R[s]["roi_streams"] == k, kind)
                        for k in sorted(STREAMS)] for s in subs])
        ia = [names.index(n) for n in ("midparietal", "parietal")]
        ib = [names.index(n) for n in ("midventral", "ventral")]
        inter = np.nanmean(st[:, ia], 1) - np.nanmean(st[:, ib], 1)
        # hMT+ vs VO
        mt = np.array([stat(R[s], np.isin(kast[s], [12, 13]), kind)
                       for s in subs])
        vo = np.array([stat(R[s], np.isin(kast[s], [8, 9]), kind)
                       for s in subs])
        eff[m] = dict(slope=sl, fovea=fovea, cross=cross, inter=inter,
                      mt=mt, mtvo=mt - vo)

    line("ecc slope (log-ecc)", eff[MAIN]["slope"], eff[CTRL]["slope"])
    line("foveal band (<0 = parvo wins)", eff[MAIN]["fovea"], eff[CTRL]["fovea"])
    line("hV4 - V1v", eff[MAIN]["cross"], eff[CTRL]["cross"])
    line("interaction dorsal-ventral", eff[MAIN]["inter"], eff[CTRL]["inter"])
    line("hMT+ (TO1,TO2)", eff[MAIN]["mt"], eff[CTRL]["mt"])
    line("hMT+ - VO", eff[MAIN]["mtvo"], eff[CTRL]["mtvo"])

print("\nNote: 'SURVIVES' = significant without blur AND not significantly "
      "weakened by\nremoving it. 'weakened' = still significant without blur "
      "but reliably smaller.\n'COLLAPSES' = not significant once the blur is "
      "removed.")
