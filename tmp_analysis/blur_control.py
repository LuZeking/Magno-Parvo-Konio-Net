"""THE GATING CONTROL: does the magno stream's low-pass front end explain it?

Every neural result so far is consistent with a deflationary account. The
magno stream is low-pass BY CONSTRUCTION (Gaussian blur sigma=4 plus a
48-month scale-free CSF cut), so of course it predicts peripheral, low-SF,
non-chromatic voxels better -- no biology required.

`untrained_seed0_noblur` reruns the identical pipeline and the identical
untrained network with --no-lum-blur --no-lum-cs, i.e. magno = raw luminance
and parvo = raw L-M. The two models therefore differ ONLY in whether the
luminance channel is low-passed.

  survives -> the effect is carried by achromatic-vs-chromatic channel
              content, which is the real biological claim
  collapses -> the effect is a restatement of the blur, and must be demoted
              from a finding to a front-end validation

All four headline results are re-tested side by side. Reported per subject
(n=8, the unit of inference) with the paired within-voxel statistic
throughout, matching tmp_analysis/group_interaction_paired.py.
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
DORSAL, VENTRAL = ["midparietal", "parietal"], ["midventral", "ventral"]
KAST_MT = [12, 13]      # TO2, TO1  = hMT+
KAST_VO = [8, 9]        # VO1, VO2
NC_MIN, MIN_VOX = 0.1, 20

res = {m: load_results("results/encoding", m) for m in (MAIN, CTRL)}
subs = sorted(set(res[MAIN]) & set(res[CTRL]))
print(f"models: {MAIN} vs {CTRL} | common subjects: {len(subs)}\n")

# Kastner labels are not stored in the npz; index them on the shared mask.
kast = {s: nsd_data.load_roi_labels(s, "Kastner2015",
                                    nsd_data.load_mask(s, "nsdgeneral"))
        for s in subs}


def per_subject(model, fn):
    return np.array([fn(res[model][s], kast[s]) for s in subs], float)


def med_spi(z, roi, lab):
    um, up = z["unique_magno"], z["unique_parvo"]
    v = (roi == lab) & (z["noise_ceiling"] > NC_MIN) & ((um + up) > 1e-3)
    if v.sum() < MIN_VOX:
        return np.nan
    r = (um[v] - up[v]) / (um[v] + up[v])
    return np.median(r[np.isfinite(r)])


def med_diff(z, sel):
    d = z["unique_magno"] - z["unique_parvo"]
    v = sel & (z["noise_ceiling"] > NC_MIN) & np.isfinite(d)
    return np.median(d[v]) if v.sum() >= MIN_VOX else np.nan


def test(x, label, indent="  "):
    x = x[np.isfinite(x)]
    if len(x) < 5:
        print(f"{indent}{label:<32} n={len(x)} too few")
        return np.nan
    W, p = sps.wilcoxon(x)
    rng = np.random.default_rng(0)
    null = (rng.choice([-1.0, 1.0], size=(100000, len(x))) * x[None, :]).mean(1)
    pp = (np.sum(np.abs(null) >= abs(x.mean())) + 1) / 100001
    print(f"{indent}{label:<32} mean={x.mean():+.4f}  p={p:.4f}  "
          f"p_perm={pp:.5f}  {int((x > 0).sum())}/{len(x)}  "
          f"dz={x.mean() / x.std(ddof=1):+.2f}")
    return x.mean()


def paired(a, b, label):
    """Is the effect itself different between the two models?"""
    d = (a - b)[np.isfinite(a - b)]
    W, p = sps.wilcoxon(d)
    print(f"    {label:<30} delta={d.mean():+.4f}  p={p:.4f}  "
          f"{int((d > 0).sum())}/{len(d)} larger with blur")


# ---- 1. eccentricity gradient ------------------------------------------
print("=" * 74)
print("1. ECCENTRICITY GRADIENT (5 bands, SPI on log-eccentricity)")
print("=" * 74)
slopes = {}
for m in (MAIN, CTRL):
    spi = np.array([[med_spi(res[m][s], res[m][s]["roi_prf-eccrois"], b)
                     for b in sorted(CENTRE)] for s in subs])
    x = np.log([CENTRE[b] for b in sorted(CENTRE)])
    sl = np.array([np.polyfit(x[np.isfinite(r)], r[np.isfinite(r)], 1)[0]
                   if np.isfinite(r).sum() >= 3 else np.nan for r in spi])
    slopes[m] = sl
    print(f"\n  [{m}]")
    print("    band   " + "  ".join(f"{b:>7}" for b in ("0.5", "1", "2", "4", "4+")))
    print("    SPI    " + "  ".join(f"{v:+7.3f}" for v in np.nanmean(spi, 0)))
    test(sl, "slope on log-ecc", indent="    ")
    test(spi[:, -1] - spi[:, 0], "SPI(4+) - SPI(0.5)", indent="    ")
print()
paired(slopes[MAIN], slopes[CTRL], "slope: blur - noblur")

# ---- 2. V1 -> hV4 -------------------------------------------------------
print("\n" + "=" * 74)
print("2. V1 -> hV4 CROSSOVER (prf-visualrois)")
print("=" * 74)
cross = {}
for m in (MAIN, CTRL):
    spi = np.array([[med_spi(res[m][s], res[m][s]["roi_prf-visualrois"], k)
                     for k in sorted(VIS)] for s in subs])
    cross[m] = spi[:, 6] - spi[:, 0]     # hV4 - V1v
    print(f"\n  [{m}]")
    print("    area   " + "  ".join(f"{VIS[k]:>7}" for k in sorted(VIS)))
    print("    SPI    " + "  ".join(f"{v:+7.3f}" for v in np.nanmean(spi, 0)))
    test(cross[m], "SPI(hV4) - SPI(V1v)", indent="    ")
print()
paired(cross[MAIN], cross[CTRL], "crossover: blur - noblur")

# ---- 3. pre-declared dorsal/ventral interaction -------------------------
print("\n" + "=" * 74)
print("3. PRE-DECLARED dorsal-vs-ventral INTERACTION (streams)")
print("=" * 74)
inter = {}
for m in (MAIN, CTRL):
    names = [STREAMS[k] for k in sorted(STREAMS)]
    tab = np.array([[med_diff(res[m][s], res[m][s]["roi_streams"] == k)
                     for k in sorted(STREAMS)] for s in subs])
    ia = [names.index(n) for n in DORSAL]
    ib = [names.index(n) for n in VENTRAL]
    inter[m] = np.nanmean(tab[:, ia], 1) - np.nanmean(tab[:, ib], 1)
    print(f"\n  [{m}]")
    for j, n in enumerate(names):
        print(f"    {n:<12} {np.nanmean(tab[:, j]):+.4f}")
    test(inter[m], "INTERACTION dorsal - ventral", indent="    ")
    test(np.nanmean(tab[:, ia], 1), "simple: dorsal", indent="    ")
    test(np.nanmean(tab[:, ib], 1), "simple: ventral", indent="    ")
print()
paired(inter[MAIN], inter[CTRL], "interaction: blur - noblur")

# ---- 4. hMT+ and VO (Kastner) ------------------------------------------
print("\n" + "=" * 74)
print("4. hMT+ (TO1/TO2) and VO (VO1/VO2), Kastner2015")
print("=" * 74)
mt = {}
for m in (MAIN, CTRL):
    print(f"\n  [{m}]")
    mt[m] = per_subject(m, lambda z, k: med_diff(z, np.isin(k, KAST_MT)))
    vo = per_subject(m, lambda z, k: med_diff(z, np.isin(k, KAST_VO)))
    test(mt[m], "hMT+ (TO1,TO2)", indent="    ")
    test(vo, "VO (VO1,VO2)", indent="    ")
    test(mt[m] - vo, "INTERACTION hMT+ - VO", indent="    ")
print()
paired(mt[MAIN], mt[CTRL], "hMT+: blur - noblur")

# ---- 5. does the control model fit the brain at all? -------------------
print("\n" + "=" * 74)
print("5. SANITY: overall model quality (median R2_joint/NC, NC>0.1)")
print("=" * 74)
for m in (MAIN, CTRL):
    q = np.array([np.median(res[m][s]["r2_joint_nc"][
        res[m][s]["noise_ceiling"] > NC_MIN]) for s in subs])
    print(f"  {m:<28} median={np.mean(q):.4f}  "
          f"range {q.min():.3f}-{q.max():.3f}")
