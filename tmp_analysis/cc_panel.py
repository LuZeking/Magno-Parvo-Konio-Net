"""PAPER_PLAN §7.3b/§24: group analysis of the in-silico cue-conflict panel.

Two statistics, per the §8 rule, because VSB is a bounded ratio and the
project does not claim a result that only one statistic supports:

    norm = (eta2_shape - eta2_texture) / (eta2_shape + eta2_texture)   [VSB]
    raw  =  eta2_shape - eta2_texture                                  [unbounded]

Sections:
  1. LEG (a) cross-check: per-band stream dissociation. The primary
     measurement of leg (a) is src/analysis/behavior/stream_shape_texture.py
     (§24.7); this is the encoding-model consistency check, not a discovery.
  2. The cortical profile along V1 -> hV4/LO -- the actual §7.3b contribution.
  3. rho(quantity, SPI) across voxels, WITH the §24.3 circularity caveat.
  4. Profile after removing all monotone dependence on SPI (20 quantile bins).
  5. Profile after removing SPI *and* predictability (10x10 cells).

Usage: PYTHONPATH=. python tmp_analysis/cc_panel.py [MODEL ...]
"""
from __future__ import annotations
import glob, os, sys
import numpy as np
from scipy import stats as sps
from src.analysis.encoding import nsd_data

DIR = "results/encoding_cc"
VIS = {1: "V1v", 2: "V1d", 3: "V2v", 4: "V2d", 5: "V3v", 6: "V3d", 7: "hV4"}
# Verified against nsd/rois/subj03/freesurfer/label/Kastner2015.mgz.ctab:
# 14 = LO2, 15 = LO1 (taken as a union, so the order is cosmetic).
KAST_LO = {14: "LO2", 15: "LO1"}
NC_MIN, MIN_VOX = 0.1, 20
NAMES = [VIS[k] for k in sorted(VIS)] + ["LO"]

models = sys.argv[1:] or sorted({
    os.path.basename(f).rsplit("_subj", 1)[0] for f in glob.glob(f"{DIR}/*.npz")
})


def load(model):
    out = {}
    for f in sorted(glob.glob(f"{DIR}/{model}_subj*.npz")):
        out[int(f.rsplit("subj", 1)[1][:2])] = np.load(f)
    return out


def quantity(z, band, stat):
    """The per-voxel shape-vs-texture measure, in one of the two statistics."""
    es, et = z[f"cc_{band}_eta2_shape"], z[f"cc_{band}_eta2_texture"]
    return z[f"cc_{band}_vsb"] if stat == "norm" else es - et


def med(v, sel):
    v = v[sel & np.isfinite(v)]
    return np.median(v) if v.size >= MIN_VOX else np.nan


def wil(x):
    """mean, p, n_positive, n. Below n=5 the exact Wilcoxon cannot reach any
    useful p, so p is withheld -- but the positive COUNT is still reported.
    (An earlier version returned 0 here, which printed "0/4 positive" beside a
    +1.57 mean and read as a sign reversal.)"""
    x = np.asarray(x)[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan, 0, 0
    n_pos = int((x > 0).sum())
    if x.size < 5:
        return x.mean(), np.nan, n_pos, x.size
    return x.mean(), sps.wilcoxon(x)[1], n_pos, x.size


def report(label, x, extra=""):
    m, p, n, N = wil(x)
    ps = "n/a" if np.isnan(p) else f"{p:.4f}"
    print(f"   {label:<34} {m:+.4f}  p={ps}  {n}/{N} positive{extra}")


def kast_of(s):
    try:
        return nsd_data.load_roi_labels(
            s, "Kastner2015", nsd_data.load_mask(s, "nsdgeneral"))
    except Exception:
        return None


def roi_profile(vals, ok, z, s):
    vroi = z["roi_prf-visualrois"]
    row = [med(vals, ok & (vroi == k)) for k in sorted(VIS)]
    k = kast_of(s)
    row.append(np.nan if k is None else med(vals, ok & np.isin(k, list(KAST_LO))))
    return row


def show_profile(prof, tag):
    prof = np.array(prof)
    print("   " + "  ".join(f"{n:>7s}" for n in NAMES))
    print("   " + "  ".join(f"{x:+7.3f}" for x in np.nanmean(prof, 0)))
    report(f"{tag} hV4 - V1", prof[:, 6] - np.nanmean(prof[:, [0, 1]], 1))
    rho = np.array([sps.spearmanr(np.arange(7), r[:7], nan_policy="omit")[0]
                    for r in prof])
    rho = rho[np.isfinite(rho)]
    if rho.size:
        print(f"   {'monotonicity rho V1v..hV4':<34} {rho.mean():+.4f}"
              f"        {int((rho > 0).sum())}/{rho.size} positive")


for model in models:
    R = load(model)
    if not R:
        continue
    subs = sorted(R)
    print(f"\n{'#'*78}\n# {model}   n={len(subs)}  subjects {subs}\n{'#'*78}")

    for stat in ("norm", "raw"):
        print(f"\n{'='*78}\nSTATISTIC: {stat}\n{'='*78}")

        print("\n1. Stream dissociation (leg (a) cross-check; primary = stream_shape_texture.py)")
        tab = {}
        for band in ("magno", "parvo", "joint"):
            tab[band] = np.array([
                med(quantity(R[s], band, stat), R[s]["noise_ceiling"] > NC_MIN)
                for s in subs])
            report(band, tab[band])
        report("magno - parvo", tab["magno"] - tab["parvo"])

        print("\n2. Cortical profile (higher = more shape-driven)  [the §7.3b contribution]")
        prof = []
        for s in subs:
            z = R[s]
            ok = (z["noise_ceiling"] > NC_MIN) & np.isfinite(quantity(z, "joint", stat))
            prof.append(roi_profile(quantity(z, "joint", stat), ok, z, s))
        show_profile(prof, "raw-profile")

        print("\n3. rho(quantity, SPI) across voxels   [§24.3: joint row partly circular]")
        for band in ("joint", "magno", "parvo"):
            rr = []
            for s in subs:
                z = R[s]
                v, spi = quantity(z, band, stat), z["spi"]
                m = (z["noise_ceiling"] > NC_MIN) & np.isfinite(v) & np.isfinite(spi)
                rr.append(sps.spearmanr(v[m], spi[m])[0] if m.sum() >= MIN_VOX else np.nan)
            note = "   <- band mixing" if band == "joint" else "   <- within-band control"
            report(band, np.array(rr), note)

        for nbin_desc, ncell in (("4. minus SPI (20 quantile bins)", None),
                                 ("5. minus SPI x predictability (10x10)", 10)):
            print(f"\n{nbin_desc}")
            rprof, shares = [], []
            for s in subs:
                z = R[s]
                v, spi = quantity(z, "joint", stat), z["spi"]
                q = z["r2_joint_nc"] if "r2_joint_nc" in z.files else z["r2_joint"]
                m = (z["noise_ceiling"] > NC_MIN) & np.isfinite(v) & np.isfinite(spi) & np.isfinite(q)

                def dig(x, n):
                    e = np.nanquantile(x[m], np.linspace(0, 1, n + 1))
                    e[0], e[-1] = -np.inf, np.inf
                    return np.digitize(x, e) - 1

                resid = np.full(v.shape, np.nan)
                if ncell is None:
                    b = dig(spi, 20)
                    cells = [(m & (b == k)) for k in range(20)]
                else:
                    ba, bb = dig(spi, ncell), dig(q, ncell)
                    cells = [(m & (ba == ka) & (bb == kb))
                             for ka in range(ncell) for kb in range(ncell)]
                for sel in cells:
                    if sel.sum() >= 5:
                        resid[sel] = v[sel] - np.median(v[sel])
                ok = np.isfinite(resid) & m
                shares.append(1.0 - np.nanvar(resid[ok]) / np.nanvar(v[ok]))
                rprof.append(roi_profile(resid, ok, z, s))
            print(f"   variance removed: {np.mean(shares):.3f}")
            show_profile(rprof, "residual")
