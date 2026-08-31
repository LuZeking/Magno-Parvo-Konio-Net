"""PAPER_PLAN §7.3b/§24: group analysis of the in-silico cue-conflict panel.

Three questions, in order of how much the paper can lean on them:

  1. LEG (a): do the two streams' predicted cue-conflict responses dissociate
     into shape- vs texture-driven? Per-band median VSB, per subject.
  2. THE NON-CIRCULAR TEST: does the joint model's predicted shape weighting
     rise along V1 -> hV4/LO, as the human literature says shape selectivity
     does? This is checkable against outside knowledge and can fail.
  3. VSB vs SPI across voxels -- reported WITH the §24.3 caveat, since the
     within-band rows show the correlation is largely carried by band mixing.

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
# 14 = LO2, 15 = LO1 (the pair is taken as a union, so the order is cosmetic).
# The same table confirms VO1/VO2 = 8/9 and TO2/TO1 = 12/13 as used in
# tmp_analysis/model_verdict.py, and V1v..hV4 = 1..7 as used in VIS above.
KAST_LO = {14: "LO2", 15: "LO1"}
NC_MIN, MIN_VOX = 0.1, 20

models = sys.argv[1:] or sorted({
    os.path.basename(f).rsplit("_subj", 1)[0] for f in glob.glob(f"{DIR}/*.npz")
})


def load(model):
    out = {}
    for f in sorted(glob.glob(f"{DIR}/{model}_subj*.npz")):
        s = int(f.rsplit("subj", 1)[1][:2])
        out[s] = np.load(f)
    return out


def med(v, sel):
    v = v[sel & np.isfinite(v)]
    return np.median(v) if v.size >= MIN_VOX else np.nan


for model in models:
    R = load(model)
    if not R:
        continue
    subs = sorted(R)
    print(f"\n{'='*78}\n{model}   n={len(subs)} subjects {subs}\n{'='*78}")

    # ---- 1. leg (a): per-band VSB ----
    print("\n1. Stream dissociation (median VSB over reliable voxels)")
    tab = {}
    for band in ("magno", "parvo", "joint"):
        vals = []
        for s in subs:
            z = R[s]
            rel = z["noise_ceiling"] > NC_MIN
            vals.append(med(z[f"cc_{band}_vsb"], rel))
        tab[band] = np.array(vals)
        v = tab[band][np.isfinite(tab[band])]
        p = sps.wilcoxon(v)[1] if v.size >= 5 else np.nan
        print(f"   {band:6s} {np.mean(v):+.4f}  (p={p:.4f}, {int((v>0).sum())}/{v.size} positive)"
              f"   per-subject: {' '.join(f'{x:+.2f}' for x in tab[band])}")
    d = tab["magno"] - tab["parvo"]
    d = d[np.isfinite(d)]
    if d.size >= 5:
        print(f"   magno - parvo  {d.mean():+.4f}  p={sps.wilcoxon(d)[1]:.4f}  "
              f"{int((d>0).sum())}/{d.size} subjects")

    # ---- 2. non-circular: VSB along the visual hierarchy ----
    print("\n2. Joint-model VSB along the hierarchy (higher = more shape-driven)")
    prof = []
    for s in subs:
        z = R[s]
        rel = z["noise_ceiling"] > NC_MIN
        vroi = z["roi_prf-visualrois"]
        row = [med(z["cc_joint_vsb"], rel & (vroi == k)) for k in sorted(VIS)]
        try:
            kast = nsd_data.load_roi_labels(s, "Kastner2015",
                                            nsd_data.load_mask(s, "nsdgeneral"))
            row.append(med(z["cc_joint_vsb"], rel & np.isin(kast, list(KAST_LO))))
        except Exception:
            row.append(np.nan)
        prof.append(row)
    prof = np.array(prof)
    names = [VIS[k] for k in sorted(VIS)] + ["LO"]
    print("   " + "  ".join(f"{n:>7s}" for n in names))
    print("   " + "  ".join(f"{x:+7.3f}" for x in np.nanmean(prof, 0)))
    # monotonic rise V1 -> hV4 ?
    early = np.nanmean(prof[:, [0, 1]], 1)      # V1v, V1d
    late = prof[:, 6]                            # hV4
    dd = (late - early)[np.isfinite(late - early)]
    if dd.size >= 5:
        print(f"   hV4 - V1: {dd.mean():+.4f}  p={sps.wilcoxon(dd)[1]:.4f}  "
              f"{int((dd>0).sum())}/{dd.size} subjects "
              f"[predicted POSITIVE if the model reproduces cortical organisation]")
    rho = [sps.spearmanr(np.arange(7), r[:7], nan_policy="omit")[0] for r in prof]
    rho = np.array(rho)[np.isfinite(rho)]
    if rho.size:
        print(f"   monotonicity rho over V1v..hV4: mean {rho.mean():+.3f} "
              f"({int((rho>0).sum())}/{rho.size} positive)")

    # ---- 3. VSB vs SPI, with the caveat ----
    print("\n3. rho(VSB, SPI) across voxels   [§24.3: joint row is partly circular]")
    for band in ("joint", "magno", "parvo"):
        rr = []
        for s in subs:
            z = R[s]
            rel = z["noise_ceiling"] > NC_MIN
            v, spi = z[f"cc_{band}_vsb"], z["spi"]
            m = rel & np.isfinite(v) & np.isfinite(spi)
            rr.append(sps.spearmanr(v[m], spi[m])[0] if m.sum() >= MIN_VOX else np.nan)
        rr = np.array(rr); f = rr[np.isfinite(rr)]
        p = sps.wilcoxon(f)[1] if f.size >= 5 else np.nan
        note = "  <- band mixing, see §24.3" if band == "joint" else "  <- within-band control"
        print(f"   {band:6s} mean rho {f.mean():+.4f}  p={p:.4f}  "
              f"{int((f>0).sum())}/{f.size}{note}")

    # ---- 4. SPI-residualised profile: the non-circular remainder ----
    # §24.5 established that the raw VSB profile re-expresses the band-mixing
    # gradient. Two voxels with the SAME SPI can still differ in VSB, though,
    # depending on WHICH features inside a band they read out -- and that
    # remainder is not determined by the variance partition. Removing all
    # dependence on SPI at the resolution of 20 quantile bins (stronger than
    # linear residualisation: it kills any monotone relation) leaves exactly
    # that remainder. If its ROI profile is flat, §7.3b is fully a
    # re-description of leg (c); if it is not, the model predicts shape/texture
    # organisation beyond band weighting.
    print("\n4. SPI-residualised VSB profile  [the non-circular remainder]")
    NBIN = 20
    rprof, shares = [], []
    for s in subs:
        z = R[s]
        rel = z["noise_ceiling"] > NC_MIN
        v, spi = z["cc_joint_vsb"].copy(), z["spi"]
        m = rel & np.isfinite(v) & np.isfinite(spi)
        resid = np.full(v.shape, np.nan)
        edges = np.nanquantile(spi[m], np.linspace(0, 1, NBIN + 1))
        edges[0], edges[-1] = -np.inf, np.inf
        b = np.digitize(spi, edges) - 1
        for k in range(NBIN):
            sel = m & (b == k)
            if sel.sum() >= 5:
                resid[sel] = v[sel] - np.median(v[sel])
        ok = np.isfinite(resid) & m
        shares.append(1.0 - np.nanvar(resid[ok]) / np.nanvar(v[ok]))
        vroi = z["roi_prf-visualrois"]
        row = [med(resid, ok & (vroi == k)) for k in sorted(VIS)]
        try:
            kast = nsd_data.load_roi_labels(s, "Kastner2015",
                                            nsd_data.load_mask(s, "nsdgeneral"))
            row.append(med(resid, ok & np.isin(kast, list(KAST_LO))))
        except Exception:
            row.append(np.nan)
        rprof.append(row)
    rprof = np.array(rprof)
    print(f"   variance in VSB explained by SPI alone: "
          f"{np.mean(shares):.3f} (per subject {' '.join(f'{x:.2f}' for x in shares)})")
    print("   " + "  ".join(f"{n:>7s}" for n in names))
    print("   " + "  ".join(f"{x:+7.3f}" for x in np.nanmean(rprof, 0)))
    dd = (rprof[:, 6] - np.nanmean(rprof[:, [0, 1]], 1))
    dd = dd[np.isfinite(dd)]
    if dd.size >= 5:
        print(f"   residual hV4 - V1: {dd.mean():+.4f}  p={sps.wilcoxon(dd)[1]:.4f}  "
              f"{int((dd>0).sum())}/{dd.size} subjects")
    else:
        print(f"   residual hV4 - V1: {dd.mean():+.4f}  (n={dd.size}, need 5 for Wilcoxon)")

    # ---- 5. stricter control: residualise on SPI *and* predictability ----
    # SPI is only a summary of the band weighting, and a voxel the model
    # predicts poorly has a noisier VSB. Subtracting the median within each
    # cell of a 10x10 grid of (SPI decile) x (r2_joint_nc decile) removes both
    # at once, so anything left cannot be band preference or fit quality.
    print("\n5. Residualised on SPI x predictability (10x10 cells)")
    r2prof = []
    for s in subs:
        z = R[s]
        rel = z["noise_ceiling"] > NC_MIN
        v, spi = z["cc_joint_vsb"].copy(), z["spi"]
        q = z["r2_joint_nc"] if "r2_joint_nc" in z.files else z["r2_joint"]
        m = rel & np.isfinite(v) & np.isfinite(spi) & np.isfinite(q)
        def dig(x, n=10):
            e = np.nanquantile(x[m], np.linspace(0, 1, n + 1))
            e[0], e[-1] = -np.inf, np.inf
            return np.digitize(x, e) - 1
        ba, bb = dig(spi), dig(q)
        resid = np.full(v.shape, np.nan)
        for ka in range(10):
            for kb in range(10):
                sel = m & (ba == ka) & (bb == kb)
                if sel.sum() >= 5:
                    resid[sel] = v[sel] - np.median(v[sel])
        ok = np.isfinite(resid) & m
        vroi = z["roi_prf-visualrois"]
        row = [med(resid, ok & (vroi == k)) for k in sorted(VIS)]
        try:
            kast = nsd_data.load_roi_labels(s, "Kastner2015",
                                            nsd_data.load_mask(s, "nsdgeneral"))
            row.append(med(resid, ok & np.isin(kast, list(KAST_LO))))
        except Exception:
            row.append(np.nan)
        r2prof.append(row)
    r2prof = np.array(r2prof)
    print("   " + "  ".join(f"{n:>7s}" for n in names))
    print("   " + "  ".join(f"{x:+7.3f}" for x in np.nanmean(r2prof, 0)))
    dd2 = (r2prof[:, 6] - np.nanmean(r2prof[:, [0, 1]], 1))
    dd2 = dd2[np.isfinite(dd2)]
    if dd2.size >= 5:
        print(f"   residual hV4 - V1: {dd2.mean():+.4f}  p={sps.wilcoxon(dd2)[1]:.4f}  "
              f"{int((dd2>0).sum())}/{dd2.size} subjects")
    else:
        print(f"   residual hV4 - V1: {dd2.mean():+.4f}  (n={dd2.size}, need 5 for Wilcoxon)")
