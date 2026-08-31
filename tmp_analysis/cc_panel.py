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
