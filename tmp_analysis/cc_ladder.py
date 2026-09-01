"""Capacity ladder for the §7.3b step, paired within subject (PAPER_PLAN §36.4).

The question §34.1 raised: is the Gabor null's flat profile a front-end fact or
a capacity artefact? The ladder answers it by truncating the TRAINED model's
PCA toward the bank's effective dimensionality and asking whether its step
survives. Pairing within subject removes between-subject variance, which is the
dominant term here.

Usage: PYTHONPATH=. python tmp_analysis/cc_ladder.py
"""
from __future__ import annotations
import glob, os, sys
import numpy as np
from scipy import stats as sps

D = "results/encoding_cc"
NC_MIN, MIN_VOX = 0.1, 20
REF = "dkl_bcs_seed0_trained"
RUNGS = ["dkl_bcs_seed0_pca16", "dkl_bcs_seed0_pca64", REF, "gabor_blur_cs"]


def med(v, sel):
    v = v[sel & np.isfinite(v)]
    return np.median(v) if v.size >= MIN_VOX else np.nan


def resid(z, keys, nb):
    v = z["cc_joint_vsb"]
    m = (z["noise_ceiling"] > NC_MIN) & np.isfinite(v)
    codes = []
    for k, n in zip(keys, nb):
        x = z[k]
        m = m & np.isfinite(x)
        if n is None:
            codes.append(x.astype(int))
        else:
            e = np.nanquantile(x[m], np.linspace(0, 1, n + 1))
            e[0], e[-1] = -np.inf, np.inf
            codes.append(np.digitize(x, e) - 1)
    key = np.zeros_like(codes[0])
    for c in codes:
        key = key * (c.max() + 2) + (c - c.min())
    out = np.full(v.shape, np.nan)
    for k in np.unique(key[m]):
        s = m & (key == k)
        if s.sum() >= 5:
            out[s] = v[s] - np.median(v[s])
    return out, np.isfinite(out) & m


def step(vals, ok, vroi):
    return (med(vals, ok & (vroi == 7))
            - np.nanmean([med(vals, ok & (vroi == 1)), med(vals, ok & (vroi == 2))]))


def measures(path):
    z = np.load(path)
    rel = z["noise_ceiling"] > NC_MIN
    vroi = z["roi_prf-visualrois"]
    v = z["cc_joint_vsb"]
    r1, ok1 = resid(z, ["spi", "r2_joint_nc"], [10, 10])
    r2, ok2 = resid(z, ["spi", "r2_joint_nc", "roi_prf-eccrois"], [10, 10, None])
    return {
        "fit": float(np.nanmedian(z["r2_joint_nc"][rel])),
        "mp": med(z["cc_magno_vsb"], rel) - med(z["cc_parvo_vsb"], rel),
        "raw": step(v, rel & np.isfinite(v), vroi),
        "res": step(r1, ok1, vroi),
        "res_ecc": step(r2, ok2, vroi),
    }


rows = {}
for m in RUNGS:
    per = {}
    for f in sorted(glob.glob(f"{D}/{m}_subj*.npz")):
        per[int(f.rsplit("subj", 1)[1][:2])] = measures(f)
    if per:
        rows[m] = per

print(f"{'model':26s} {'n':>3s} {'fit':>7s} {'M-P':>8s} {'hV4-V1':>8s} "
      f"{'-SPIxp':>8s} {'-SPIxpxe':>9s}")
for m, per in rows.items():
    n = len(per)
    agg = {k: np.array([per[s][k] for s in sorted(per)]) for k in
           ("fit", "mp", "raw", "res", "res_ecc")}
    print(f"{m:26s} {n:3d} " + " ".join(
        f"{np.nanmean(agg[k]):8.4f}" for k in ("fit", "mp", "raw", "res", "res_ecc")))

# paired: each rung against the full-dimensional reference, same subjects
print(f"\npaired vs {REF} (same subjects only)")
for m in RUNGS:
    if m == REF or m not in rows:
        continue
    shared = sorted(set(rows[m]) & set(rows[REF]))
    if not shared:
        continue
    print(f"\n  {m}  (n={len(shared)}, subjects {shared})")
    for k, label in (("raw", "hV4-V1 uncorrected"),
                     ("res", "hV4-V1 - SPIxpred"),
                     ("res_ecc", "hV4-V1 - SPIxpredxecc")):
        a = np.array([rows[m][s][k] for s in shared])
        b = np.array([rows[REF][s][k] for s in shared])
        d = (a - b)[np.isfinite(a - b)]
        keep = float(np.nanmean(a) / np.nanmean(b)) if np.nanmean(b) else np.nan
        p = sps.wilcoxon(d)[1] if d.size >= 5 else np.nan
        ps = "n/a" if np.isnan(p) else f"{p:.4f}"
        print(f"    {label:24s} rung {np.nanmean(a):+.4f}  ref {np.nanmean(b):+.4f}"
              f"  retained {keep:5.1%}  diff p={ps}")
