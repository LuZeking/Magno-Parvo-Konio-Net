"""Diet vs no-diet on the §7.3b measures, paired within subject.

This is the contrast that decides the framing of §7.3b. "A two-stream network
predicts a texture-dominated hV4/LO" and "the developmental DIET produces a
texture-dominated hV4/LO" are different papers, and only this comparison
separates them.

Paired within subject (both models are fit to the same 8 subjects' voxels), so
the test is a Wilcoxon on per-subject differences -- the same inferential axis
used throughout the project.

Both statistics per §8; a difference is claimed only where both agree.

Usage: PYTHONPATH=. python tmp_analysis/cc_contrast.py [MODEL_A MODEL_B]
"""
from __future__ import annotations
import glob, sys
import numpy as np
from scipy import stats as sps

D = "results/encoding_cc"
NC_MIN, MIN_VOX = 0.1, 20
A = sys.argv[1] if len(sys.argv) > 2 else "dkl_bcs_seed0_trained"
B = sys.argv[2] if len(sys.argv) > 2 else "dkl_base_seed0_trained"


def med(v, sel):
    v = v[sel & np.isfinite(v)]
    return np.median(v) if v.size >= MIN_VOX else np.nan


def qty(z, band, stat):
    es, et = z[f"cc_{band}_eta2_shape"], z[f"cc_{band}_eta2_texture"]
    return z[f"cc_{band}_vsb"] if stat == "norm" else es - et


def resid(z, stat, keys, nb):
    v = qty(z, "joint", stat)
    m = (z["noise_ceiling"] > NC_MIN) & np.isfinite(v)
    codes = []
    for k, n in zip(keys, nb):
        x = z[k]; m = m & np.isfinite(x)
        if n is None:
            codes.append(x.astype(int))
        else:
            e = np.nanquantile(x[m], np.linspace(0, 1, n + 1)); e[0], e[-1] = -np.inf, np.inf
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


def step(v, ok, vroi):
    return (med(v, ok & (vroi == 7))
            - np.nanmean([med(v, ok & (vroi == 1)), med(v, ok & (vroi == 2))]))


def load(m):
    return {int(f.rsplit("subj", 1)[1][:2]): np.load(f)
            for f in sorted(glob.glob(f"{D}/{m}_subj*.npz"))}


ra, rb = load(A), load(B)
subs = sorted(set(ra) & set(rb))
print(f"A = {A}  (n={len(ra)})\nB = {B}  (n={len(rb)})\npaired: {subs}\n")
if not subs:
    sys.exit("no paired subjects yet")


def measures(z, stat):
    rel = z["noise_ceiling"] > NC_MIN
    vroi = z["roi_prf-visualrois"]
    v = qty(z, "joint", stat)
    r1, ok1 = resid(z, stat, ["spi", "r2_joint_nc"], [10, 10])
    r2, ok2 = resid(z, stat, ["spi", "r2_joint_nc", "roi_prf-eccrois"], [10, 10, None])
    return {
        "magno": med(qty(z, "magno", stat), rel),
        "parvo": med(qty(z, "parvo", stat), rel),
        "mp": med(qty(z, "magno", stat), rel) - med(qty(z, "parvo", stat), rel),
        "raw": step(v, rel & np.isfinite(v), vroi),
        "res": step(r1, ok1, vroi),
        "res_ecc": step(r2, ok2, vroi),
        "fit": float(np.nanmedian(z["r2_joint_nc"][rel])),
    }


for stat in ("norm", "raw"):
    print(f"{'='*84}\nSTATISTIC: {stat}\n{'='*84}")
    print(f"  {'measure':<24} {'A':>9} {'B':>9} {'A-B':>9} {'p':>8}  subjects")
    ma = [measures(ra[s], stat) for s in subs]
    mb = [measures(rb[s], stat) for s in subs]
    for k, label in (("fit", "median r2/NC"),
                     ("magno", "magno stream"),
                     ("parvo", "parvo stream"),
                     ("mp", "magno - parvo"),
                     ("raw", "hV4-V1 uncorrected"),
                     ("res", "hV4-V1 - SPIxpred"),
                     ("res_ecc", "hV4-V1 - SPIxpredxecc")):
        a = np.array([m[k] for m in ma]); b = np.array([m[k] for m in mb])
        d = (a - b)[np.isfinite(a - b)]
        p = sps.wilcoxon(d)[1] if d.size >= 5 else np.nan
        ps = "n/a" if np.isnan(p) else f"{p:.4f}"
        print(f"  {label:<24} {np.nanmean(a):+9.4f} {np.nanmean(b):+9.4f} "
              f"{np.nanmean(d):+9.4f} {ps:>8}  {int((d>0).sum())}/{d.size} A>B")
