"""Does the ECCENTRICITY effect actually DIFFER between arms? (§98.4 corrected)

§98 claimed "the eccentricity component is absent in the no-diet arm under
every convention -- an arm difference on the retinotopic claim". The evidence
offered was diet significant (8/8, p=.0078) and no-diet not significant. That
is the difference-between-significant-and-not fallacy (Gelman & Stern 2006): it
is not a test of the difference, and §8 (line 346) already refuses exactly this
move elsewhere in the paper.

The test the claim needs is paired and within subject: compute the ecc|size
partial per subject per arm, difference them within subject, and test that
across the 8 subjects with the same estimator and the same resampling unit.

Also recorded here because §98 did not say it: at n = 8 the two-sided Wilcoxon
signed-rank FLOORS at p = 2/2^8 = .0078125. Every ".0078" in §98 is therefore
"8/8 sign consistency", not an independently strong test, and the interaction
below cannot be read as a further confirmation on top of them.
"""
import glob
import os
import sys

import numpy as np
from scipy import stats as sps

sys.path.insert(0, "/home/byrong/projects/TSN/Magno-Parvo-Konio-Net/"
                   ".claude/worktrees/mpk-dev")

from src.analysis.encoding import nsd_data  # noqa: E402

R = ("/home/byrong/projects/TSN/Magno-Parvo-Konio-Net/.claude/worktrees/"
     "mpk-dev/results/encoding_cc")
FLOOR = 2 / 2 ** 8


def z(v):
    return (v - v.mean()) / v.std()


_prf_cache = {}


def prf(s):
    if s not in _prf_cache:
        mask = nsd_data.load_mask(s, "nsdgeneral")
        _prf_cache[s] = (mask,
                         nsd_data.load_prf(s, "eccentricity", mask),
                         nsd_data.load_prf(s, "size", mask),
                         nsd_data.load_prf(s, "R2", mask))
    return _prf_cache[s]


def partial_by_subject(model, r2_min=5.0, use_log=True, stat="norm"):
    """{subject: (beta_ecc_given_size, beta_size_given_ecc)}"""
    out = {}
    for f in sorted(glob.glob(f"{R}/{model}_subj*.npz")):
        s = int(os.path.basename(f).split("subj")[-1].split(".")[0])
        d = np.load(f, allow_pickle=True)
        um, up, nc = d["unique_magno"], d["unique_parvo"], d["noise_ceiling"]
        _, ecc, size, r2 = prf(s)
        ok = ((nc > 0.1) & ((um + up) > 1e-3) & (r2 > r2_min)
              & np.isfinite(ecc) & np.isfinite(size) & (size > 0))
        if ok.sum() < 200:
            continue
        e, sz = ecc[ok], size[ok]
        le = z(np.log(e + 0.5)) if use_log else z(e)
        ls = z(np.log(sz)) if use_log else z(sz)
        y = ((um[ok] - up[ok]) / (um[ok] + up[ok])) if stat == "norm" \
            else (um[ok] - up[ok])
        A = np.column_stack([np.ones(int(ok.sum())), le, ls])
        coef, *_ = np.linalg.lstsq(A, z(y), rcond=None)
        out[s] = (coef[1], coef[2])
    return out


def interaction(diet, base, idx, **kw):
    """Paired diet - no-diet across subjects, for term idx (0=ecc, 1=size)."""
    a, b = partial_by_subject(diet, **kw), partial_by_subject(base, **kw)
    subs = sorted(set(a) & set(b))
    d = np.array([a[s][idx] - b[s][idx] for s in subs])
    return d, sps.wilcoxon(d).pvalue if len(d) else np.nan


print(f"n=8 two-sided Wilcoxon floor = {FLOOR:.7f}; a p at the floor means "
      f"8/8 sign\nconsistency, not an independently strong test.\n")

print("=" * 74)
print("1. INTERACTION on ecc|size, seed-matched pairs, statistic = norm")
print("=" * 74)
print(f"{'pairing':22s} {'mean diff':>10s} {'range':>18s} {'pos':>5s} {'p':>8s}")
for sd in (0, 1, 2):
    d, p = interaction(f"dkl_bcs_seed{sd}_trained",
                       f"dkl_base_seed{sd}_trained", 0)
    print(f"{'seed' + str(sd) + ' vs seed' + str(sd):22s} {d.mean():+10.3f} "
          f"{f'{d.min():+.3f}..{d.max():+.3f}':>18s} "
          f"{int((d > 0).sum()):3d}/{len(d)} {p:8.4f}")

# seed-averaged per subject: removes seed noise before differencing
print("\nseed-averaged within arm (the cleaner pairing):")
for stat in ("norm", "raw"):
    A = [partial_by_subject(f"dkl_bcs_seed{s}_trained", stat=stat)
         for s in (0, 1, 2)]
    B = [partial_by_subject(f"dkl_base_seed{s}_trained", stat=stat)
         for s in (0, 1, 2)]
    subs = sorted(set(A[0]) & set(B[0]))
    for idx, term in ((0, "ecc|size"), (1, "size|ecc")):
        d = np.array([np.mean([a[s][idx] for a in A])
                      - np.mean([b[s][idx] for b in B]) for s in subs])
        p = sps.wilcoxon(d).pvalue
        print(f"  {stat:5s} {term:9s} {d.mean():+10.3f} "
              f"{f'{d.min():+.3f}..{d.max():+.3f}':>18s} "
              f"{int((d > 0).sum()):3d}/{len(d)} {p:8.4f}")

print("\n" + "=" * 74)
print("2. IS THE INTERACTION ROBUST to the same two arbitrary choices?")
print("=" * 74)
print(f"{'R2>':>4s} {'pred':>7s} {'stat':>5s} {'ecc interaction':>26s}")
for use_log in (True, False):
    for r2_min in (0.0, 5.0, 10.0, 20.0):
        for stat in ("norm", "raw"):
            d, p = interaction("dkl_bcs_seed0_trained",
                               "dkl_base_seed0_trained", 0,
                               r2_min=r2_min, use_log=use_log, stat=stat)
            print(f"{r2_min:4.0f} {'log' if use_log else 'linear':>7s} "
                  f"{stat:>5s} "
                  f"{f'{d.mean():+.3f} {int((d>0).sum())}/8 p={p:.4f}':>26s}")

print("\nThe arm difference is claimable only if the interaction holds sign")
print("and significance across these cells -- the same bar §98.3 set for the")
print("within-arm effects, applied to the between-arm one.")
