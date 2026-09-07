"""§102.3's reversals, redone with the interval the DECLARED test implies.

`tost_paired` -> `_tost_from` builds a paired-t interval:

    se = d.std(ddof=1)/sqrt(n);  lo, hi = diff -+ t(.95, n-1)*se

but PAPER_DRAFT.md:1351 declares "the paired exact Wilcoxon signed-rank test,
n = 8", and every p-value in this thread is Wilcoxon (the discrete
.0078/.0156/.1094 ladder). Bounding a null with a t-TOST interval is this
paper's existing practice (§2.6.1, §2.6.2) and is fine -- how big an effect the
data could hide is a different question from whether the declared test
rejected. What that interval cannot do is REVERSE the declared verdict, which
is exactly what §102.3 used it for.

Hodges-Lehmann with the Walsh-average distribution-free CI is the interval
Wilcoxon implies: exact at n=8 and consistent with the p-values by
construction. Implementation is verified against scipy below -- the CI must
exclude zero precisely when the two-sided Wilcoxon p < alpha, and that
equivalence is asserted rather than assumed.
"""
import glob
import itertools
import os
import sys

import numpy as np
from scipy import stats as sps

sys.path.insert(0, "/home/byrong/projects/TSN/Magno-Parvo-Konio-Net/"
                   ".claude/worktrees/mpk-dev")

from src.analysis.encoding import nsd_data  # noqa: E402

SRC = ("/home/byrong/projects/TSN/Magno-Parvo-Konio-Net/.claude/worktrees/"
       "mpk-dev/results/encoding")
_c = {}


def signed_rank_null(n):
    """Exact null distribution of W+ by enumeration (n=8 -> 256 patterns)."""
    ranks = np.arange(1, n + 1)
    vals = [sum(r for r, s in zip(ranks, signs) if s)
            for signs in itertools.product([0, 1], repeat=n)]
    return np.array(sorted(vals))


def hl_ci(d, alpha):
    """Hodges-Lehmann estimate + exact distribution-free CI from Walsh averages."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    n = d.size
    walsh = np.sort([(d[i] + d[j]) / 2
                     for i in range(n) for j in range(i, n)])
    m = walsh.size                                   # n(n+1)/2
    null = signed_rank_null(n)
    # k = largest integer with P(W+ <= k) <= alpha/2; the (1-alpha) interval is
    # then [W_(k+1), W_(M-k)] in 1-indexed Walsh order statistics, i.e.
    # walsh[k] .. walsh[m-1-k] zero-indexed. Taking walsh[k+1] instead -- the
    # off-by-one this file was first written with -- narrows the interval and
    # makes it exclude zero where the declared test does not reject. The
    # assertion block below is what caught that.
    k = -1
    for cand in range(m + 1):
        if (null <= cand).mean() <= alpha / 2:
            k = cand
        else:
            break
    lo = walsh[k] if 0 <= k < m else walsh[0]
    hi = walsh[m - 1 - k] if 0 <= k < m else walsh[-1]
    # Discreteness means the nominal level is not attained: at n=8 with
    # alpha=.10 the largest admissible k is 5, P(W+ <= 5) = 10/256 = .0391,
    # so ACTUAL coverage is 1 - 2(.0391) = 92.19%, not 90%. The construction
    # rounds conservative -- the interval is WIDER than its nominal label -- so
    # reporting it as "90%" understates it. Return the attained coverage and
    # quote that, rather than a label the interval does not have. (The draft
    # names the .0078 floor explicitly rather than rounding it; a nominal
    # label here would be out of character in the same document.)
    cover = 1.0 - 2.0 * float((null <= k).mean()) if 0 <= k < m else 1.0
    return float(np.median(walsh)), float(lo), float(hi), cover


def z(v):
    return (v - v.mean()) / v.std()


def prf(s):
    if s not in _c:
        mk = nsd_data.load_mask(s, "nsdgeneral")
        _c[s] = (nsd_data.load_prf(s, "eccentricity", mk),
                 nsd_data.load_prf(s, "size", mk),
                 nsd_data.load_prf(s, "R2", mk))
    return _c[s]


def betas(model, stat, r2_min=5.0):
    out = {}
    for f in sorted(glob.glob(f"{SRC}/{model}_subj*.npz")):
        s = int(os.path.basename(f).split("subj")[-1].split(".")[0])
        d = np.load(f, allow_pickle=True)
        um, up, nc = d["unique_magno"], d["unique_parvo"], d["noise_ceiling"]
        ecc, size, r2 = prf(s)
        ok = ((nc > 0.1) & ((um + up) > 1e-3) & (r2 > r2_min)
              & np.isfinite(ecc) & np.isfinite(size) & (size > 0))
        if ok.sum() < 200:
            continue
        le, ls = z(np.log(ecc[ok] + 0.5)), z(np.log(size[ok]))
        y = z(((um[ok] - up[ok]) / (um[ok] + up[ok])) if stat == "norm"
              else (um[ok] - up[ok]))
        n = int(ok.sum())
        m2, *_ = np.linalg.lstsq(np.column_stack([np.ones(n), le, ls]), y,
                                 rcond=None)
        out[s] = m2[1]
    return out


def paired(a, b, stat):
    A, B = betas(a, stat), betas(b, stat)
    subs = sorted(set(A) & set(B))
    return np.array([A[s] - B[s] for s in subs])


# --- verify the implementation against the declared test before using it ----
rng = np.random.default_rng(0)
for _ in range(300):
    d = rng.normal(rng.uniform(-1, 1), 1, 8)
    for alpha in (0.05, 0.10):
        _, lo, hi, _cov = hl_ci(d, alpha)
        p = sps.wilcoxon(d).pvalue
        assert (lo > 0 or hi < 0) == (p < alpha), (
            f"HL CI disagrees with Wilcoxon at alpha={alpha}: "
            f"CI=({lo:.4f},{hi:.4f}) p={p:.4f}"
        )
print("implementation verified: HL CI excludes zero iff Wilcoxon p < alpha, "
      "300 random samples, alpha in {.05, .10}\n")

NULLS = [
    ("N1 diet: trained - own init",
     "dkl_bcs_seed0_trained", "untrained_seed0"),
    ("N2 trained no-diet - input null (noblur)",
     "dkl_base_seed0_trained", "gabor_noblur"),
]
print(f"{'null':40s} {'stat':5s} {'HL':>8s} {'exact CI':>22s} "
      f"{'cover':>7s} {'Wilcoxon p':>11s} {'excl 0':>7s}")
for label, a, b in NULLS:
    for stat in ("raw", "norm"):
        d = paired(a, b, stat)
        hl, lo, hi, cover = hl_ci(d, 0.10)
        p = sps.wilcoxon(d).pvalue
        print(f"{label:40s} {stat:5s} {hl:+8.3f} "
              f"{f'[{lo:+.3f}, {hi:+.3f}]':>22s} {cover:7.2%} {p:11.4f} "
              f"{'YES' if (lo > 0 or hi < 0) else 'no':>7s}")

print("\nIf these intervals contain zero, §102.3's two reversals were artefacts")
print("of the t-based interval and the nulls return to UNBOUNDED under the")
print("declared test -- neither asserted nor overturned.")
