"""Participation ratio per stream, all subjects, all cached models.

PAPER_PLAN §41.2: the diet appears to raise magno's effective dimensionality
above parvo's, while the no-diet model leaves the two comparable. §41.2 measured
subject 1 only; this settles it across subjects.

PR = (sum lambda)^2 / sum(lambda^2) over the feature covariance eigenvalues,
computed from the Gram side since n_images << n_features.
"""
from __future__ import annotations
import glob, os, re, sys
import numpy as np
from scipy import stats as sps

CACHE = "/scratch/byrong/VST_data/mpk_feature_cache"
NIMG = 1000   # PR is stable well below this; keeps 32 decompositions tractable

def pr(X):
    X = np.asarray(X[:NIMG], dtype=np.float64)
    X -= X.mean(0, keepdims=True)
    G = X @ X.T                       # (NIMG, NIMG) -- cheaper than the feature side
    lam = np.linalg.eigvalsh(G)
    lam = lam[lam > 0]
    return (lam.sum() ** 2) / (lam ** 2).sum()

models = sys.argv[1:] or ["dkl_bcs_seed0_trained", "dkl_base_seed0_trained",
                          "thesis_dvd_trans", "gabor_blur_cs", "untrained_seed0"]
print(f"{'model':26s} {'n':>2s} {'magno PR':>18s} {'parvo PR':>18s} {'magno/parvo':>18s}")
rows = {}
for m in models:
    per = []
    for f in sorted(glob.glob(f"{CACHE}/{m}_subj*.npz")):
        z = np.load(f)
        per.append((pr(z["magno"]), pr(z["parvo"])))
    if not per:
        print(f"{m:26s} (no cache)"); continue
    a = np.array([p[0] for p in per]); b = np.array([p[1] for p in per])
    r = a / b
    rows[m] = r
    print(f"{m:26s} {len(per):2d} {a.mean():8.1f} +- {a.std():5.1f} "
          f"{b.mean():8.1f} +- {b.std():5.1f} {r.mean():9.2f} +- {r.std():5.2f}")

if "dkl_bcs_seed0_trained" in rows and "dkl_base_seed0_trained" in rows:
    a, b = rows["dkl_bcs_seed0_trained"], rows["dkl_base_seed0_trained"]
    n = min(len(a), len(b))
    if n >= 5:
        p = sps.wilcoxon(a[:n] - b[:n])[1]
        print(f"\ndiet vs no-diet magno/parvo ratio: {a[:n].mean():.2f} vs "
              f"{b[:n].mean():.2f}, paired p={p:.4f}, {int((a[:n]>b[:n]).sum())}/{n}")
    else:
        print(f"\ndiet vs no-diet ratio: {a.mean():.2f} vs {b.mean():.2f} (n={n}, no test)")
