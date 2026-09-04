"""Within-seed test of the checkpoint confound on the §2.2 PR statistic.

§68 Test C correlated the magno/parvo PR ratio against val_top1 ACROSS the three
diet seeds (n=3, final checkpoints) and found r = -0.99, and I read that as the
PR instrument tracking checkpoint degradation. That inference confounds seed
with accuracy and rests on three points.

The clean test varies accuracy WITHIN one seed, holding the seed fixed: the
cached epoch trajectories (ep025-150) give six checkpoints per run spanning a
wide accuracy range. If the PR ratio tracks accuracy, it must do so here.

Usage: PYTHONPATH=. python tmp_analysis/pr_epoch_within_seed.py
"""
from __future__ import annotations
import glob, json
import numpy as np
from scipy import stats as sps

CACHE = "/scratch/byrong/VST_data/mpk_feature_cache"
NIMG = 1000
RUNS = ["dkl_bcs_seed0", "dkl_base_seed0"]


def pr(X):
    X = np.asarray(X[:NIMG], dtype=np.float64)
    X -= X.mean(0, keepdims=True)
    lam = np.linalg.eigvalsh(X @ X.T)
    lam = lam[lam > 0]
    return (lam.sum() ** 2) / (lam ** 2).sum()


def ratios(model):
    out = {}
    for f in sorted(glob.glob(f"{CACHE}/{model}_subj*.npz")):
        s = int(f.split("_subj")[-1].split(".")[0])
        z = np.load(f)
        out[s] = pr(z["magno"]) / pr(z["parvo"])
    return out


for run in RUNS:
    hist = json.load(open(f"/scratch/byrong/VST_data/mpk_runs/{run}/history.json"))
    print("=" * 78)
    print(f"{run} -- magno/parvo PR ratio vs checkpoint   (parity = 1.0)")
    print(f"{'ckpt':>6} {'val_top1':>9} {'mean':>8} {'side':>7}  per-subject range")
    pts = []
    for ep in (25, 50, 75, 100, 125, 150):
        r = ratios(f"{run}_ep{ep:03d}_trained")
        if not r:
            continue
        v = np.array([r[k] for k in sorted(r)])
        acc = hist[ep - 1]["val_top1"]
        pts.append((acc, v.mean()))
        side = f"{int((v > 1).sum())}/{len(v)}>1"
        print(f"{ep:>6} {acc:>9.4f} {v.mean():>8.4f} {side:>7}  [{v.min():.3f}, {v.max():.3f}]")
    r = ratios(f"{run}_trained")
    v = np.array([r[k] for k in sorted(r)])
    print(f"{'last':>6} {hist[-1]['val_top1']:>9.4f} {v.mean():>8.4f} "
          f"{int((v>1).sum())}/{len(v)}>1  [{v.min():.3f}, {v.max():.3f}]  <- PUBLISHED")
    if len(pts) >= 3:
        a = np.array([x[0] for x in pts]); b = np.array([x[1] for x in pts])
        rr, pp = sps.pearsonr(a, b)
        print(f"\n  within-seed r(val_top1, PR ratio) = {rr:+.3f}  p={pp:.3f}  n={len(a)}")
        print(f"  accuracy span {a.min():.3f}-{a.max():.3f}; PR-ratio span "
              f"{b.min():.3f}-{b.max():.3f} (range {np.ptp(b):.3f})")
    print()
