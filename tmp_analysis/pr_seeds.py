"""Seed replication of the magno/parvo PR ratio -- the §2.3 parity claim.

Every number in §2.3 rested on ONE checkpoint per arm. Array 26413641 filled
seeds 1-2 for all four training conditions (56 cache-only tasks, all COMPLETED).
This script asks the three questions that the n=1 table could not answer:

  Q1  Does the SIDE OF PARITY replicate?  Diet arms > 1, no-diet arms < 1,
      in every seed?  This is the actual claim of §2.3 after §50 killed the
      "universal set point" reading.
  Q2  How big is the seed-to-seed spread?  §49.4 wanted to call the two
      diet-on-luminance arms "converged to within 2%" (a gap of 0.037). That
      reading is only legal if seed spread is SMALLER than 0.037.
  Q3  Diet vs no-diet at the seed level: 3 v 3 Mann-Whitney (exact two-sided
      floor p=0.100, reached only on complete separation).

Per §8, per-subject ratios are the raw statistic; the paired subject-level
Wilcoxon (n=8, exact floor p=.0078) is the within-arm test.
"""
from __future__ import annotations
import glob, os, itertools
import numpy as np
from scipy import stats as sps

CACHE = "/scratch/byrong/VST_data/mpk_feature_cache"
STORE = os.path.join(os.environ.get("CLAUDE_JOB_DIR", "/tmp"), "tmp", "pr_store.npz")
NIMG = 1000

def pr(X):
    X = np.asarray(X[:NIMG], dtype=np.float64)
    X -= X.mean(0, keepdims=True)
    lam = np.linalg.eigvalsh(X @ X.T)
    lam = lam[lam > 0]
    return (lam.sum() ** 2) / (lam ** 2).sum()

_store = dict(np.load(STORE)) if os.path.exists(STORE) else {}

def streams(model):
    dirty = False
    for f in sorted(glob.glob(f"{CACHE}/{model}_subj*.npz")):
        subj = int(f.split("_subj")[-1].split(".")[0])
        key = f"{model}|subj{subj:02d}"
        if key in _store:
            continue
        z = np.load(f)
        _store[key] = np.array([pr(z["magno"]), pr(z["parvo"])])
        dirty = True
        print(f"    [pr] {key}", flush=True)
    if dirty:
        os.makedirs(os.path.dirname(STORE), exist_ok=True)
        np.savez(STORE, **_store)
    return {int(k.partition("|subj")[2]): v for k, v in _store.items()
            if k.partition("|subj")[0] == model}

def ratios(model):
    return {s: v[0] / v[1] for s, v in streams(model).items()}

# arm -> (label, front-end reference, diet?, three trained checkpoints)
ARMS = [
    ("diet / normal",      "untrained_seed0",        True,
     [f"dkl_bcs_seed{i}_trained" for i in (0, 1, 2)]),
    ("diet / parvo_highsf","untrained_parvo_highsf", True,
     [f"dkl_bcs_parvo_highsf_seed{i}_trained" for i in (0, 1, 2)]),
    ("diet / swap (rg)",   "untrained_swap",         True,
     [f"dkl_bcs_swap_seed{i}_trained" for i in (0, 1, 2)]),
    ("no diet",            "untrained_seed0_noblur", False,
     [f"dkl_base_seed{i}_trained" for i in (0, 1, 2)]),
]

print("=" * 100)
print("SEED REPLICATION OF THE magno/parvo PR RATIO   (n=8 subjects x 3 seeds, NIMG=1000)")
print("=" * 100)

table = {}   # arm label -> list of (seed, mean, sem, n_above_1, per-subject array)
for label, front, is_diet, models in ARMS:
    fr = ratios(front)
    fv = np.array([fr[s] for s in sorted(fr)])
    print(f"\n{label}   [front end {fv.mean():.3f} +/-{sps.sem(fv):.3f}, "
          f"{(fv>1).sum()}/{len(fv)} above parity]")
    print(f"  {'seed':>4s} {'mean':>8s} {'sem':>7s} {'min':>7s} {'max':>7s} {'>1':>6s}")
    rows = []
    for si, m in enumerate(models):
        r = ratios(m)
        subs = sorted(r)
        v = np.array([r[s] for s in subs])
        if v.size == 0:
            print(f"  {si:>4d}   MISSING")
            continue
        rows.append((si, v))
        print(f"  {si:>4d} {v.mean():8.3f} {sps.sem(v):7.3f} {v.min():7.3f} "
              f"{v.max():7.3f} {(v>1).sum():>3d}/{v.size}")
    table[label] = (is_diet, rows, fv)

print("\n" + "=" * 100)
print("Q1  SIDE OF PARITY -- does every seed of every arm land on the predicted side?")
print("=" * 100)
ok = True
for label, (is_diet, rows, fv) in table.items():
    side = "above" if is_diet else "below"
    for si, v in rows:
        m = v.mean()
        good = (m > 1) if is_diet else (m < 1)
        frac = (v > 1).sum() if is_diet else (v < 1).sum()
        ok &= bool(good)
        print(f"  {label:22s} seed{si}  mean={m:6.3f}  predicted {side} 1.0  "
              f"-> {'OK ' if good else 'FAIL'}  ({frac}/{v.size} subjects on that side)")
verdict = ("the side-of-parity claim replicates in every seed of every arm" if ok
           else "AT LEAST ONE SEED CROSSES PARITY -- the claim does not replicate")
print("\n  VERDICT: " + verdict)

print("\n" + "=" * 100)
print("Q2  SEED SPREAD vs the 0.037 gap that §49.4 wanted to call 'within 2%'")
print("=" * 100)
for label, (is_diet, rows, fv) in table.items():
    if len(rows) < 2:
        continue
    means = np.array([v.mean() for _, v in rows])
    print(f"  {label:22s} seed means {np.array2string(means, precision=3)}  "
          f"range={means.ptp():.3f}  sd={means.std(ddof=1):.3f}")
allm = {l: np.array([v.mean() for _, v in r]) for l, (d, r, f) in table.items() if r}
if "diet / normal" in allm and "diet / parvo_highsf" in allm:
    gap = abs(allm["diet / normal"].mean() - allm["diet / parvo_highsf"].mean())
    spread = max(allm["diet / normal"].ptp(), allm["diet / parvo_highsf"].ptp())
    print(f"\n  gap between the two diet-on-luminance arms : {gap:.3f}")
    print(f"  largest within-arm seed range              : {spread:.3f}")
    print(f"  -> the 'converge to within 2%' reading is "
          f"{'LEGAL' if spread < gap else 'ILLEGAL: seed noise exceeds the gap'}")

print("\n" + "=" * 100)
print("Q3  DIET vs NO DIET at the seed level (3 v 3 Mann-Whitney, exact floor p=0.100)")
print("=" * 100)
nod = allm.get("no diet")
for label in ("diet / normal", "diet / parvo_highsf", "diet / swap (rg)"):
    if label not in allm or nod is None:
        continue
    a = allm[label]
    if a.size < 3 or nod.size < 3:
        print(f"  {label:22s} incomplete"); continue
    u = sps.mannwhitneyu(a, nod, alternative="two-sided", method="exact")
    sep = "complete separation" if (a.min() > nod.max() or a.max() < nod.min()) else "OVERLAP"
    print(f"  {label:22s} vs no diet   {a.mean():6.3f} vs {nod.mean():6.3f}   "
          f"p={u.pvalue:.3f}   {sep}")
