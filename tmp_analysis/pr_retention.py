"""Does training PRESERVE or DESTROY the front end's magno/parvo PR asymmetry?

PAPER_PLAN §41.3 claimed "the front end sets the asymmetry, no-diet training
destroys it", but referenced every trained model to the *diet* front end
(`untrained_seed0`, blur+CS). That is the wrong reference for the no-diet
model, which is trained through a different front end. Pairing each trained
model with its OWN front end asks the causal question directly.

Two statistics, per the §8 rule -- a claim is made only where both agree:
  norm  retention = ratio_trained / ratio_frontend      (>1 amplified)
  raw   retention = ratio_trained - ratio_frontend      (>0 amplified)

Per-stream PRs are cached to an npz so the eigendecompositions run once.
"""
from __future__ import annotations
import glob, os
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

def ratios(model):
    """magno/parvo PR per subject, keyed by subject number."""
    hit = f"{model}|subj"
    if not any(k.startswith(hit) for k in _store):
        for f in sorted(glob.glob(f"{CACHE}/{model}_subj*.npz")):
            subj = int(f.split("_subj")[-1].split(".")[0])
            z = np.load(f)
            _store[f"{model}|subj{subj:02d}"] = np.array([pr(z["magno"]), pr(z["parvo"])])
        os.makedirs(os.path.dirname(STORE), exist_ok=True)
        np.savez(STORE, **_store)
    out = {}
    for k, v in _store.items():
        m, _, s = k.partition("|subj")
        if m == model:
            out[int(s)] = v[0] / v[1]
    return out

PAIRS = [
    ("dkl_bcs_seed0_trained",  "untrained_seed0",        "diet"),
    ("thesis_dvd_trans",       "untrained_seed0",        "diet"),
    ("dkl_base_seed0_trained", "untrained_seed0_noblur", "no diet"),
    ("dkl_base_seed1_trained", "untrained_seed0_noblur", "no diet"),
]
GABOR = {"diet": "gabor_blur_cs", "no diet": "gabor_noblur"}

def retention(trained, front):
    t, f = ratios(trained), ratios(front)
    subs = sorted(set(t) & set(f))
    norm = np.array([t[s] / f[s] for s in subs])
    raw = np.array([t[s] - f[s] for s in subs])
    return norm, raw, len(subs)

print("RETENTION vs each model's OWN front end (untrained-network reference)")
print(f"{'trained model':24s} {'diet?':8s} {'n':>2s} {'norm':>16s} {'>1':>5s} {'raw':>16s} {'>0':>5s}")
res = {}
for trained, front, label in PAIRS:
    n_, r_, n = retention(trained, front)
    res[trained] = (n_, r_, label)
    print(f"{trained:24s} {label:8s} {n:2d} {n_.mean():8.2f} +-{n_.std():4.2f} "
          f"{int((n_>1).sum()):2d}/{n} {r_.mean():8.2f} +-{r_.std():4.2f} {int((r_>0).sum()):2d}/{n}")

print("\nRETENTION vs the learning-free input-level bank (gabor reference)")
print(f"{'trained model':24s} {'gabor front':16s} {'norm':>16s} {'>1':>5s} {'raw':>16s} {'>0':>5s}")
resg = {}
for trained, _f, label in PAIRS:
    n_, r_, n = retention(trained, GABOR[label])
    resg[trained] = (n_, r_, label)
    print(f"{trained:24s} {GABOR[label]:16s} {n_.mean():8.2f} +-{n_.std():4.2f} "
          f"{int((n_>1).sum()):2d}/{n} {r_.mean():8.2f} +-{r_.std():4.2f} {int((r_>0).sum()):2d}/{n}")

def wil(a, b):
    n = min(len(a), len(b)); a, b = a[:n], b[:n]
    pos = int((a > b).sum())
    if n < 5:
        return f"{a.mean():+.2f} vs {b.mean():+.2f}  {pos}/{n} (n<5, no p)"
    return (f"{a.mean():+.2f} vs {b.mean():+.2f}  p={sps.wilcoxon(a-b)[1]:.4f}  {pos}/{n}")

for tag, R in (("own front end", res), ("gabor front end", resg)):
    print(f"\n-- diet vs no-diet retention, paired by subject [{tag}] --")
    print(f"{'contrast':50s} {'NORM':>34s} {'RAW':>34s}")
    for d in ("dkl_bcs_seed0_trained", "thesis_dvd_trans"):
        for nd in ("dkl_base_seed0_trained", "dkl_base_seed1_trained"):
            print(f"{d + ' vs ' + nd:50s} {wil(R[d][0], R[nd][0]):>34s} "
                  f"{wil(R[d][1], R[nd][1]):>34s}")

print("\n-- front-end asymmetry itself (no training involved) --")
for m in ("untrained_seed0", "untrained_seed0_noblur", "gabor_blur_cs", "gabor_noblur"):
    r = np.array(list(ratios(m).values()))
    print(f"  {m:24s} magno/parvo = {r.mean():.2f} +-{r.std():.2f}  (n={len(r)})")
