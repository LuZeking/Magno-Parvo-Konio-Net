"""Does a model trained on an actual developmental SCHEDULE land where our
fixed-age-48 models land?

The paper's most likely reviewer objection is that nothing in it develops: our
diet is a STATIC front end matched to 48-month acuity, not a trajectory. That
is a deliberate design choice (it isolates the mechanism from the schedule),
but it is only defensible if the schedule does not change the answer.

`thesis_dvd_trans` is the BA thesis checkpoint: trained with a progressive age
transition rather than a fixed age. Different codebase, so per §45.6 this is a
CONVERGENT check across independent implementations, not a matched arm -- which
is exactly what makes agreement informative and disagreement interpretable.

The test is the §2.2 claim: which side of parity does training end on?
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
    lam = np.linalg.eigvalsh(X @ X.T); lam = lam[lam > 0]
    return (lam.sum() ** 2) / (lam ** 2).sum()

_store = dict(np.load(STORE)) if os.path.exists(STORE) else {}

def ratios(model):
    dirty = False
    for f in sorted(glob.glob(f"{CACHE}/{model}_subj*.npz")):
        s = int(f.split("_subj")[-1].split(".")[0]); k = f"{model}|subj{s:02d}"
        if k in _store: continue
        z = np.load(f); _store[k] = np.array([pr(z["magno"]), pr(z["parvo"])])
        dirty = True; print(f"   [pr] {k}", flush=True)
    if dirty:
        np.savez(STORE, **_store)
    return {int(k.partition("|subj")[2]): v[0]/v[1] for k, v in _store.items()
            if k.partition("|subj")[0] == model}

print("=" * 78)
print("SCHEDULE-TRAINED MODEL vs OUR FIXED-AGE-48 MODELS  (magno/parvo PR ratio)")
print("=" * 78)
rows = [("thesis_dvd_trans (SCHEDULE, BA codebase)", "thesis_dvd_trans")]
rows += [(f"diet fixed-age-48 seed{i}", f"dkl_bcs_seed{i}_trained") for i in (0,1,2)]
rows += [(f"no diet seed{i}",           f"dkl_base_seed{i}_trained") for i in (0,1,2)]
for label, m in rows:
    r = ratios(m)
    if not r:
        print(f"  {label:42s}  (no cache)"); continue
    v = np.array([r[s] for s in sorted(r)])
    print(f"  {label:42s} {v.mean():6.3f} +/-{v.std(ddof=1):.3f}   "
          f"{(v>1).sum()}/{v.size} above parity")
