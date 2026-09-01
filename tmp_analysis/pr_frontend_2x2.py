"""The front-end 2x2: does the magno/parvo PR asymmetry follow the CARRIER or
the developmental TRANSFORM?

PAPER_PLAN §45.5 pre-registered, before job 26406434 was read:

    parvo_highsf < blur_cs < noblur < swap

from one principle -- PR tracks the bandwidth of the carrier a stream receives,
and the transform removes bandwidth from whichever stream it lands on.

Note on the §8 two-statistics rule: it does NOT add an independent check here.
§8 exists because Um-Up can be normalised by Um+Up, giving two genuinely
different statistics. Here each subject contributes a single scalar (the ratio
magno/parvo), so "a > b" and "a/b > 1" have identical signs by construction.
Reporting both would be theatre. What IS reported instead is each stream's PR
separately, which is the real discriminating test: the carrier account predicts
WHICH stream moves in each cell, not just which way the ratio goes.
"""
from __future__ import annotations
import glob, itertools, os
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
    """(magno PR, parvo PR) per subject."""
    # Check the store PER SUBJECT, not per model. Keying on "model is present"
    # would freeze a cell at however many subjects happened to exist the first
    # time it was read -- the partial-cell trap of §41.2/§39.3, in code.
    dirty = False
    for f in sorted(glob.glob(f"{CACHE}/{model}_subj*.npz")):
        subj = int(f.split("_subj")[-1].split(".")[0])
        key = f"{model}|subj{subj:02d}"
        if key in _store:
            continue
        z = np.load(f)
        _store[key] = np.array([pr(z["magno"]), pr(z["parvo"])])
        print(f"  [pr] {model} subj{subj:02d}", flush=True)
        dirty = True
    if dirty:
        os.makedirs(os.path.dirname(STORE), exist_ok=True)
        np.savez(STORE, **_store)
    out = {}
    for k, v in _store.items():
        m, _, s = k.partition("|subj")
        if m == model:
            out[int(s)] = v
    return out

# pre-registered order, lowest predicted ratio first
ORDER = ["parvo_highsf", "blur_cs", "noblur", "swap"]
CELLS = {
    "gabor": {"parvo_highsf": "gabor_parvo_highsf", "blur_cs": "gabor_blur_cs",
              "noblur": "gabor_noblur", "swap": "gabor_swap"},
    "untrained": {"parvo_highsf": "untrained_parvo_highsf", "blur_cs": "untrained_seed0",
                  "noblur": "untrained_seed0_noblur", "swap": "untrained_swap"},
}
WHAT = {
    "parvo_highsf": "magno lum+transform | parvo LUM+rg, no transform",
    "blur_cs":      "magno lum+transform | parvo rg, no transform",
    "noblur":       "magno lum, no transform | parvo rg, no transform",
    "swap":         "magno lum, no transform | parvo rg+TRANSFORM",
}

for ref, cells in CELLS.items():
    print(f"\n{'='*78}\n{ref.upper()} reference\n{'='*78}")
    data = {k: streams(v) for k, v in cells.items()}
    missing = [k for k, v in data.items() if len(v) < 8]
    print(f"{'cell':14s} {'n':>2s} {'magno PR':>16s} {'parvo PR':>16s} {'ratio':>15s}   front end")
    for k in ORDER:
        d = data[k]
        if not d:
            print(f"{k:14s} -- NOT ON DISK --"); continue
        subs = sorted(d)
        m = np.array([d[s][0] for s in subs]); p = np.array([d[s][1] for s in subs])
        r = m / p
        print(f"{k:14s} {len(subs):2d} {m.mean():8.1f} +-{m.std():5.1f} "
              f"{p.mean():8.1f} +-{p.std():5.1f} {r.mean():8.2f} +-{r.std():5.2f}   {WHAT[k]}")
    if missing:
        print(f"\n  INCOMPLETE: {missing} -- §45.5 says do not read this table yet.")
        continue

    subs = sorted(set.intersection(*[set(d) for d in data.values()]))
    R = {k: np.array([data[k][s][0] / data[k][s][1] for s in subs]) for k in ORDER}

    print(f"\n  pre-registered ordering {' < '.join(ORDER)}")
    full = np.ones(len(subs), bool)
    for a, b in itertools.pairwise(ORDER):
        ok = R[a] < R[b]
        full &= ok
        p = sps.wilcoxon(R[a] - R[b])[1] if len(subs) >= 5 else float("nan")
        print(f"    {a:13s} < {b:13s}  {R[a].mean():6.2f} < {R[b].mean():6.2f}  "
              f"{int(ok.sum())}/{len(subs)} subjects  p={p:.4f}")
    print(f"    FULL ORDERING HOLDS IN {int(full.sum())}/{len(subs)} SUBJECTS "
          f"-> {'CONFIRMED' if full.all() else 'NOT confirmed'}")
    print(f"    parvo_highsf below 1.0? mean {R['parvo_highsf'].mean():.2f}, "
          f"{int((R['parvo_highsf'] < 1).sum())}/{len(subs)} subjects below 1")

    print("\n  mechanism -- which stream moves (vs blur_cs baseline, per subject):")
    base = data["blur_cs"]
    for k in ORDER:
        if k == "blur_cs":
            continue
        dm = np.array([data[k][s][0] / base[s][0] for s in subs])
        dp = np.array([data[k][s][1] / base[s][1] for s in subs])
        print(f"    {k:14s} magno x{dm.mean():5.2f}   parvo x{dp.mean():5.2f}")
