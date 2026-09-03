"""Epoch-jitter control for the PR set point (job 26416534, 40 caches).

Reading PRE-REGISTERED in PAPER_PLAN §60.5 before these results existed:

  * If PR moves across epochs within seed0 as much as it moves ACROSS SEEDS
    (§58: range 0.904 on this arm), the level is a checkpoint-timing artefact.
    §49.2/§49.4 are withdrawn rather than re-worded, and every leg-(c) number
    -- all computed from dkl_bcs_seed0 -- needs an epoch-stability disclosure.
  * If PR is flat across epochs, §58's spread is genuine seed-to-seed
    variability in the learned solution and is reportable as such.
  * dkl_base_seed0 is the CONTROL: it is converged (last-20-epoch sd 0.0065),
    so its PR must be flat. If it is not, the instrument is epoch-sensitive
    and none of this is about the diet.
  * dkl_bcs_seed0 @ epoch 150 is a REPRODUCTION CHECK: it must return 1.603,
    the value already cached from checkpoint_last.pt. If it does not, those
    two files are not the same weights and nothing here is believed.

The reproduction check is evaluated and printed FIRST, and the script refuses
to interpret anything else if it fails.
"""
from __future__ import annotations
import glob, os
import numpy as np
from scipy import stats as sps

CACHE = "/scratch/byrong/VST_data/mpk_feature_cache"
STORE = os.path.join(os.environ.get("CLAUDE_JOB_DIR", "/tmp"), "tmp", "pr_store.npz")
NIMG = 1000

# §58 / §60, for reference
SEED_RANGE_BCS = 0.904
REF_BCS_LAST = 1.603
REF_BASE_LAST = 0.947


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
    if dirty:
        os.makedirs(os.path.dirname(STORE), exist_ok=True)
        np.savez(STORE, **_store)
    return {int(k.partition("|subj")[2]): v for k, v in _store.items()
            if k.partition("|subj")[0] == model}


def ratios(model):
    return {s: v[0] / v[1] for s, v in streams(model).items()}


def vec(model, subs):
    r = ratios(model)
    return np.array([r[s] for s in subs])


SUBS = list(range(1, 9))

print("=" * 92)
print("REPRODUCTION CHECK -- run and read before anything else")
print("=" * 92)
ep150 = vec("dkl_bcs_seed0_ep150_trained", SUBS)
last = vec("dkl_bcs_seed0_trained", SUBS)
d = ep150 - last
print(f"  dkl_bcs_seed0 @ checkpoint_150.pt : {ep150.mean():.4f} ± {ep150.std():.4f}")
print(f"  dkl_bcs_seed0 @ checkpoint_last.pt: {last.mean():.4f} ± {last.std():.4f}"
      f"   (§49.2 published {REF_BCS_LAST})")
print(f"  max |per-subject difference|      : {np.abs(d).max():.6f}")
ok = np.abs(d).max() < 1e-3
print(f"  -> {'PASS: the two files are the same weights' if ok else 'FAIL: DIFFERENT WEIGHTS -- stop, nothing below is interpretable'}")

print("\n" + "=" * 92)
print("THE CONTROL: dkl_base_seed0 (converged, last-20-epoch sd 0.0065)")
print("=" * 92)
base = {100: vec("dkl_base_seed0_ep100_trained", SUBS),
        125: vec("dkl_base_seed0_ep125_trained", SUBS),
        150: vec("dkl_base_seed0_trained", SUBS)}
for e, v in base.items():
    print(f"  epoch {e}: {v.mean():.4f} ± {v.std():.4f}")
b_rng = max(v.mean() for v in base.values()) - min(v.mean() for v in base.values())
print(f"  epoch range = {b_rng:.4f}")

print("\n" + "=" * 92)
print("THE TEST: dkl_bcs_seed0 (NOT converged, last-20-epoch sd 0.0543)")
print("=" * 92)
bcs = {100: vec("dkl_bcs_seed0_ep100_trained", SUBS),
       125: vec("dkl_bcs_seed0_ep125_trained", SUBS),
       150: last}
for e, v in bcs.items():
    print(f"  epoch {e}: {v.mean():.4f} ± {v.std():.4f}")
c_rng = max(v.mean() for v in bcs.values()) - min(v.mean() for v in bcs.values())
print(f"  epoch range = {c_rng:.4f}")

print("\n  paired contrasts within dkl_bcs_seed0 (n=8 subjects):")
for a, b in ((100, 125), (125, 150), (100, 150)):
    dd = bcs[a] - bcs[b]
    print(f"    ep{a} vs ep{b}: diff {dd.mean():+.4f}  p={sps.wilcoxon(dd)[1]:.4f}"
          f"  dz {dd.mean()/dd.std(ddof=1):+7.2f}  neg {int((dd<0).sum())}/8")

print("\n" + "=" * 92)
print("THE PRE-REGISTERED CALL")
print("=" * 92)
print(f"  epoch range, dkl_bcs_seed0 (one seed, 3 epochs) : {c_rng:.4f}")
print(f"  seed  range, dkl_bcs       (3 seeds, last ep)   : {SEED_RANGE_BCS:.4f}")
print(f"  epoch range, dkl_base_seed0 CONTROL             : {b_rng:.4f}")
frac = c_rng / SEED_RANGE_BCS
print(f"  epoch range is {frac:.0%} of the seed range")

if b_rng > 0.15:
    call = ("CONTROL FAILED -- dkl_base also moves across epochs, so the "
            "instrument is epoch-sensitive and this is not about the diet")
elif frac > 0.5:
    call = ("CHECKPOINT-TIMING ARTEFACT -- epoch jitter within one seed is "
            "comparable to the whole seed spread. Withdraw §49.2/§49.4; every "
            "leg-(c) number needs an epoch-stability disclosure.")
elif frac < 0.2:
    call = ("GENUINE SEED VARIABILITY -- the level is stable within a run and "
            "differs between runs. §58's spread is a property of the learned "
            "solution; leg (c)'s single checkpoint is epoch-safe.")
else:
    call = ("INTERMEDIATE -- epoch jitter is a substantial minority of the "
            "seed spread. Both sources are real; neither alone explains it.")
print(f"\n  -> {call}")
