"""Full PR-vs-epoch trajectories for the two matched arms (job 26417753).

Tests the §61.6 hypothesis: both arms begin magno-dominant, training ERODES
that dominance in the no-diet network, and the diet PREVENTS the erosion.

FALSIFIERS, stated in slurm/cache_epoch_trajectory.sh before the job ran:

  F1. If dkl_base is FLAT from 25 to 100 and only drops after 100, the
      "erosion during training" reading is wrong -- the drop belongs to
      something specific to late training (LR schedule) rather than to the
      diet contrast.
  F2. If dkl_bcs is NOT flat over 25-75, then its epoch-stability at 100-150
      is a late-training coincidence and §61.2's dissociation is not real.

SCOPE, disclosed with any result: one seed per arm. §58 showed the trained
LEVEL varies across seeds on the diet arm by 0.904; the trajectory SHAPE may
vary too, and seed generality of the shape is untested.
"""
from __future__ import annotations
import glob, os
import numpy as np
from scipy import stats as sps

CACHE = "/scratch/byrong/VST_data/mpk_feature_cache"
STORE = os.path.join(os.environ.get("CLAUDE_JOB_DIR", "/tmp"), "tmp", "pr_store.npz")
NIMG = 1000
EPOCHS = [25, 50, 75, 100, 125, 150]


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


SUBS = list(range(1, 9))


def vec(model):
    r = ratios(model)
    miss = [s for s in SUBS if s not in r]
    if miss:
        raise SystemExit(f"{model}: missing subjects {miss}")
    return np.array([r[s] for s in SUBS])


def name(run, ep):
    # epoch 150 for dkl_bcs/dkl_base_seed0 is the original checkpoint_last
    # cache for base; for bcs the ep150 cache exists and is bit-identical
    # (§61.1). Prefer whichever exists.
    cand = f"{run}_ep{ep:03d}_trained"
    if glob.glob(f"{CACHE}/{cand}_subj01.npz"):
        return cand
    return f"{run}_trained"


ARMS = [("diet    (dkl_bcs_seed0)", "dkl_bcs_seed0", "untrained_seed0"),
        ("no diet (dkl_base_seed0)", "dkl_base_seed0", "untrained_seed0_noblur")]

print("=" * 96)
print("MAGNO/PARVO PR RATIO vs TRAINING EPOCH   (n=8 subjects, NIMG=1000, 1 seed per arm)")
print("=" * 96)
T = {}
for lab, run, untr in ARMS:
    u = vec(untr)
    row = {e: vec(name(run, e)) for e in EPOCHS}
    T[run] = (u, row)
    cells = "  ".join(f"{row[e].mean():6.3f}" for e in EPOCHS)
    print(f"{lab:26s} untrained {u.mean():6.3f} | " + cells)
print(f"{'':26s} {'':16s} | " + "  ".join(f"ep{e:<4d}" for e in EPOCHS))

print("\n  per-epoch detail (mean ± sd across the 8 subjects):")
for lab, run, untr in ARMS:
    u, row = T[run]
    print(f"    {lab}")
    print(f"      untrained  {u.mean():.4f} ± {u.std():.4f}")
    for e in EPOCHS:
        v = row[e]
        print(f"      epoch {e:3d}  {v.mean():.4f} ± {v.std():.4f}"
              f"   above 1.0 in {int((v > 1).sum())}/8")

print("\n" + "=" * 96)
print("FALSIFIER F1 — is the no-diet arm flat from 25 to 100, dropping only after?")
print("=" * 96)
u, row = T["dkl_base_seed0"]
early = row[100].mean() - row[25].mean()
late = row[150].mean() - row[100].mean()
print(f"  ep25 -> ep100 change : {early:+.4f}")
print(f"  ep100 -> ep150 change: {late:+.4f}")
print("\n  step-by-step, paired over the 8 subjects:")
seq = [("untr", u)] + [(f"ep{e}", row[e]) for e in EPOCHS]
for (na, a), (nb, b) in zip(seq, seq[1:]):
    d = b - a
    print(f"    {na:5s} -> {nb:5s}: {a.mean():6.3f} -> {b.mean():6.3f}"
          f"  change {d.mean():+.4f}  p={sps.wilcoxon(d)[1]:.4f}"
          f"  down {int((d<0).sum())}/8")
frac_early = early / (early + late) if (early + late) != 0 else np.nan
f1 = abs(early) < 0.1 and abs(late) > 0.3
print(f"\n  share of total decline occurring BEFORE epoch 100: {frac_early:.0%}")
print(f"  -> F1 {'TRIGGERED: the decline is late-training only, hypothesis WRONG' if f1 else 'NOT triggered: the decline is progressive, not late-training-specific'}")

print("\n" + "=" * 96)
print("FALSIFIER F2 — is the diet arm flat over epochs 25-75?")
print("=" * 96)
u2, row2 = T["dkl_bcs_seed0"]
early_rng = max(row2[e].mean() for e in (25, 50, 75)) - min(row2[e].mean() for e in (25, 50, 75))
full_rng = max(row2[e].mean() for e in EPOCHS) - min(row2[e].mean() for e in EPOCHS)
print(f"  ep25/50/75 means: " + "  ".join(f"{row2[e].mean():.4f}" for e in (25, 50, 75)))
print(f"  range over 25-75 : {early_rng:.4f}")
print(f"  range over 25-150: {full_rng:.4f}")
print(f"  (no-diet range over 25-150 for comparison: "
      f"{max(row[e].mean() for e in EPOCHS)-min(row[e].mean() for e in EPOCHS):.4f})")
f2 = early_rng > 0.3
print(f"  -> F2 {'TRIGGERED: the diet arm is NOT flat early, dissociation not real' if f2 else 'NOT triggered: the diet arm is flat early as well as late'}")

print("\n" + "=" * 96)
print("THE COMPARISON THE HYPOTHESIS ACTUALLY MAKES")
print("=" * 96)
print("  At every epoch: is the diet arm above the no-diet arm, and when do they part?")
for e in EPOCHS:
    a, b = row2[e], row[e]
    d = a - b
    print(f"    ep{e:3d}: diet {a.mean():6.3f}  no diet {b.mean():6.3f}"
          f"  diff {d.mean():+7.3f}  p={sps.wilcoxon(d)[1]:.4f}"
          f"  pos {int((d>0).sum())}/8  dz {d.mean()/d.std(ddof=1):+6.2f}")

print("\n  total movement from each arm's own untrained front end to epoch 150:")
for lab, run, untr in ARMS:
    uu, rr = T[run]
    d = rr[150] - uu
    print(f"    {lab}: {uu.mean():6.3f} -> {rr[150].mean():6.3f}"
          f"  change {d.mean():+.4f}  p={sps.wilcoxon(d)[1]:.4f}"
          f"  down {int((d<0).sum())}/8")

print("\n  VERDICT on §61.6, by the two falsifiers:")
if f1 or f2:
    print("    HYPOTHESIS FALSIFIED -- see above.")
else:
    print("    HYPOTHESIS SURVIVES its pre-registered falsifiers. Still one seed")
    print("    per arm; the trajectory SHAPE has no seed replication (§58 range")
    print("    0.904 on the level), so state it as a single-seed trajectory.")
