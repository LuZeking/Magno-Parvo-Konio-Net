"""Seed replication of the PR set-point numbers (job 26413641, 56 caches).

THE QUESTION. Every headline in PAPER_PLAN §47/§49/§50 rests on ONE checkpoint
per condition. §49.4 instructs reporting the two diet-on-luminance arms as
"converge to within 2%" -- a gap of 0.037 in the trained magno/parvo PR ratio
(1.603 vs 1.640). §50 then made the level load-bearing by showing it is
channel-specific (swap 2.225). Neither number is interpretable until we know
how much the subject-mean ratio moves under nothing but a different training
seed. That has never been measured.

WHAT THIS DECIDES.

  If seed spread << 0.037: the 0.037 gap is systematic. The two diet-on-lum
  arms differ slightly but reliably, and §55's d_z=+1.02 for that contrast is
  measuring a real (small) condition difference.

  If seed spread >~ 0.037: the two arms are within seed noise of each other.
  This STRENGTHENS "converge" (they are statistically identical) but kills any
  claim that the set point is a precise number, and §55's d_z=+1.02 was
  measuring seed noise rather than a condition effect. §49.4 must then be
  worded as "indistinguishable within seed noise", not "within 2%".

Either outcome is reportable. What is NOT acceptable is continuing to quote
0.037 as if it were resolved.

SEEDS ARE NOT SUBJECTS. n=3 seeds gives no usable inferential power on its
own. The seed axis is used here as a *reproducibility* check on the
subject-mean, and the paired subject tests are run WITHIN each seed so that
every reported p-value still has n=8 pairs. A contrast is called robust only
if it holds in all three seeds independently -- a sign-consistency criterion,
not a pooled test that would inflate n.
"""
from __future__ import annotations
import glob, os, itertools
import numpy as np
from scipy import stats as sps

CACHE = "/scratch/byrong/VST_data/mpk_feature_cache"
STORE = os.path.join(os.environ.get("CLAUDE_JOB_DIR", "/tmp"), "tmp", "pr_store.npz")
NIMG = 1000

# The gap §49.4 calls "converge to within 2%".
GAP_49_4 = 0.037


def pr(X):
    X = np.asarray(X[:NIMG], dtype=np.float64)
    X -= X.mean(0, keepdims=True)
    lam = np.linalg.eigvalsh(X @ X.T)
    lam = lam[lam > 0]
    return (lam.sum() ** 2) / (lam ** 2).sum()


_store = dict(np.load(STORE)) if os.path.exists(STORE) else {}


def streams(model):
    """Per-subject (magno PR, parvo PR); store checked PER SUBJECT so a cell
    cannot freeze at a partial subject count (§39.3/§41.2)."""
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


# condition -> the three seed checkpoints, in seed order
CONDS = {
    "diet/lum":    ["dkl_bcs_seed0_trained",
                    "dkl_bcs_seed1_trained",
                    "dkl_bcs_seed2_trained"],
    "diet/highsf": ["dkl_bcs_parvo_highsf_seed0_trained",
                    "dkl_bcs_parvo_highsf_seed1_trained",
                    "dkl_bcs_parvo_highsf_seed2_trained"],
    "diet/rg":     ["dkl_bcs_swap_seed0_trained",
                    "dkl_bcs_swap_seed1_trained",
                    "dkl_bcs_swap_seed2_trained"],
    "no diet":     ["dkl_base_seed0_trained",
                    "dkl_base_seed1_trained",
                    "dkl_base_seed2_trained"],
}

# Pull every arm, keeping the per-subject vectors on a common subject set.
R = {}
for cond, models in CONDS.items():
    for si, m in enumerate(models):
        r = ratios(m)
        if len(r) != 8:
            print(f"  !! {m}: {len(r)} subjects, expected 8")
        R[(cond, si)] = r

SUBS = sorted(set.intersection(*[set(v) for v in R.values()]))
V = {k: np.array([v[s] for s in SUBS]) for k, v in R.items()}
print(f"common subjects: n={len(SUBS)}  {SUBS}\n")

print("=" * 96)
print("SEED VARIANCE IN THE TRAINED MAGNO/PARVO PR RATIO  (n=8 subjects per cell, NIMG=1000)")
print("=" * 96)
print(f"{'condition':12s} {'seed0':>14s} {'seed1':>14s} {'seed2':>14s}"
      f" {'mean':>7s} {'range':>7s} {'sd':>6s}")
mu = {}
for cond in CONDS:
    m = np.array([V[(cond, i)].mean() for i in range(3)])
    mu[cond] = m
    cells = "".join(f"{V[(cond,i)].mean():8.3f}±{V[(cond,i)].std():5.3f}" for i in range(3))
    print(f"{cond:12s} {cells} {m.mean():7.3f} {m.max()-m.min():7.3f} {m.std(ddof=1):6.3f}")

print("\n  'range' and 'sd' are ACROSS SEEDS of the subject-mean; the ± inside each")
print("  cell is across the 8 subjects within that one checkpoint.")

print("\n" + "=" * 96)
print(f"THE §49.4 QUESTION: is seed spread smaller than the {GAP_49_4} gap it calls 'within 2%'?")
print("=" * 96)
worst = max(mu[c].max() - mu[c].min() for c in CONDS)
lum_spread = mu["diet/lum"].max() - mu["diet/lum"].min()
hsf_spread = mu["diet/highsf"].max() - mu["diet/highsf"].min()
print(f"  seed range, diet/lum      : {lum_spread:.3f}")
print(f"  seed range, diet/highsf   : {hsf_spread:.3f}")
print(f"  seed range, worst arm     : {worst:.3f}")
print(f"  the §49.4 between-arm gap : {GAP_49_4:.3f}")
verdict = ("SYSTEMATIC -- seed spread is well below the gap; the 0.037 is a real "
           "condition difference" if max(lum_spread, hsf_spread) < GAP_49_4 / 2 else
           "WITHIN SEED NOISE -- the two diet-on-lum arms are indistinguishable; "
           "0.037 must not be quoted as a measured convergence")
print(f"  -> {verdict}")

# Every seed x seed pairing of the two diet-on-luminance arms: does the sign of
# the 0.037 gap even survive changing which checkpoint is picked?
print("\n  all 9 diet/lum x diet/highsf pairings (the contrast §49.4 reports):")
signs = []
for i, j in itertools.product(range(3), range(3)):
    a, b = V[("diet/lum", i)], V[("diet/highsf", j)]
    d = a - b
    p = sps.wilcoxon(d)[1]
    dz = d.mean() / d.std(ddof=1)
    signs.append(np.sign(d.mean()))
    print(f"    lum s{i} vs highsf s{j}: {a.mean():6.3f} vs {b.mean():6.3f}"
          f"  diff {d.mean():+7.3f}  p={p:.4f}  dz {dz:+6.2f}"
          f"  pos {int((d>0).sum())}/8")
print(f"    sign of the difference is consistent across pairings: "
      f"{'YES' if len(set(signs)) == 1 else 'NO -- it flips'}")

print("\n" + "=" * 96)
print("DOES §50's CHANNEL-SPECIFIC SET POINT SURVIVE SEEDS?")
print("=" * 96)
print("  §50: diet/rg trained at 2.225 vs diet/lum 1.603, diet/highsf 1.640,")
print("  differing from both at p=.0078, 8/8. Re-tested over all seed pairings:")
for other in ("diet/lum", "diet/highsf"):
    ok = 0
    tot = 0
    worst_p = 0.0
    for i, j in itertools.product(range(3), range(3)):
        a, b = V[("diet/rg", i)], V[(other, j)]
        d = a - b
        p = sps.wilcoxon(d)[1]
        tot += 1
        if p < 0.05 and (d > 0).all():
            ok += 1
        worst_p = max(worst_p, p)
    print(f"    diet/rg vs {other:12s}: {ok}/{tot} pairings significant AND 8/8 positive"
          f"   (worst p = {worst_p:.4f})")
rg = mu["diet/rg"].mean()
print(f"    diet/rg  mean over seeds {rg:.3f} (range {mu['diet/rg'].max()-mu['diet/rg'].min():.3f})")
print(f"    diet/lum mean over seeds {mu['diet/lum'].mean():.3f}")
print(f"    separation {rg - mu['diet/lum'].mean():.3f} vs worst seed range {worst:.3f}"
      f"  -> {(rg - mu['diet/lum'].mean())/max(worst,1e-9):.1f}x the seed noise")

print("\n" + "=" * 96)
print("THE LOAD-BEARING CONTRAST: diet vs no diet, all 27 seed pairings")
print("=" * 96)
print("  §50.4's surviving claim is that the diet fixes WHICH SIDE OF PARITY")
print("  training ends on. If that is real it must hold for every seed pairing.")
allok, alltot = 0, 0
for diet in ("diet/lum", "diet/highsf", "diet/rg"):
    ok, tot, worst_p, mind = 0, 0, 0.0, 9e9
    for i, j in itertools.product(range(3), range(3)):
        a, b = V[(diet, i)], V[("no diet", j)]
        d = a - b
        p = sps.wilcoxon(d)[1]
        tot += 1
        mind = min(mind, d.mean())
        if p < 0.05 and (d > 0).all():
            ok += 1
        worst_p = max(worst_p, p)
    allok += ok
    alltot += tot
    print(f"    {diet:12s} vs no diet: {ok}/{tot} pairings p<.05 AND 8/8 positive"
          f"   worst p={worst_p:.4f}  smallest mean diff {mind:+.3f}")
print(f"    TOTAL {allok}/{alltot}")

print("\n" + "=" * 96)
print("PARITY CHECK: where does each condition sit relative to 1.0?")
print("=" * 96)
for cond in CONDS:
    above = [int((V[(cond, i)] > 1).sum()) for i in range(3)]
    print(f"    {cond:12s} above 1.0 in {above} of 8 subjects (seeds 0,1,2)")
