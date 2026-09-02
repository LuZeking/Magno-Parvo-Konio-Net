"""PAPER_PLAN §47.6: the pre-registered test on a THIRD condition.

§45.3's account is that the diet's role is not to *supply* the magno/parvo
dimensionality asymmetry -- the front end does that -- but to make training
KEEP whatever asymmetry the input supplies. Every condition tested so far had
a front end whose asymmetry pointed the same way (magno > parvo, ratio > 1),
so "keep" and "drift toward magno-dominance" made the same prediction.

`parvo_highsf` inverts the front end (§47.1: ratio 0.56 gabor / 0.59 untrained,
below 1.0 in 8/8 subjects) because stream 2 receives the full-bandwidth
luminance carrier. A diet-trained model on that front end therefore separates
the two accounts:

  KEEP (§45.3)          -> retention ~ 1, the trained ratio stays BELOW 1.0
  drift to magno        -> retention > 1, the trained ratio climbs toward/past 1

The reference is exact: `untrained_parvo_highsf` carries provenance identical
to the trained model on every key except `prov_checkpoint` (lum_blur=True,
lum_cs=True, parvo_highsf=True, age=48, seed=0), so this is the cleanest
front-end pairing in the retention series -- see §45.6 for why that matters.
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

def streams(model):
    """Per-subject (magno PR, parvo PR). Checks the store PER SUBJECT, so a
    cell cannot freeze at whatever subject count existed on first read --
    the partial-table trap of §39.3/§41.2, in code."""
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

# (trained, own front end, learning-free front end, label)
ROWS = [
    ("dkl_bcs_seed0_trained",             "untrained_seed0",        "gabor_blur_cs",      "diet / normal"),
    ("dkl_bcs_parvo_highsf_seed0_trained","untrained_parvo_highsf", "gabor_parvo_highsf", "diet / parvo_highsf"),
    ("dkl_base_seed0_trained",            "untrained_seed0_noblur", "gabor_noblur",       "no diet / normal"),
]

print("=" * 96)
print("§47.6  RETENTION ON AN INVERTED FRONT END (n=8, NIMG=1000)")
print("=" * 96)
hdr = f"{'condition':22s} {'front ratio':>13s} {'trained ratio':>14s} {'norm ret':>13s} {'>1':>5s} {'raw ret':>13s} {'>0':>5s}"
for ref_kind, ref_idx in (("own untrained front end", 1), ("learning-free gabor bank", 2)):
    print(f"\n-- reference: {ref_kind} " + "-" * (96 - 15 - len(ref_kind)))
    print(hdr)
    for row in ROWS:
        trained, front, label = row[0], row[ref_idx], row[3]
        t, f = ratios(trained), ratios(front)
        subs = sorted(set(t) & set(f))
        tv = np.array([t[s] for s in subs]); fv = np.array([f[s] for s in subs])
        nrm, raw = tv / fv, tv - fv
        print(f"{label:22s} {fv.mean():7.2f}±{fv.std():4.2f} {tv.mean():8.2f}±{tv.std():4.2f} "
              f"{nrm.mean():8.2f}±{nrm.std():4.2f} {int((nrm>1).sum()):2d}/{len(subs)} "
              f"{raw.mean():8.2f}±{raw.std():4.2f} {int((raw>0).sum()):2d}/{len(subs)}")

# The pre-registered question, stated as a sign test on the trained ratio.
t = ratios("dkl_bcs_parvo_highsf_seed0_trained")
f = ratios("untrained_parvo_highsf")
subs = sorted(set(t) & set(f))
tv = np.array([t[s] for s in subs]); fv = np.array([f[s] for s in subs])
print("\n" + "=" * 96)
print("PRE-REGISTERED CALL (§47.6): does the trained model KEEP the inverted asymmetry?")
print("=" * 96)
print(f"  front end   magno/parvo = {fv.mean():.3f} ± {fv.std():.3f}   below 1.0 in {int((fv<1).sum())}/{len(subs)}")
print(f"  trained     magno/parvo = {tv.mean():.3f} ± {tv.std():.3f}   below 1.0 in {int((tv<1).sum())}/{len(subs)}")
print(f"  paired Wilcoxon trained vs front end: p = {sps.wilcoxon(tv - fv)[1]:.4f}")
print(f"  per-subject trained ratios: " + " ".join(f"{tv[i]:.2f}" for i in range(len(subs))))
print("\n  KEEP  predicts trained ratio stays < 1.0.")
print("  DRIFT predicts trained ratio climbs toward / above 1.0.")
print(f"  -> observed: {'KEEP (stays inverted)' if (tv<1).sum()==len(subs) else 'MIXED / DRIFT -- read the per-subject row'}")

# Contrast against the normal-front-end diet model: same diet, opposite input asymmetry.
a = ratios("dkl_bcs_seed0_trained"); b = ratios("untrained_seed0")
ss = sorted(set(a) & set(b))
na = np.array([a[s] / b[s] for s in ss])
nb = tv / fv
n = min(len(na), len(nb))
print(f"\n  retention, diet/normal {na[:n].mean():.2f} vs diet/parvo_highsf {nb[:n].mean():.2f}, "
      f"paired p = {sps.wilcoxon(na[:n] - nb[:n])[1]:.4f}, {int((na[:n]>nb[:n]).sum())}/{n}")

# ---------------------------------------------------------------------------
# §47.6 addendum. The KEEP account is falsified: the trained parvo_highsf model
# does not stay inverted, it lands at 1.64 -- within 0.04 of the trained
# NORMAL-front-end diet model (1.60), despite being handed a front end that is
# 2.4x lower (0.59 vs 1.40). That is not retention. It is a SET POINT: the diet
# drives training to a target ratio the input does not choose.
#
# The test that distinguishes the two accounts directly: compare the spread of
# the FRONT-END ratios across conditions against the spread of the TRAINED
# ratios across the same conditions. Retention predicts the trained spread
# tracks the front-end spread. A set point predicts the trained spread collapses.
# ---------------------------------------------------------------------------
print("\n" + "=" * 96)
print("SET-POINT TEST -- does training track the input, or converge on a target?")
print("=" * 96)
ARMS = [
    ("diet",    "dkl_bcs_seed0_trained",              "untrained_seed0"),
    ("diet",    "dkl_bcs_parvo_highsf_seed0_trained", "untrained_parvo_highsf"),
    ("no diet", "dkl_base_seed0_trained",             "untrained_seed0_noblur"),
    ("no diet", "dkl_base_seed1_trained",             "untrained_seed0_noblur"),
]
print(f"{'arm':9s} {'model':36s} {'front end':>11s} {'trained':>11s}")
fe, tr, lab = [], [], []
for arm, trained, front in ARMS:
    t, f = ratios(trained), ratios(front)
    subs = sorted(set(t) & set(f))
    fv = np.array([f[s] for s in subs]); tv = np.array([t[s] for s in subs])
    fe.append(fv); tr.append(tv); lab.append(arm)
    print(f"{arm:9s} {trained:36s} {fv.mean():7.2f}±{fv.std():4.2f} {tv.mean():7.2f}±{tv.std():4.2f}")

d0, d1 = 0, 1                      # the two diet arms
print(f"\nTwo diet arms, identical schedule and seed, front ends 2.4x apart:")
print(f"  front end : {fe[d0].mean():.3f} vs {fe[d1].mean():.3f}   ratio of means = {fe[d0].mean()/fe[d1].mean():.2f}x"
      f"   paired p = {sps.wilcoxon(fe[d0]-fe[d1])[1]:.4f}")
print(f"  trained   : {tr[d0].mean():.3f} vs {tr[d1].mean():.3f}   ratio of means = {tr[d0].mean()/tr[d1].mean():.2f}x"
      f"   paired p = {sps.wilcoxon(tr[d0]-tr[d1])[1]:.4f}")
sp_f = abs(fe[d0].mean() - fe[d1].mean()); sp_t = abs(tr[d0].mean() - tr[d1].mean())
print(f"  the front-end gap is {sp_f:.3f}; after training it is {sp_t:.3f}"
      f"  -> {sp_f/sp_t:.0f}x compression")
print(f"\nDiet arms vs no-diet arms, trained ratios only:")
print(f"  diet    trained: {np.concatenate([tr[0],tr[1]]).mean():.3f} ± {np.concatenate([tr[0],tr[1]]).std():.3f}")
print(f"  no diet trained: {np.concatenate([tr[2],tr[3]]).mean():.3f} ± {np.concatenate([tr[2],tr[3]]).std():.3f}")
for i in (0, 1):
    for j in (2, 3):
        print(f"  {ARMS[i][1][:28]:28s} vs {ARMS[j][1][:24]:24s} p={sps.wilcoxon(tr[i]-tr[j])[1]:.4f} "
              f"{int((tr[i]>tr[j]).sum())}/8")
