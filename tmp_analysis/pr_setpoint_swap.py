"""PRE-REGISTERED (written and committed BEFORE the run): the set-point claim
of PAPER_PLAN §49.2 tested on a FOURTH front end, `swap`.

§49.2 replaced the KEEP account with a SET POINT: diet-trained networks
converge on magno/parvo PR ~1.62 from front ends that are magno-dominant
(1.40) and parvo-dominant (0.59) alike, a 22x compression of a 0.816
front-end gap. That claim currently rests on TWO diet-trained models
(`dkl_bcs_seed0`, `dkl_bcs_parvo_highsf_seed0`), so "converges on a target"
and "happens to land near 1.6 twice" are not yet separated.

`swap` separates them, and it costs nothing: its features are already cached
for all 8 subjects (§48), and it was never measured on the PR instrument
(§49.6's "no cheap measurement left" is wrong on this one point).

Why `swap` is the right fourth case:

  1. Its front end is the MOST EXTREME of the four (§47.1): 1.91 untrained /
     3.37 gabor, against parvo_highsf's 0.59. Front ends now span 3.2x.
  2. It is diet-trained but the transform sits on the CHROMATIC carrier
     (`transform_target=parvo`), luminance untouched. So it dissociates
     "diet-trained" from "diet on the achromatic carrier" -- the two are
     confounded in every arm measured so far.
  3. The reference is provenance-EXACT. `untrained_swap` matches
     `dkl_bcs_swap_seed0_trained` on every prov_* key except prov_checkpoint
     (lum_blur=True, lum_cs=True, age=48, transform_target=parvo, seed=0).
     Only the parvo_highsf pair (§49.2) is this clean; the diet/normal arm's
     reference is unstamped and audited only by submit line (§45.6).

THE THREE OUTCOMES, declared in advance so no result can be storytold:

  A. trained ~1.6 (comes DOWN from 1.91).  The set point is a property of the
     diet RECIPE, holding across three front ends spanning 0.59-1.91. This is
     the strongest outcome for §49.2 and widens it.

  B. trained ~0.97 (parity, like the no-diet arms).  The set point requires
     the diet on the ACHROMATIC carrier; applied to rg it does not install
     one, and training collapses to parity as it does without any diet. This
     NARROWS §49.2 but unifies it with the §25/§47.4 conjunction result --
     arguably the more interesting outcome, and it would make the set point
     and the shape bias require the same conjunction.

  C. trained tracks its front end (stays >=1.9).  Retention, on the one
     condition handed the most asymmetry. §49.2's generality fails: the
     honest statement becomes "the diet on luminance installs a set point;
     the diet on rg does not compress", and §49.2 must be qualified wherever
     it appears.

Outcome C is the one that costs the paper something, and it is a live
possibility precisely because swap's front end is the extreme case -- which
is why this is worth running rather than assuming.

DISCLOSURE that travels with any result here: swap's CORTICAL behaviour is
degenerate (§48: Um 0.288 vs Up 0.006 -- the parvo stream carries almost no
unique cortical variance), so swap must not be used to make a cortical claim.
This is a statement about feature geometry only. Unlike `parvo_highsf` (§49.4
disclosure 3) swap is architecture-MATCHED to the diet arm -- same channel
counts, same fan-in, only the transform target differs -- so it carries no
architectural confound.
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


def paired(model, ref):
    t, f = ratios(model), ratios(ref)
    subs = sorted(set(t) & set(f))
    return (np.array([t[s] for s in subs]),
            np.array([f[s] for s in subs]), subs)


# arm, trained model, own untrained front end, learning-free gabor front end
ARMS = [
    ("diet/lum",   "dkl_bcs_seed0_trained",              "untrained_seed0",        "gabor_blur_cs"),
    ("diet/highsf","dkl_bcs_parvo_highsf_seed0_trained", "untrained_parvo_highsf", "gabor_parvo_highsf"),
    ("diet/rg",    "dkl_bcs_swap_seed0_trained",         "untrained_swap",         "gabor_swap"),
    ("no diet",    "dkl_base_seed0_trained",             "untrained_seed0_noblur", "gabor_noblur"),
    ("no diet",    "dkl_base_seed1_trained",             "untrained_seed0_noblur", "gabor_noblur"),
]

print("=" * 100)
print("PRE-REGISTERED: the §49.2 set point on a FOURTH front end (`swap`), n=8, NIMG=1000")
print("=" * 100)
print(f"{'arm':11s} {'model':37s} {'front(untr)':>12s} {'front(gabor)':>13s} {'TRAINED':>12s}")
tab = {}
for arm, trained, front, gab in ARMS:
    tv, fv, subs = paired(trained, front)
    gv = np.array([ratios(gab)[s] for s in subs])
    tab[trained] = (tv, fv, gv, subs, arm)
    print(f"{arm:11s} {trained:37s} {fv.mean():7.2f}±{fv.std():4.2f} "
          f"{gv.mean():8.2f}±{gv.std():4.2f} {tv.mean():7.3f}±{tv.std():5.3f}")

SWAP = "dkl_bcs_swap_seed0_trained"
tv, fv, gv, subs, _ = tab[SWAP]
print("\n" + "=" * 100)
print("THE PRE-REGISTERED CALL")
print("=" * 100)
print(f"  front end (untrained)  magno/parvo = {fv.mean():.3f} ± {fv.std():.3f}"
      f"   above 1.0 in {int((fv > 1).sum())}/{len(subs)}")
print(f"  trained                magno/parvo = {tv.mean():.3f} ± {tv.std():.3f}"
      f"   above 1.0 in {int((tv > 1).sum())}/{len(subs)}")
print(f"  paired Wilcoxon trained vs own front end: p = {sps.wilcoxon(tv - fv)[1]:.4f}")
print(f"  per-subject trained: " + " ".join(f"{v:.2f}" for v in tv))

diet_lum = tab["dkl_bcs_seed0_trained"][0]
diet_hsf = tab["dkl_bcs_parvo_highsf_seed0_trained"][0]
setpoint = np.concatenate([diet_lum, diet_hsf]).mean()
nodiet = np.concatenate([tab["dkl_base_seed0_trained"][0],
                         tab["dkl_base_seed1_trained"][0]]).mean()
d_sp, d_nd = abs(tv.mean() - setpoint), abs(tv.mean() - nodiet)
print(f"\n  §49.2 set point (two diet arms) = {setpoint:.3f}")
print(f"  no-diet trained parity          = {nodiet:.3f}")
print(f"  swap trained sits {d_sp:.3f} from the set point and {d_nd:.3f} from parity")
verdict = ("A: SET POINT IS RECIPE-LEVEL" if d_sp < d_nd / 2 else
           "B: SET POINT NEEDS THE ACHROMATIC CARRIER" if d_nd < d_sp / 2 else
           "C: TRACKS ITS FRONT END / AMBIGUOUS -- read the rows")
print(f"  -> outcome {verdict}")

print("\n  contrasts on the trained ratio (paired, n=8):")
for name in ("dkl_bcs_seed0_trained", "dkl_bcs_parvo_highsf_seed0_trained",
             "dkl_base_seed0_trained", "dkl_base_seed1_trained"):
    ov = tab[name][0]
    n = min(len(tv), len(ov))
    print(f"    swap vs {name:37s} p={sps.wilcoxon(tv[:n] - ov[:n])[1]:.4f} "
          f"swap>other {int((tv[:n] > ov[:n]).sum())}/{n}")

print("\n  front-end span now covered by diet-trained models:")
fes = [tab[a[1]][1].mean() for a in ARMS if a[0].startswith("diet")]
trs = [tab[a[1]][0].mean() for a in ARMS if a[0].startswith("diet")]
print(f"    front ends {min(fes):.2f} .. {max(fes):.2f}  (span {max(fes)-min(fes):.3f}, {max(fes)/min(fes):.2f}x)")
print(f"    trained    {min(trs):.2f} .. {max(trs):.2f}  (span {max(trs)-min(trs):.3f})")
print(f"    compression = {(max(fes)-min(fes))/max(max(trs)-min(trs), 1e-9):.1f}x")
