"""Score the section 2.2 PR ratio at each seed's BEST SAVED checkpoint (PLAN 71.8).

Pre-registration (71.8.2), written before any of these numbers existed:
  P1  side of parity survives: diet arms > 1, no-diet arms < 1, all three seeds.
  P2  the LEVEL may move (probably down); no level is claimed, so a shift refutes nothing
      but must be reported and the published numbers replaced, not kept alongside.
  P3  FATAL: if the diet arm falls to or below parity on undegraded weights, the
      dissociation was an artefact of late-training collapse and the section is withdrawn.
"""
import glob, json, os
import numpy as np

CACHE = "/scratch/byrong/VST_data/mpk_feature_cache"
RUNDIR = "/scratch/byrong/VST_data/mpk_runs"
NIMG = 1000

# run -> best saved checkpoint (None => checkpoint_last == checkpoint_150)
ARMS = {
    "diet": [("dkl_bcs_seed0", 125), ("dkl_bcs_seed1", 75), ("dkl_bcs_seed2", 50)],
    "no diet": [("dkl_base_seed0", None), ("dkl_base_seed1", None), ("dkl_base_seed2", None)],
    "swap": [("dkl_bcs_swap_seed0", 100), ("dkl_bcs_swap_seed1", 100), ("dkl_bcs_swap_seed2", None)],
    "parvo_highsf": [("dkl_bcs_parvo_highsf_seed0", None), ("dkl_bcs_parvo_highsf_seed1", None),
                     ("dkl_bcs_parvo_highsf_seed2", None)],
}


def pr(X):
    X = np.asarray(X[:NIMG], dtype=np.float64)
    X -= X.mean(0, keepdims=True)
    lam = np.linalg.eigvalsh(X @ X.T)
    lam = lam[lam > 0]
    return (lam.sum() ** 2) / (lam ** 2).sum()


def verify(run, ep):
    """Re-derive the best saved checkpoint from history.json; refuse a mismatch."""
    h = json.load(open(f"{RUNDIR}/{run}/history.json"))
    avail = [n for n in (25, 50, 75, 100, 125, 150)
             if os.path.exists(f"{RUNDIR}/{run}/checkpoint_{n:03d}.pt") and n - 1 < len(h)]
    best = max(avail, key=lambda n: h[n - 1]["val_top1"])   # checkpoint_NNN == history[NNN-1]
    want = 150 if ep is None else ep
    assert best == want, f"FATAL {run}: history says best={best}, table says {want}"
    return h[best - 1]["val_top1"]


def ratios(model):
    out = {}
    for f in sorted(glob.glob(f"{CACHE}/{model}_subj*.npz")):
        s = int(f.split("_subj")[-1].split(".")[0])
        z = np.load(f)
        out[s] = pr(z["magno"]) / pr(z["parvo"])
    return out


print(f"{'arm':<14}{'seed':<28}{'ckpt':>6}{'acc':>8}{'mean PR ratio':>15}{'side':>8}  per-subj range")
print("-" * 100)
summary = {}
for arm, runs in ARMS.items():
    means = []
    for run, ep in runs:
        acc = verify(run, ep)
        model = f"{run}_trained" if ep is None else f"{run}_ep{ep:03d}_trained"
        r = ratios(model)
        assert len(r) == 8, f"{model}: only {len(r)} subjects cached"
        v = np.array([r[s] for s in sorted(r)])
        means.append(v.mean())
        above = int((v > 1).sum())
        print(f"{arm:<14}{run:<28}{(ep or 150):>6}{acc:>8.4f}{v.mean():>15.4f}{above:>6}/8  "
              f"[{v.min():.3f}, {v.max():.3f}]")
    summary[arm] = np.array(means)
    print(f"{'':14}{'-> arm mean ' + f'{np.mean(means):.4f}':<28}"
          f"seed range {np.ptp(means):.4f}   seeds>1: {int((np.array(means) > 1).sum())}/3")
    print()

d, b = summary["diet"], summary["no diet"]
print("=" * 100)
print(f"P1  diet arms > 1 in {int((d > 1).sum())}/3 seeds; no-diet arms < 1 in {int((b < 1).sum())}/3 seeds")
print(f"    diet  seeds: {np.round(d, 4).tolist()}")
print(f"    base  seeds: {np.round(b, 4).tolist()}")
print(f"    complete separation: {bool(d.min() > b.max())}   (gap {d.min() - b.max():+.4f})")
print(f"P1 VERDICT: {'HELD' if (d > 1).all() and (b < 1).all() else 'FAILED -> see P3'}")
print(f"P3 (fatal) triggered: {bool((d <= 1).any())}")
