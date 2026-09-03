"""§67 Test C: does any headline measure track clean accuracy within the diet arm?

The NMI M1 criticism is that the diet arm's low accuracy, not the diet, drives
the dissociations. The diet arm supplies its own natural experiment: its three
seeds span 0.347-0.527 final val_top1 (§67.1), a 0.18 range, at identical
architecture, data, schedule and front end. A measure that is an accuracy
artefact should track that range. One that is a property of the diet should not.

n=3, so this cannot establish anything on its own. It is a screen: a measure
that moves monotonically with accuracy across the widest accuracy range we have
inside one condition is EXPOSED and needs Test B; one that is flat is not.
"""
from __future__ import annotations
import json, os
import numpy as np

R = "/scratch/byrong/VST_data/mpk_runs"
SEEDS = (0, 1, 2)

acc = {}
for s in SEEDS:
    h = json.load(open(f"{R}/dkl_bcs_seed{s}/history.json"))
    acc[s] = h[-1]["val_top1"]

def probe(s):
    d = json.load(open(f"results/shape_texture/dkl_bcs_seed{s}_trained.json"))
    return d["streams"]["magno"]["bias"]

def behaviour(s):
    d = json.load(open(f"results/cue_conflict/dkl_bcs_seed{s}.json"))
    return d["conditions"]["magno_only"]["shape_bias"]

def pr_ratio(s):
    z = dict(np.load(os.environ["CLAUDE_JOB_DIR"] + "/tmp/pr_store.npz"))
    m = f"dkl_bcs_seed{s}_trained"
    v = [z[k] for k in z if k.partition("|subj")[0] == m]
    return float(np.mean([a[0] / a[1] for a in v]))

def ood_auc(s):
    d = json.load(open(f"results/ood/dkl_bcs_seed{s}.json"))
    fam = d["families"]
    keys = [k for k in fam if k not in ("low-pass", "contrast")]
    out = []
    for k in keys:
        f = fam[k]
        a = f.get("rel_auc", f.get("relative_auc"))
        if a is not None:
            out.append(a)
    return float(np.mean(out)) if out else np.nan

MEAS = [("probe bias (magno)", probe),
        ("behaviour shape bias", behaviour),
        ("PR ratio (magno/parvo)", pr_ratio),
        ("OOD rel-AUC (untrained-on)", ood_auc)]

a = np.array([acc[s] for s in SEEDS])
print("=" * 86)
print("§67 TEST C -- does a headline measure track accuracy WITHIN the diet arm?")
print("=" * 86)
print(f"  final val_top1 by seed: {np.array2string(a, precision=3)}  (range {np.ptp(a):.3f})")
print(f"\n  {'measure':30s} {'seed0':>8s} {'seed1':>8s} {'seed2':>8s} {'range':>8s} "
      f"{'r w/ acc':>9s} {'verdict':>10s}")
for name, fn in MEAS:
    try:
        v = np.array([fn(s) for s in SEEDS], dtype=float)
    except Exception as e:
        print(f"  {name:30s} (unavailable: {type(e).__name__})"); continue
    if not np.all(np.isfinite(v)):
        print(f"  {name:30s} (unavailable)"); continue
    r = float(np.corrcoef(a, v)[0, 1])
    # EXPOSED if it moves monotonically with accuracy AND the swing is a large
    # fraction of the between-condition effect this project reports.
    verdict = "EXPOSED" if abs(r) > 0.90 else "flat"
    print(f"  {name:30s} {v[0]:8.3f} {v[1]:8.3f} {v[2]:8.3f} {np.ptp(v):8.3f} "
          f"{r:+9.2f} {verdict:>10s}")
print("\n  n=3: |r| here is descriptive, not a test. A measure marked EXPOSED is")
print("  ordered with accuracy across the widest within-condition accuracy range")
print("  available and cannot be cleared without the Test B matched control.")
