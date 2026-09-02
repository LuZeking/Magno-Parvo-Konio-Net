"""Seed replication of the paper's LEAD result: the 2x2 conjunction (§2.1).

After §58 seed-replicated the parity claim and §59 the human comparison, §2.1 --
the headline -- was the least-replicated section in the paper, citing "n=2 seeds
agreeing to 0.017". All three seeds exist on disk for both readouts, so this
costs nothing and removes the most obvious reviewer target.

The 2x2 cells, and which stream of which model reads each one out:

                       | no transform            | with transform
  chromatic (L-M)      | diet     -> parvo       | swap  -> parvo
  achromatic (lum)     | no diet  -> magno       | diet  -> magno

`swap` applies the transform to the chromatic channel; in `parvo_highsf` the
second stream gets an untransformed high-acuity achromatic carrier, which is the
separate capacity control of §2.1 and is reported alongside, not in the 2x2.

Probe bias is read from results/shape_texture (linear probe, every image
classified). Behaviour is read from results/cue_conflict (16-way forced choice,
scored only on images decided as shape or texture -- the §51.3 denominator
caveat applies to the behavioural row and not to the probe).
"""
from __future__ import annotations
import json
import numpy as np

SEEDS = (0, 1, 2)

# (cell label, model stem, stream)
CELLS = [
    ("chromatic, no transform", "dkl_bcs_seed{}",  "parvo"),
    ("chromatic, transform",    "dkl_bcs_swap_seed{}", "parvo"),
    ("achromatic, no transform","dkl_base_seed{}", "magno"),
    ("achromatic, TRANSFORM",   "dkl_bcs_seed{}",  "magno"),
]
EXTRA = ("capacity control (parvo_highsf, stream 2)",
         "dkl_bcs_parvo_highsf_seed{}", "parvo")

READOUT = {"magno": "magno_only", "parvo": "parvo_only"}


def probe(model, stream):
    d = json.load(open(f"results/shape_texture/{model}_trained.json"))
    return d["streams"][stream]["bias"]


def behaviour(model, stream):
    d = json.load(open(f"results/cue_conflict/{model}.json"))
    c = d["conditions"][READOUT[stream]]
    return c["shape_bias"], c["n_shape"] + c["n_texture"]


print("=" * 88)
print("SEED REPLICATION OF THE 2x2 CONJUNCTION (§2.1) -- n=3 training seeds")
print("=" * 88)
print(f"\n{'cell':34s} {'probe bias (3 seeds)':>26s} {'mean':>8s} {'range':>7s}")
pm = {}
for label, stem, stream in CELLS + [EXTRA]:
    v = np.array([probe(stem.format(s), stream) for s in SEEDS])
    pm[label] = v
    print(f"  {label:32s} {np.array2string(v, precision=3, floatmode='fixed'):>26s} "
          f"{v.mean():8.3f} {np.ptp(v):7.3f}")

print(f"\n{'cell':34s} {'behaviour (3 seeds)':>26s} {'mean':>8s} {'range':>7s} {'n dec':>7s}")
bm = {}
for label, stem, stream in CELLS + [EXTRA]:
    vals, ns = zip(*[behaviour(stem.format(s), stream) for s in SEEDS])
    v = np.array(vals); bm[label] = v
    print(f"  {label:32s} {np.array2string(v, precision=3, floatmode='fixed'):>26s} "
          f"{v.mean():8.3f} {np.ptp(v):7.3f} {int(np.mean(ns)):7d}")

print("\n" + "=" * 88)
print("THE CLAIM: only the conjunction crosses the shape/texture boundary (bias > 0)")
print("=" * 88)
for name, M in (("probe", pm), ("behaviour", bm)):
    thr = 0.0 if name == "probe" else 0.5      # behaviour is a fraction, 0.5 is the flip
    conj = M["achromatic, TRANSFORM"]
    others = np.concatenate([M[k] for k, _, _ in CELLS if k != "achromatic, TRANSFORM"])
    ok = conj.min() > thr and others.max() < thr
    print(f"  {name:10s} conjunction {conj.min():.3f}-{conj.max():.3f} "
          f"vs all other cells {others.min():.3f}-{others.max():.3f} "
          f"(boundary {thr})  -> {'HOLDS in 3/3 seeds' if ok else 'FAILS'}")
    print(f"             separation margin = {conj.min() - others.max():+.3f}")
