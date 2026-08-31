"""Leg (b): aggregate the cue-conflict shape-bias results.

Reports the per-stream logit decomposition (full / magno_only / parvo_only)
for every model in the leg (a)/(c) 2x2, so the three legs can be read on one
model set.

Usage: PYTHONPATH=. python tmp_analysis/legb_table.py
"""

from __future__ import annotations

import json
import os

import numpy as np

CELLS = {
    "untrained_noblur":   ("untrained, no diet", "untrained_nodiet"),
    "untrained_blur_cs":  ("untrained, with diet", "untrained_diet"),
    "dkl_base_seed0":     ("trained, no diet", "nodiet"),
    "dkl_base_seed1":     ("trained, no diet", "nodiet"),
    "dkl_base_seed2":     ("trained, no diet", "nodiet"),
    "dkl_bcs_seed0":      ("trained, with diet", "diet"),
    "dkl_bcs_seed1":      ("trained, with diet", "diet"),
    "dkl_bcs_seed2":      ("trained, with diet", "diet"),
    "dkl_bcs_swap_seed0": ("trained, swapped diet", "swap"),
    "dkl_bcs_swap_seed1": ("trained, swapped diet", "swap"),
    "thesis_dvd_trans":   ("thesis, with diet", "diet_thesis"),
    "thesis_s1_90":       ("thesis, s1=0.9", "s1_90"),
}
COND = ["full", "magno_only", "parvo_only"]

hdr = " ".join(f"{c:>22}" for c in COND)
print(f"{'model':<20} {'cell':<22} {hdr}   16cls-acc")
rows: dict[str, list[list[float]]] = {}
for name, (lab, cell) in CELLS.items():
    fp = f"results/cue_conflict/{name}.json"
    if not os.path.exists(fp):
        print(f"{name:<20} MISSING")
        continue
    d = json.load(open(fp))["conditions"]
    txt = []
    for c in COND:
        b = d[c]["shape_bias"]
        lo, hi = d[c]["shape_bias_ci95"]
        txt.append(f"{b:.3f} [{lo:.3f},{hi:.3f}]")
    acc = d["full"]["shape_accuracy"] + d["full"]["texture_accuracy"]
    print(f"{name:<20} {lab:<22} " + " ".join(f"{s:>22}" for s in txt)
          + f"   {acc:.3f}")
    rows.setdefault(cell, []).append([d[c]["shape_bias"] for c in COND])

print("\n--- cell means ---")
print(f"{'cell':<18} {'full':>7} {'magno':>7} {'parvo':>7}  {'magno-parvo':>12}"
      f"  {'full-max(m,p)':>14}  n")
for cell, v in rows.items():
    m = np.array(v).mean(0)
    print(f"{cell:<18} {m[0]:>7.3f} {m[1]:>7.3f} {m[2]:>7.3f}  {m[1]-m[2]:>12.3f}"
          f"  {m[0]-max(m[1], m[2]):>14.3f}  {len(v)}")
