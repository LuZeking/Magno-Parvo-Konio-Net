"""Per-cell probe and cue-conflict summaries at the seed counts now available.

Prints every condition with mean +- spread across seeds on both measures, so
the conjunction 2x2 and the rgb_tsn control can be read off one table.
"""

from __future__ import annotations

import glob
import json
import os
import re

import numpy as np

PROBE = "results/shape_texture"
CC = "results/cue_conflict"

CELLS = {
    "no diet (dkl_base)": "dkl_base_seed{}_trained",
    "diet on lum (dkl_bcs)": "dkl_bcs_seed{}_trained",
    "diet on rg (swap)": "dkl_bcs_swap_seed{}_trained",
    "diet + hi-SF parvo": "dkl_bcs_parvo_highsf_seed{}_trained",
    "no split (rgb_tsn)": "rgb_tsn_seed{}_trained",
}


def probe_bias(path):
    d = json.load(open(path))
    out = {}
    for stream in ("magno", "parvo"):
        blk = d["streams"].get(stream, {})
        if "bias" in blk:
            out[stream] = blk["bias"]
        elif "shape_acc" in blk:
            out[stream] = blk["shape_acc"] - blk["texture_acc"]
    return out


def cc_bias(path):
    d = json.load(open(path))
    out = {}
    for k in ("full", "magno_only", "parvo_only"):
        blk = d["conditions"].get(k, {})
        for cand in ("shape_bias", "B_shape", "bias"):
            if cand in blk:
                out[k] = blk[cand]
                break
    return out


print(f"{'cell':<26} {'n':>2}  {'probe S1':>9} {'probe S2':>9} {'S1-S2':>8}"
      f"   {'cc S1':>7} {'cc S2':>7} {'cc full':>8}")
print("-" * 92)
for label, tpl in CELLS.items():
    pm, pp, cm, cp, cf = [], [], [], [], []
    for s in range(3):
        p = os.path.join(PROBE, tpl.format(s) + ".json")
        if os.path.exists(p):
            b = probe_bias(p)
            if "magno" in b:
                pm.append(b["magno"])
                pp.append(b["parvo"])
        c = os.path.join(CC, tpl.format(s).replace("_trained", "") + ".json")
        if os.path.exists(c):
            b = cc_bias(c)
            if "magno_only" in b:
                cm.append(b["magno_only"])
                cp.append(b["parvo_only"])
                cf.append(b.get("full", float("nan")))

    def f(v):
        return f"{np.mean(v):>8.4f}" if v else "     n/a"

    n = max(len(pm), len(cm))
    d = f"{np.mean(pm) - np.mean(pp):+8.4f}" if pm else "     n/a"
    print(f"{label:<26} {n:>2}  {f(pm)} {f(pp)} {d}   "
          f"{f(cm)[1:]} {f(cp)[1:]} {f(cf)}")

print("\nper-seed spread (probe S1-S2):")
for label, tpl in CELLS.items():
    ds = []
    for s in range(3):
        p = os.path.join(PROBE, tpl.format(s) + ".json")
        if os.path.exists(p):
            b = probe_bias(p)
            if "magno" in b:
                ds.append(b["magno"] - b["parvo"])
    if ds:
        print(f"  {label:<26} " + "  ".join(f"{d:+.4f}" for d in ds))
