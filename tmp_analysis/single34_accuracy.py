"""§64.6's pre-declared statistics for the single-stream capacity control.

Protocol fixed BEFORE the resnet34 run existed (PLAN §64.6):
  - assert 150 epochs
  - sd(val_top1) over the LAST 20 epochs      (late-training stability)
  - best-epoch AND last-epoch accuracy, both
  - three seeds each side
  - complete separation + raw range, NOT a p-value (3v3 floors at .10)

"Best epoch" means best among SAVED checkpoints, not the argmax of the val
curve. This is not a detail: Limitation 1 records the paper reporting a 37%
then a 16.5% accuracy cost, the second taken from per-epoch validation maxima
at epochs whose weights were never saved (§71). Only a checkpoint that exists
can be re-evaluated. checkpoint_{epoch+1:03d}.pt -> history epoch NNN-1.
"""
import json
import os
import sys

import numpy as np

RUNS = "/scratch/byrong/VST_data/mpk_runs"
SAVED = [24, 49, 74, 99, 124, 149]          # history epochs with a checkpoint


def stats(run):
    p = os.path.join(RUNS, run, "history.json")
    if not os.path.exists(p):
        return None
    h = json.load(open(p))
    v = np.array([e["val_top1"] for e in h])
    ep = [e["epoch"] for e in h]
    saved = [e for e in SAVED if e < len(v)]
    bi = max(saved, key=lambda e: v[e])
    return dict(n_epochs=len(v), last_epoch=ep[-1], last=float(v[-1]),
                best_saved_epoch=bi, best_saved=float(v[bi]),
                curve_argmax_epoch=int(v.argmax()), curve_max=float(v.max()),
                sd_last20=float(v[-20:].std(ddof=1)))


def show(title, runs):
    print(f"\n=== {title} ===")
    rows = []
    for r in runs:
        s = stats(r)
        if s is None:
            print(f"  {r:28s} NOT PRESENT")
            continue
        rows.append(s)
        print(f"  {r:28s} epochs={s['n_epochs']:3d} "
              f"best_saved={s['best_saved']:.4f}@{s['best_saved_epoch']:3d} "
              f"last={s['last']:.4f} sd20={s['sd_last20']:.4f} "
              f"(curve_max={s['curve_max']:.4f}@{s['curve_argmax_epoch']})")
    return rows


A = show("ResNet18 single-stream (dkl_bcs_single)",
         [f"dkl_bcs_single_seed{s}" for s in (0, 1, 2)])
B = show("ResNet34 single-stream (dkl_bcs_single34) -- IN PROGRESS",
         [f"dkl_bcs_single34_seed{s}" for s in (0, 1, 2)])
C = show("Two-stream (dkl_bcs)", [f"dkl_bcs_seed{s}" for s in (0, 1, 2)])

for name, key in (("best_saved", "best_saved"), ("last", "last"),
                  ("sd_last20", "sd_last20")):
    print(f"\n--- {name}: ranges ---")
    for lbl, rows in (("r18_single", A), ("r34_single", B), ("two_stream", C)):
        if rows:
            vals = [r[key] for r in rows]
            print(f"  {lbl:11s} n={len(vals)} "
                  f"[{min(vals):.4f}, {max(vals):.4f}]")
