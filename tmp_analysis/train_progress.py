"""Print epochs-done / epochs-wanted for training runs under mpk_runs.

Usage: python tmp_analysis/train_progress.py [glob-substring ...]
"""

from __future__ import annotations

import glob
import json
import os
import sys

RUNS = "/scratch/byrong/VST_data/mpk_runs"
pats = sys.argv[1:] or [""]

for pat in pats:
    for d in sorted(glob.glob(f"{RUNS}/*{pat}*")):
        if not os.path.isdir(d):
            continue
        name = os.path.basename(d)
        hist_p, cfg_p = f"{d}/history.json", f"{d}/config.json"
        if not os.path.exists(hist_p):
            print(f"{name:<34} no history.json")
            continue
        try:
            done = len(json.load(open(hist_p)))
            want = json.load(open(cfg_p)).get("epochs") if os.path.exists(cfg_p) else None
        except json.JSONDecodeError:
            # a run mid-write leaves truncated json; not an error worth stopping for
            print(f"{name:<34} history.json mid-write")
            continue
        pct = f"{100 * done / want:.0f}%" if want else "?"
        print(f"{name:<34} {done:>4}/{want} epochs  ({pct})")
