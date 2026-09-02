"""Human cue-conflict shape bias from Geirhos et al.'s published raw data.

PAPER_PLAN §20.3 and §46.5 blocked every "human-like" statement on this data
not being in our possession. It is now on disk, downloaded 2026-09-02 from

    github.com/rgeirhos/texture-vs-shape
    raw-data/style-transfer-512-nomask-experiment/

The directory name is why it was hard to find: the cue-conflict stimuli ARE
the style-transfer-512-nomask images, so there is no folder called
"cue-conflict" in that repo. 10 human observers, one session each, plus the
three CNNs from the same paper as a published cross-check on our arithmetic.

Shape bias, as Geirhos defines it: over trials where the observer answered
either the shape category or the texture category (all others discarded), the
fraction that were the SHAPE category. Trials whose shape and texture category
coincide are not cue conflicts and are excluded.

    imagename = 0001_s5n_dnn_0_airplane_00_airplane1-bicycle2.png
                                                 ^shape    ^texture

Usage: PYTHONPATH=. python tmp_analysis/geirhos_humans.py
"""
from __future__ import annotations
import csv, glob, os, re
import numpy as np

D = "/scratch/byrong/VST_data/geirhos_human_data/cue_conflict"
PAIR = re.compile(r"_([a-z]+)\d+-([a-z]+)\d+\.png$")


def shape_bias(path):
    shape_hits = texture_hits = conflicts = 0
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            m = PAIR.search(row["imagename"].strip())
            if not m:
                continue
            shape_cat, texture_cat = m.group(1), m.group(2)
            if shape_cat == texture_cat:          # not a cue conflict
                continue
            conflicts += 1
            resp = row["object_response"].strip()
            if resp == shape_cat:
                shape_hits += 1
            elif resp == texture_cat:
                texture_hits += 1
    decided = shape_hits + texture_hits
    return (shape_hits / decided if decided else float("nan"),
            decided, conflicts)


def report(pattern, label):
    vals = []
    for f in sorted(glob.glob(os.path.join(D, pattern))):
        who = re.search(r"experiment_(.+?)_session", os.path.basename(f)).group(1)
        sb, decided, conflicts = shape_bias(f)
        vals.append(sb)
        print(f"  {who:12s} shape bias {sb:.4f}   decided {decided:4d}/{conflicts}")
    v = np.array(vals)
    sd = f" ± {v.std(ddof=1):.4f}" if len(v) > 1 else ""
    rng = f", range {v.min():.3f}–{v.max():.3f}" if len(v) > 1 else ""
    print(f"  {'-' * 52}\n  {label}: {v.mean():.4f}{sd} (n={len(v)}{rng})\n")
    return v


print("HUMAN OBSERVERS (Geirhos et al., ICLR 2019)")
h = report("*subject-*.csv", "human mean")
print("CNNs FROM THE SAME PAPER — a published cross-check on this arithmetic")
report("*_alexnet_*.csv", "alexnet")
report("*_googlenet_*.csv", "googlenet")
report("*_vgg_*.csv", "vgg")

print("OUR MODELS, on the identical measure (PAPER_PLAN §18)")
for lab, v in [("diet, integrated readout", 0.675),
               ("diet, magno stream", 0.878),
               ("no diet, integrated", 0.416),
               ("no diet, parvo stream", 0.364)]:
    print(f"  {lab:26s} {v:.3f}   ({v - h.mean():+.3f} vs human)")
print(f"\n  Human shape bias is {h.mean():.3f}. Our best integrated readout is "
      f"0.675.\n  The gap is {h.mean() - 0.675:+.3f} -- DIRECTIONAL claims only "
      f"(§20.3).")
