"""Accuracy gate for the fixed-head gain sweep.

  python -m src.analysis.generality.gain_accuracy --factor DIR/<tag>_unit_factor.npz \
      --val DIR/<tag>_imagenet_val.npz | DIR/<tag>_miniecoset.npz --out-dir DIR

At each w the network's own head reads g(w) * y on held-out natural images
(ImageNet validation subset for the torchvision networks, MiniEcoset validation
for ours) and top-1 accuracy is recorded, so the behavioural span can be
restricted to readouts within 5 points of the trained accuracy, as in Fig. 4b.
The split is the all-image split (no cue-conflict image is involved here).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from src.analysis.generality.unit_factor import WS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--factor", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--out-suffix", default="", help="e.g. _positive, matching the unit_factor file")
    args = p.parse_args()
    fac = np.load(args.factor, allow_pickle=True)
    z = np.load(args.val, allow_pickle=True)
    tag = str(fac["tag"])
    S = fac["S_full"].astype(bool)
    W, b = z["W"].astype(np.float64), z["b"].astype(np.float64)
    if "val_features" in z.files:
        y, lab, which = z["val_features"].astype(np.float64), z["val_y"].astype(int), "miniecoset_val"
    else:
        y, lab, which = z["features"].astype(np.float64), z["labels"].astype(int), "imagenet_val_subset"
    acc = []
    for w in WS:
        g = np.where(S, 2 * w, 2 * (1 - w))
        acc.append(float((np.argmax((y * g) @ W.T + b, 1) == lab).mean()))
    trained = acc[10]
    out = dict(tag=tag, set=which, rule=str(fac["rule"]) if "rule" in fac.files else "median", n=int(len(lab)), ws=WS.tolist(), top1=acc, trained_top1=trained,
               w_within_5pts=[float(w) for w, a in zip(WS, acc) if a >= trained - 0.05])
    os.makedirs(args.out_dir, exist_ok=True)
    dest = os.path.join(args.out_dir, f"{tag}_gain_accuracy{args.out_suffix}.json")
    json.dump(out, open(dest, "w"), indent=1)
    print(f"[gain_accuracy] {tag} on {which} (n={len(lab)}): trained {trained:.3f}; "
          f"w=0 {acc[0]:.3f}, w=1 {acc[-1]:.3f}; within 5 pts: w in [{out['w_within_5pts'][0]:.2f}, {out['w_within_5pts'][-1]:.2f}]", flush=True)


if __name__ == "__main__":
    main()
