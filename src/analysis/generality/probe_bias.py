"""Representational shape/texture bias of one frozen network by the paper's linear probe.

  python -m src.analysis.generality.probe_bias --features FEAT/<tag>_cue_conflict.npz --out-dir OUT

Methods 4.3 applied to a single feature vector (PLAN_ZOO, 2026-09-15): the penultimate
features y of the 1,200 cue-conflict images are reduced to 256 PCA components and a
multinomial logistic regression (C = 1.0) is fit under 5-fold stratified cross-validation to
the shape labels and, separately, to the texture labels; probe bias = shape accuracy minus
texture accuracy on the same folds. The estimator is imported from
src/analysis/behavior/stream_shape_texture.py, so it is the one that produced Fig. 2a.
Writes <tag>_probe_bias.json.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--pca-dim", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-splits", type=int, default=10,
                   help="accepted for pipeline symmetry; the probe uses the 5-fold CV of Methods 4.3")
    p.add_argument("--out-suffix", default="")
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    from sklearn.decomposition import PCA
    from src.analysis.behavior.stream_shape_texture import probe

    z = np.load(args.features, allow_pickle=True)
    tag = str(z["tag"])
    y = z["features"].astype(np.float64)
    shapes, textures = [str(s) for s in z["shapes"]], [str(t) for t in z["textures"]]
    n, d = y.shape
    k = int(min(args.pca_dim, n - 1, d))
    zc = PCA(n_components=k, random_state=args.seed).fit_transform(y)
    acc_s, fold_s = probe(zc, shapes, seed=args.seed)
    acc_t, fold_t = probe(zc, textures, seed=args.seed)
    chance = 1.0 / len(set(shapes))
    out = dict(tag=tag, n_images=int(n), n_units=int(d), n_components=k, chance=chance,
               shape_acc=float(acc_s), texture_acc=float(acc_t), probe_bias=float(acc_s - acc_t),
               shape_folds=[float(v) for v in fold_s], texture_folds=[float(v) for v in fold_t],
               fold_diffs=[float(a - b) for a, b in zip(fold_s, fold_t)],
               estimator="PCA-256 -> multinomial logistic regression (C = 1.0), 5-fold stratified CV (Methods 4.3)")
    os.makedirs(args.out_dir, exist_ok=True)
    dest = os.path.join(args.out_dir, f"{tag}_probe_bias{args.out_suffix}.json")
    json.dump(out, open(dest, "w"), indent=1)
    print(f"[probe_bias] {tag}: shape {acc_s:.3f} texture {acc_t:.3f} bias {acc_s - acc_t:+.3f} "
          f"(chance {chance:.3f}, k={k}, {time.time() - t0:.0f}s) -> {dest}", flush=True)


if __name__ == "__main__":
    main()
