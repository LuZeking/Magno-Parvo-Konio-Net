"""Accuracy gate on the held-out bases (docs/PLAN_EXEMPLAR_2026-09-21.md, addendum C; post hoc).

  python -m src.analysis.generality.exemplar_gate --tag <tag> --feat-dir FEAT --beh-dir BEH --out-dir OUT

The E1 span (exemplar.span_folds, mode 'both') scores each fold on bases fitted on the other exemplars, but
Methods 4.8 gates it with the top-1 curve of bases fitted on all 1,200 images. This module re-runs the E1
folds with the same seeds, checks the mean curve against array A's, and for every fold basis computes top-1
on the stored ILSVRC-2012 validation subset over the 21 values of w.

Writes <tag>_gate.json.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from src.analysis.generality.exemplar import curve_on, exemplar_folds, parse_exemplars, pooled_curve
from src.analysis.generality.scoring import Scorer
from src.analysis.generality.subspace_factor import gain_features, subspace_bases
from src.analysis.generality.unit_factor import WS

N_REPS = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--feat-dir", required=True)
    ap.add_argument("--beh-dir", required=True)
    ap.add_argument("--exemplar-dir", required=True, help="array A outputs (<tag>_exemplar.json)")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(a.out_dir, exist_ok=True)

    z = np.load(os.path.join(a.feat_dir, f"{a.tag}_cue_conflict.npz"), allow_pickle=True)
    y = z["features"].astype(np.float64); W = z["W"].astype(np.float64); b = z["b"].astype(np.float64)
    shapes = np.array([str(s) for s in z["shapes"]]); textures = np.array([str(t) for t in z["textures"]])
    obj, tex = parse_exemplars(z["stems"])
    scorer = Scorer(str(z["scorer"]))
    s_idx = np.array([scorer.cat_of[s] for s in shapes]); t_idx = np.array([scorer.cat_of[t] for t in textures])

    val = np.load(os.path.join(a.feat_dir, f"{a.tag}_imagenet_val.npz"), allow_pickle=True)
    yv = val["features"].astype(np.float64); lab = val["labels"].astype(int)
    Wv, bv = val["W"].astype(np.float64), val["b"].astype(np.float64)
    allimg = json.load(open(os.path.join(a.beh_dir, f"{a.tag}_gain_accuracy_subspace.json")))
    trained_top1 = float(allimg["trained_top1"])        # the reference Methods 4.8 gates against
    trained_top1_recomputed = float((np.argmax(yv @ Wv.T + bv, 1) == lab).mean())

    prim, forc, top1 = [], [], []
    for r in range(N_REPS):                          # same RNG path as exemplar.span_folds(mode='both')
        rng = np.random.default_rng(r)
        tf = exemplar_folds(textures, tex, rng)
        of = exemplar_folds(shapes, obj, rng)
        parts = []
        for k in range(3):
            ev = np.flatnonzero((tf == k) & (of == k)); fit = np.flatnonzero((tf != k) & (of != k))
            B_S, B_T, _ = subspace_bases(y, shapes, textures, fit)
            parts.append(curve_on(y, W, b, B_S, B_T, ev, scorer, s_idx, t_idx))
            top1.append([float((np.argmax(gain_features(yv, B_S, B_T, float(w)) @ Wv.T + bv, 1) == lab).mean())
                         for w in WS])
        p, f = pooled_curve(parts)
        prim.append(p); forc.append(f)
        print(f"[gate] {a.tag} rep {r}: fold top-1 min {min(min(t) for t in top1[-3:]):.4f}  {time.time()-t0:.0f}s",
              flush=True)

    ref = json.load(open(os.path.join(a.exemplar_dir, f"{a.tag}_exemplar.json")))
    ref_curve = np.array(ref["spans"]["E1"]["primary"]["mean_curve"])
    mc = np.array(prim).mean(0)
    out = dict(tag=a.tag, ws=WS.tolist(), trained_top1=trained_top1,
               trained_top1_recomputed=trained_top1_recomputed,
               allimage_top1=allimg["top1"], allimage_gate=allimg["w_within_5pts"],
               fold_top1=top1, mean_curve=mc.tolist(), mean_curve_forced=np.array(forc).mean(0).tolist(),
               repro_max_abs_diff=float(np.abs(mc - ref_curve).max()))
    json.dump(out, open(os.path.join(a.out_dir, f"{a.tag}_gate.json"), "w"), indent=1)
    print(f"[gate] {a.tag}: reproduction |d| {out['repro_max_abs_diff']:.2e}; trained top-1 {trained_top1:.4f}; "
          f"worst fold-basis top-1 {np.min(top1):.4f}; {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
