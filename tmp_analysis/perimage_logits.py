"""Dump per-image, per-stream classifier logits on the Geirhos cue-conflict set.

Why per-STREAM and not per-w: the classifier is affine
(``AdaptiveAvgPool2d -> Flatten -> Linear``), and the fusion weights sum to one,
so the bias term cancels exactly:

    clf(w*F1 + (1-w)*F2) = W(w*F1 + (1-w)*F2) + b
                         = w*(W*F1 + b) + (1-w)*(W*F2 + b)
                         = w*clf(F1) + (1-w)*clf(F2)

(verified in 2.3 to a max deviation of 1.07e-06 on a logit scale of 0.32).
So storing L1 = clf(F1) and L2 = clf(F2) once reconstructs the network's output
at ANY readout weight by exact arithmetic -- the entire w-sweep becomes free,
offline, and reproducible without a GPU or the checkpoint.

This is what 2.6.2's kappa sweep could not do: it stored only summary rows per
w, so per-image model decisions -- the quantity a per-image comparison against
human observers needs -- were discarded.

Usage: PYTHONPATH=.:tmp_analysis python tmp_analysis/perimage_logits.py \
           --checkpoint ... --model-name ... --out-dir ...
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from src.analysis.encoding import dkl_frontend, features
from src.analysis.behavior.cue_conflict import (
    GEIRHOS_CLASSES,
    list_cue_conflict_images,
    load_class_index,
    load_images,
    shape_bias,
    stream_in_channels,
)
from fusion_sweep import stream_features   # identical feature path to 2.3/2.6.2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--no-lum-blur", action="store_true")
    p.add_argument("--no-lum-cs", action="store_true")
    p.add_argument("--transform-target", default="magno",
                   choices=["magno", "parvo"])
    p.add_argument("--parvo-highsf", action="store_true")
    p.add_argument("--age-months", type=int, default=48)
    p.add_argument("--dvd-age-months", type=float, default=None,
                   help="DVD magno diet at this age instead of blur + CS "
                        "(fs2_dvd* runs: 300)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--allow-partial", action="store_true")
    args = p.parse_args()
    # --checkpoint may name a Stage-2 run; a wrong diet for it is refused
    fe = dkl_frontend.frontend_from_args(args)

    features.assert_converged(args.checkpoint, args.allow_partial)
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    items = list_cue_conflict_images()
    paths    = [i[0] for i in items]
    shapes   = [i[1] for i in items]
    textures = [i[2] for i in items]
    stems    = [os.path.splitext(os.path.basename(q))[0] for q in paths]
    print(f"{args.model_name}: {len(paths)} cue-conflict images", flush=True)

    ch_min, ch_max = dkl_frontend.compute_dkl_channel_minmax()
    model = features.build_model(
        checkpoint=args.checkpoint, stream_concat=False, stream1_weight=0.5,
        seed=args.seed, device=device, in_channels=stream_in_channels(args))

    F1, F2 = [], []
    for start in range(0, len(paths), 128):
        rgb = load_images(paths[start:start + 128])
        x_m, x_p = dkl_frontend.magno_parvo_inputs(
            rgb, ch_min, ch_max, device=device, **fe)
        a, b = stream_features(model, x_m, x_p, device=device)
        F1.append(a); F2.append(b)
    F1, F2 = torch.cat(F1), torch.cat(F2)

    clf = model.classifier.float().cpu()
    with torch.no_grad():
        L1 = clf(F1).numpy().astype(np.float32)
        L2 = clf(F2).numpy().astype(np.float32)

        # The affinity claim is cheap to re-verify here, so verify it rather
        # than inherit it: a silent violation would invalidate every offline w.
        worst = 0.0
        for w in (0.0, 0.25, 0.5, 0.75, 1.0):
            direct = clf(w * F1 + (1.0 - w) * F2).numpy()
            recon  = w * L1 + (1.0 - w) * L2
            worst = max(worst, float(np.abs(direct - recon).max()))
        scale = float(np.abs(L1).max())
    print(f"affine reconstruction: max |direct - recon| = {worst:.3e} "
          f"(logit scale {scale:.3f})", flush=True)
    assert worst < 1e-4, f"classifier is not affine in w: {worst}"

    cls = load_class_index()
    cols = np.array([cls[c] for c in GEIRHOS_CLASSES])
    for w in (0.0, 0.5, 1.0):
        res = shape_bias(w * L1 + (1.0 - w) * L2, shapes, textures, cls)
        print(f"  w={w:.2f}  shape_bias={res['shape_bias']:.4f}  "
              f"n_other={res['n_other']}", flush=True)

    out = os.path.join(args.out_dir, f"{args.model_name}_perimage_logits.npz")
    np.savez_compressed(
        out, L1=L1, L2=L2, geirhos_cols=cols,
        stems=np.array(stems), shapes=np.array(shapes),
        textures=np.array(textures),
        geirhos_classes=np.array(GEIRHOS_CLASSES),
        model_name=args.model_name, checkpoint=args.checkpoint,
        affine_max_dev=worst)
    print(f"saved -> {out}  ({os.path.getsize(out)/1e6:.2f} MB)", flush=True)


if __name__ == "__main__":
    main()
