"""Cue-conflict shape bias as a function of the readout fusion weight w.

The paper's §2.3 reports the integrated readout at a single operating point.
That point is not a free parameter: `TwoStreamNetwork.stream1_weight` defaults
to 0.5 and is never learned, so `full = classifier(w*y1 + (1-w)*y2)` with
w = 0.5 fixed, and `magno_only`/`parvo_only` are the same expression with one
term zeroed. An integrated value between the two stream values is therefore
close to arithmetically forced, and cannot by itself discriminate "the readout
mixes" from "the readout inherits".

Sweeping w does discriminate it, because the *components are held fixed*: y1
and y2 do not depend on w. This is the manipulation the Discussion names as the
readout account's central testable prediction -- moving the weighting should
move the behavioural rate without moving any component.

y1 and y2 are computed once and the classifier re-applied per w, so the whole
sweep costs one forward pass over the 1,200 cue-conflict images.

Scoring (16-way restriction, shape/texture decision rule, bootstrap CI over
items) is imported from src.analysis.behavior.cue_conflict, not reimplemented,
so the w = 0.5 cell must reproduce the published number exactly.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from src.analysis.encoding import dkl_frontend, features
from src.analysis.behavior.cue_conflict import (
    bootstrap_ci,
    list_cue_conflict_images,
    load_class_index,
    load_images,
    shape_bias,
    stream_in_channels,
)


@torch.no_grad()
def stream_features(model, x_m, x_p, batch_size=32, device="cpu"):
    """Pre-classifier features for each stream. Independent of w."""
    f1, f2 = [], []
    for start in range(0, x_m.shape[0], batch_size):
        xm = x_m[start:start + batch_size].to(device)
        xp = x_p[start:start + batch_size].to(device)
        y1 = model.stream1_block1(xm)
        y2 = model.stream2_block1(xp)
        if model.stream_concat:
            y1, y2 = model._mix_streams(y1, y2)
            y1 = model.stream1_block2(y1)
            y2 = model.stream2_block2(y2)
        f1.append(y1.float().cpu())
        f2.append(y2.float().cpu())
    return torch.cat(f1), torch.cat(f2)


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
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--w-step", type=float, default=0.05)
    p.add_argument("--allow-partial", action="store_true")
    args = p.parse_args()
    # --checkpoint may name a Stage-2 run; a wrong diet for it is refused
    fe = dkl_frontend.frontend_from_args(args)

    features.assert_converged(args.checkpoint, args.allow_partial)
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    class_index = load_class_index()
    items = list_cue_conflict_images()
    if not items:
        raise RuntimeError("no cue-conflict images found")
    paths = [i[0] for i in items]
    shapes = [i[1] for i in items]
    textures = [i[2] for i in items]
    print(f"[{args.model_name}] {len(items)} cue-conflict images", flush=True)

    ch_min, ch_max = dkl_frontend.compute_dkl_channel_minmax()
    model = features.build_model(
        checkpoint=args.checkpoint, stream_concat=False,
        stream1_weight=0.5, seed=args.seed, device=device,
        in_channels=stream_in_channels(args),
    )

    F1, F2 = [], []
    chunk = 128
    for start in range(0, len(paths), chunk):
        rgb = load_images(paths[start:start + chunk])
        x_m, x_p = dkl_frontend.magno_parvo_inputs(
            rgb, ch_min, ch_max, device=device, **fe)
        a, b = stream_features(model, x_m, x_p, device=device)
        F1.append(a)
        F2.append(b)
        print(f"  {min(start + chunk, len(paths))}/{len(paths)}", flush=True)
    F1 = torch.cat(F1)
    F2 = torch.cat(F2)
    print(f"  features: stream1 {tuple(F1.shape)} stream2 {tuple(F2.shape)}",
          flush=True)

    clf = model.classifier.float().cpu()
    grid = [round(w, 4) for w in np.arange(0.0, 1.0 + 1e-9, args.w_step)]
    rows = []
    with torch.no_grad():
        for w in grid:
            lg = clf(w * F1 + (1.0 - w) * F2).numpy()
            res = shape_bias(lg, shapes, textures, class_index)
            lo, hi = bootstrap_ci(lg, shapes, textures, class_index,
                                  n_boot=args.n_boot, seed=args.seed)
            res["w"] = w
            res["shape_bias_ci95"] = [lo, hi]
            res["decision_rate"] = (res["n_shape"] + res["n_texture"]) / res["n_images"]
            rows.append(res)
            print(f"  w={w:.2f}  shape_bias={res['shape_bias']:.4f} "
                  f"[{lo:.3f}, {hi:.3f}]  decided={res['decision_rate']:.3f}",
                  flush=True)

    out = {"model_name": args.model_name, "checkpoint": args.checkpoint,
           "n_boot": args.n_boot, "sweep": rows}
    path = os.path.join(args.out_dir, f"{args.model_name}_wsweep.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved -> {path}", flush=True)


if __name__ == "__main__":
    main()
