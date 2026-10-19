"""Accuracy-gated span of the frozen affine readout sweep (PAPER_final Results 2.1:
"moves the cue-conflict rate ... within five percentage points of the trained
natural-image top-1"), for any run whose stream features are cached by
tmp_analysis/nonlinear_readout_sweep.py's extraction.

For each w the trained classifier is applied unchanged to w*F1 + (1-w)*F2 of the
cached MiniEcoset validation features (top-1) and cue-conflict features (shape
bias, src.analysis.behavior.cue_conflict.shape_bias). The cue-conflict curve must
reproduce results/fusion_sweep/<tag>_wsweep.json; the gated span is max - min of
the shape bias over the w whose top-1 is within 0.05 of the top-1 at w = 0.5.

    PYTHONPATH=. python tmp_analysis/gated_affine_span.py --tags dkl_bcs_seed0_best ... --out results/figures_v2/gated_affine_span.json
"""
import argparse
import json
import os
from types import SimpleNamespace

import numpy as np
import torch

from src.analysis.behavior.cue_conflict import load_class_index, shape_bias, stream_in_channels
from src.analysis.encoding import features

NL_FEAT = "/scratch/byrong/TSN/nonlinear_readout/features/2026-09-10"
GATE = 0.05


def one(tag, class_index):
    z = np.load(os.path.join(NL_FEAT, f"{tag}_features.npz"), allow_pickle=True)
    ckpt = str(z["checkpoint"])
    model = features.build_model(checkpoint=ckpt, stream_concat=False, stream1_weight=0.5, seed=0, device="cpu",
                                 in_channels=stream_in_channels(SimpleNamespace(parvo_highsf=False)))
    # classifier = [AdaptiveAvgPool2d, Flatten, Linear] on the stream maps; the cached
    # features are already pooled and pooling commutes with the w-mix, so the Linear alone
    clf = model.classifier[-1].float().cpu().eval()
    vF1, vF2 = torch.from_numpy(z["val_F1"].astype(np.float32)), torch.from_numpy(z["val_F2"].astype(np.float32))
    vy = torch.from_numpy(z["val_y"].astype(np.int64))
    cF1, cF2 = torch.from_numpy(z["cc_F1"]), torch.from_numpy(z["cc_F2"])
    shapes, textures = list(z["cc_shapes"]), list(z["cc_textures"])
    grid = [round(w, 4) for w in np.arange(0.0, 1.0 + 1e-9, 0.05)]
    top1, sb = [], []
    with torch.no_grad():
        for w in grid:
            top1.append(float((clf(w * vF1 + (1 - w) * vF2).argmax(1) == vy).float().mean()))
            sb.append(shape_bias(clf(w * cF1 + (1 - w) * cF2).numpy(), shapes, textures, class_index)["shape_bias"])
    top1, sb = np.array(top1), np.array(sb)
    ref = sorted(json.load(open(f"results/fusion_sweep/{tag}_wsweep.json"))["sweep"], key=lambda r: r["w"])
    dev = float(np.max(np.abs(sb - np.array([r["shape_bias"] for r in ref]))))
    i05 = grid.index(0.5)
    keep = top1 >= top1[i05] - GATE
    ws = np.array(grid)[keep]
    return dict(checkpoint=ckpt, grid=grid, top1=top1.tolist(), shape_bias=sb.tolist(), sweep_reproduction_max_dev=dev,
                top1_at_half=float(top1[i05]), gated_w=[float(ws.min()), float(ws.max())],
                gated_lo=float(sb[keep].min()), gated_hi=float(sb[keep].max()), gated_span=float(sb[keep].max() - sb[keep].min()),
                ungated_span=float(sb.max() - sb.min()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ci = load_class_index()
    out = {}
    for tag in args.tags:
        r = one(tag, ci)
        out[tag] = r
        print(f"{tag:40s} top1@.5 {r['top1_at_half']:.3f}  gated w {r['gated_w'][0]:.2f}-{r['gated_w'][1]:.2f}  "
              f"rate {r['gated_lo']:.3f}->{r['gated_hi']:.3f}  gated span {r['gated_span']:.3f}  "
              f"(ungated {r['ungated_span']:.3f}; sweep reproduction max dev {r['sweep_reproduction_max_dev']:.1e})", flush=True)
    json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
