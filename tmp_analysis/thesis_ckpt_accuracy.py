"""Top-1 positive control for the Osnabrueck thesis checkpoints.

The thesis reports val accuracy per condition (plot_classification_performance.py):
DKL B&CS 0.469, M-heavy DKL B&CS 0.524.  Reproducing those numbers is the only
check that the checkpoint, the DKL front end AND `stream1_weight` are all wired
the way the model was trained -- accuracy is the one measure that runs through
the classifier, so unlike the per-stream probe it is actually sensitive to the
fusion weight (PAPER_PLAN Sec 9.1 standing lesson: a trained-vs-untrained
comparison needs a positive control that the checkpoint really loaded).

Evaluated at the native 256 px of the local DKL h5.  The thesis datamodule
records out_size=224, but the front end cannot be run at 224: the DVD
scale-free contrast cut is built with DVDConfig(image_size=256), so it returns
a 256 px magno channel while the parvo channel stays 224, and the two streams
no longer align at the fusion.  256 px is therefore the resolution the transform
-- and hence the trained model -- actually operates at.
"""

from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader

from src.analysis.encoding.features import build_model
from src.training.standalone_trainer import (
    CONDITIONS,
    H5ImageDataset,
    StreamInputBuilder,
)

DKL_H5 = "/scratch/byrong/VST_data/VST_datasets/miniecoset_256px_dkl_uint8.h5"


@torch.no_grad()
def top1(model, builder, loader, device):
    correct = total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True).float() / 255.0
        y = y.to(device, non_blocking=True)
        x1, x2 = builder(x)
        correct += (model(x1, x2).argmax(1) == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--condition", default="dkl_bcs", choices=sorted(CONDITIONS))
    p.add_argument("--stream1-weight", type=float, default=0.5)
    p.add_argument("--split", default="val")
    p.add_argument("--dataset", default=DKL_H5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--age-months", type=int, default=48)
    p.add_argument("--out", default="results/thesis_ckpt_accuracy")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = CONDITIONS[args.condition]
    builder = StreamInputBuilder(cfg, age_months=args.age_months)

    model = build_model(checkpoint=args.checkpoint,
                        stream1_weight=args.stream1_weight, device=device)

    ds = H5ImageDataset(args.dataset, args.split)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    print(f"{args.model_name}: {len(ds)} {args.split} images | cond="
          f"{args.condition} | stream1_weight={args.stream1_weight} | "
          f"device={device}", flush=True)

    out = {"model_name": args.model_name, "checkpoint": args.checkpoint,
           "condition": args.condition, "stream1_weight": args.stream1_weight,
           "split": args.split, "n_images": len(ds), "acc": {}}
    acc = top1(model, builder, loader, device)
    out["acc"]["256"] = acc
    print(f"  top-1 @ 256px = {acc:.4f}", flush=True)

    import os
    os.makedirs(args.out, exist_ok=True)
    dst = os.path.join(args.out, f"{args.model_name}.json")
    with open(dst, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {dst}", flush=True)


if __name__ == "__main__":
    main()
