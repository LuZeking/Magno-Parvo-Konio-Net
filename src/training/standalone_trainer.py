"""Self-contained TSN trainer (no lightning_kietzmannlab dependency).

The thesis pipeline inherits from the lab's private `lightning_kietzmannlab`
package (TrainingPipeline / H5dataset / BaseModule), which is not obtainable
here. This module reimplements exactly the parts the paper needs, reusing the
in-repo TwoStreamNetwork and the verified DVD transforms, so models can be
retrained locally with:
  * a single dataset version (removes the thesis's uint8/float64 confound)
  * multiple seeds
  * the corrected symmetric cross-stream mixing
  * per-stream input configuration (incl. the high-SF-luminance parvo fix and
    the channel-swap / blur-only controls)

Training defaults replicate the thesis: 150 epochs, Adam lr 1e-4 (beta1 0.9),
batch 256, cross-entropy, and the DVD-B augmentation set (random grayscale
skipped for DKL inputs, as in the original).

Run (single condition):
    python -m src.training.standalone_trainer \
        --dataset /scratch/.../miniecoset_256px_dkl_uint8.h5 \
        --condition dkl_bcs --seed 0 --out-dir /scratch/.../runs
Smoke test:
    ... --max-steps 4 --epochs 1 --batch-size 8 --num-workers 0
"""

from __future__ import annotations

import argparse
import json
import os
import time

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.models.pytorch_models import TwoStreamNetwork

# --------------------------------------------------------------------------
# Conditions: which channels feed which stream, and what is applied to them.
#   input_mode "dkl"  -> stream1 = lum, stream2 = rg (1 channel each)
#   input_mode "rgb"  -> both streams get the 3-channel image
#   parvo_highsf      -> stream2 receives lum+rg stacked (2 channels): the
#                        biologically corrected parvo input (high-acuity
#                        luminance + red-green), Track B item B2
# --------------------------------------------------------------------------
CONDITIONS = {
    # thesis replications
    "rgb_tsn":    dict(input_mode="rgb", blur=False, cs=False),
    "dkl_base":   dict(input_mode="dkl", blur=False, cs=False),
    "dkl_b":      dict(input_mode="dkl", blur=True,  cs=False),
    "dkl_cs":     dict(input_mode="dkl", blur=False, cs=True),
    "dkl_bcs":    dict(input_mode="dkl", blur=True,  cs=True),
    "dkl_bcs_mheavy": dict(input_mode="dkl", blur=True, cs=True,
                           stream1_weight=0.9),
    "dkl_bcs_inter":  dict(input_mode="dkl", blur=True, cs=True,
                           stream_concat=True),
    # controls the paper needs
    "dkl_bcs_swap":   dict(input_mode="dkl", blur=True, cs=True,
                           transform_target="parvo"),
    "rgb_tsn_blur":   dict(input_mode="rgb", blur=True, cs=True),
    # Track B: corrected parvo input (high-SF luminance + L-M)
    "dkl_bcs_parvo_highsf": dict(input_mode="dkl", blur=True, cs=True,
                                 parvo_highsf=True),
}


class H5ImageDataset(Dataset):
    """Lazily-opened HDF5 image/label dataset (worker-safe)."""

    def __init__(self, path: str, split: str, image_key: str = "data",
                 label_key: str = "labels"):
        self.path, self.split = path, split
        self.image_key, self.label_key = image_key, label_key
        self._f = None
        with h5py.File(path, "r") as f:
            self.length = f[split][image_key].shape[0]
            self.labels = np.asarray(f[split][label_key][:]).reshape(-1)

    def __len__(self) -> int:
        return self.length

    def _file(self):
        if self._f is None:  # opened per worker process
            self._f = h5py.File(self.path, "r", swmr=True)
        return self._f

    def __getitem__(self, idx: int):
        img = self._file()[self.split][self.image_key][idx]  # (H, W, C) uint8
        x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)
        return x, int(self.labels[idx])


def build_augmentations(input_mode: str):
    """Thesis augmentation set (DVD-B); grayscale skipped for DKL inputs."""
    import kornia.augmentation as K

    aug = [
        K.RandomHorizontalFlip(p=0.25),
        K.RandomRotation(degrees=15.0, p=0.25),
    ]
    if input_mode == "rgb":
        aug.append(K.RandomGrayscale(p=0.5))
    aug += [
        K.ColorJitter(brightness=0.4, p=0.5),
        K.RandomEqualize(p=0.5),
        K.RandomPerspective(0.3, p=0.5),
        K.RandomSharpness(0.5, p=0.5),
        K.RandomGaussianBlur((3, 3), (0.1, 2.0), p=0.5),
    ]
    return nn.Sequential(*aug)


class StreamInputBuilder:
    """Turns a raw batch into per-stream inputs for a given condition."""

    def __init__(self, cfg: dict, age_months: int = 48):
        self.cfg = cfg
        self.age_months = age_months
        self.input_mode = cfg["input_mode"]
        self.blur, self.cs = cfg["blur"], cfg["cs"]
        self.target = cfg.get("transform_target", "magno")
        self.parvo_highsf = cfg.get("parvo_highsf", False)

    def __call__(self, x: torch.Tensor):
        from src.analysis.encoding.dkl_frontend import (
            apply_gaussian_blur,
            apply_scale_free_contrast,
        )

        if self.input_mode == "rgb":
            x1 = x2 = x
            if self.blur:
                x1 = apply_gaussian_blur(x1)
            if self.cs:
                x1 = apply_scale_free_contrast(x1, self.age_months)
            return x1, x2

        lum, rg = x[:, 0:1], x[:, 1:2]
        tgt = lum if self.target == "magno" else rg
        if self.blur:
            tgt = apply_gaussian_blur(tgt)
        if self.cs:
            tgt = apply_scale_free_contrast(tgt, self.age_months)
        if self.target == "magno":
            lum = tgt
        else:
            rg = tgt
        if self.parvo_highsf:
            # parvo = high-acuity luminance + red-green (untransformed lum)
            return lum, torch.cat([x[:, 0:1], x[:, 1:2]], dim=1)
        return lum, rg


def in_channels_for(cfg: dict) -> tuple[int, int]:
    if cfg["input_mode"] == "rgb":
        return 3, 3
    return 1, (2 if cfg.get("parvo_highsf") else 1)


def patch_stream_input_channels(model: TwoStreamNetwork, c1: int, c2: int):
    """Set each stream's stem conv to its required channel count."""
    for block, c in ((model.stream1_block1, c1), (model.stream2_block1, c2)):
        if block is None:
            continue
        old = block[0]
        if isinstance(old, nn.Conv2d) and old.in_channels != c:
            block[0] = nn.Conv2d(
                c, old.out_channels, kernel_size=old.kernel_size,
                stride=old.stride, padding=old.padding, bias=old.bias is not None,
            )


@torch.no_grad()
def evaluate(model, builder, loader, device, max_batches=None):
    model.eval()
    correct = total = 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(device, non_blocking=True).float() / 255.0
        y = y.to(device, non_blocking=True)
        x1, x2 = builder(x)
        logits = model(x1, x2)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
    model.train()
    return correct / max(total, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--condition", required=True, choices=sorted(CONDITIONS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--num-classes", type=int, default=112)
    p.add_argument("--like-resnet", default="resnet18")
    p.add_argument("--age-months", type=int, default=48)
    p.add_argument("--max-steps", type=int, default=None,
                   help="stop early (smoke test)")
    p.add_argument("--val-batches", type=int, default=None)
    p.add_argument("--ckpt-every", type=int, default=10)
    args = p.parse_args()

    cfg = CONDITIONS[args.condition]
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run = f"{args.condition}_seed{args.seed}"
    out_dir = os.path.join(args.out_dir, run)
    os.makedirs(out_dir, exist_ok=True)

    train_ds = H5ImageDataset(args.dataset, "train")
    val_ds = H5ImageDataset(args.dataset, "val")
    common = dict(num_workers=args.num_workers, pin_memory=True,
                  persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, **common)

    c1, c2 = in_channels_for(cfg)
    model = TwoStreamNetwork(
        num_classes=args.num_classes,
        like_resnet=args.like_resnet,
        seperate_from_layer=0,
        baseline=False,
        dvd_strategy="dkl" if cfg["input_mode"] == "dkl" else "dvd",
        stream_concat=cfg.get("stream_concat", False),
        stream1_weight=cfg.get("stream1_weight", 0.5),
        legacy_asymmetric_mixing=False,  # corrected symmetric 75/25
    )
    patch_stream_input_channels(model, c1, c2)
    model = model.to(device)

    builder = StreamInputBuilder(cfg, age_months=args.age_months)
    augment = build_augmentations(cfg["input_mode"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss()

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({**vars(args), "condition_cfg": cfg,
                   "in_channels": [c1, c2]}, f, indent=2)

    print(f"[{run}] train={len(train_ds)} val={len(val_ds)} device={device} "
          f"in_channels={c1},{c2}", flush=True)

    history, step = [], 0
    for epoch in range(args.epochs):
        t0, running, seen = time.time(), 0.0, 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True).float() / 255.0
            y = y.to(device, non_blocking=True)
            with torch.no_grad():
                x = augment(x)
            x1, x2 = builder(x)
            logits = model(x1, x2)
            loss = criterion(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += loss.item() * y.numel()
            seen += y.numel()
            step += 1
            if args.max_steps is not None and step >= args.max_steps:
                break
        val_acc = evaluate(model, builder, val_loader, device,
                           max_batches=args.val_batches)
        rec = {"epoch": epoch, "train_loss": running / max(seen, 1),
               "val_top1": val_acc, "sec": round(time.time() - t0, 1)}
        history.append(rec)
        print(f"[{run}] {rec}", flush=True)
        with open(os.path.join(out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)
        if (epoch + 1) % args.ckpt_every == 0 or epoch + 1 == args.epochs:
            torch.save(
                {"epoch": epoch, "state_dict": model.state_dict(),
                 "config": {**cfg, "in_channels": [c1, c2]}},
                os.path.join(out_dir, f"checkpoint_{epoch + 1:03d}.pt"),
            )
        if args.max_steps is not None and step >= args.max_steps:
            break

    torch.save({"state_dict": model.state_dict(),
                "config": {**cfg, "in_channels": [c1, c2]}},
               os.path.join(out_dir, "checkpoint_final.pt"))
    print(f"[{run}] done -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
