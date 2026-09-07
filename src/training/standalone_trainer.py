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
    # What does splitting the input across two streams buy? One stream fed
    # BOTH DKL channels, diet on luminance as usual: identical input
    # information and an identical diet, without the architectural split.
    # `rgb_tsn` answers "two streams, no split" but confounds the colour
    # space; nothing answers "one stream at all". Uses the TwoStreamNetwork
    # baseline=True path, so the classifier and every hyperparameter are
    # unchanged. Capacity is HALVED (one ResNet18 trunk, not two) -- see the
    # §64 write-up: that confound only bites if this arm LOSES.
    "dkl_bcs_single": dict(input_mode="dkl", blur=True, cs=True,
                           single_stream=True),
    # The parameter-matched version of the arm above, triggered by the letter
    # of §64.5's outcome B: the split bought shape bias, so the capacity
    # confound has to be retired before that is attributed to the split rather
    # than to having twice the trunk parameters. One ResNet34 trunk (~21.3M)
    # against two ResNet18 trunks (~22.4M) -- see §97 for the measured counts.
    #
    # `like_resnet` lives in the CONDITION, not only in the CLI flag, because
    # the run directory is named `{condition}_seed{n}` with no architecture in
    # it. Passing --like-resnet alone would write a ResNet34 run into
    # `dkl_bcs_single_seed0`, overwriting the existing ResNet18 checkpoints and
    # letting --resume load them. Binding the architecture to the condition
    # name makes the two impossible to separate (§87 lost an arm to exactly
    # this class of collision).
    "dkl_bcs_single34": dict(input_mode="dkl", blur=True, cs=True,
                             single_stream=True, like_resnet="resnet34"),
}


class H5ImageDataset(Dataset):
    """Lazily-opened HDF5 image/label dataset (worker-safe).

    `cache_ram` reads the whole split into a resident uint8 tensor once, up
    front, and serves batches from it. The dataset lives on Lustre, where the
    per-image random reads of the uncached path are latency-bound: measured
    throughput was 127 img/s at 8 workers and still only 591 at 32, against a
    model that can consume 670 img/s. Caching removes that ceiling for a
    one-off sequential read (~48 GB for the 263200-image train split, ~150 s).

    That 48 GB fits an a5000 node (95 GB) only if the job takes essentially
    the whole node, so caching trades three of the node's four GPUs for the
    speedup -- worth it because a cached run is ~16 h against ~36 h for two
    concurrent uncached ones. The larger-memory alternatives are not usable
    here: begendiv rejects this account, and g017 (2 TB, 8x B200) does not
    mount /scratch at all, so neither the dataset nor the conda env is
    visible from it.
    """

    def __init__(self, path: str, split: str, image_key: str = "data",
                 label_key: str = "labels", cache_ram: bool = False):
        self.path, self.split = path, split
        self.image_key, self.label_key = image_key, label_key
        self._f = None
        self.cache = None
        with h5py.File(path, "r") as f:
            d = f[split][image_key]
            self.length = d.shape[0]
            self.labels = np.asarray(f[split][label_key][:]).reshape(-1)
            if cache_ram:
                gb = d.size / 2**30
                print(f"  caching {split} into RAM ({gb:.1f} GB)...",
                      flush=True)
                t0 = time.time()
                self.cache = torch.empty(d.shape, dtype=torch.uint8)
                # Sequential block reads: Lustre serves these far better than
                # the random single-image reads of the uncached path.
                block = 2048
                buf = self.cache.numpy()
                for i in range(0, self.length, block):
                    d.read_direct(buf, np.s_[i:i + block],
                                  np.s_[i:i + block])
                # Deliberately NOT share_memory_(): that would copy all 48 GB
                # into /dev/shm, which is typically far smaller. Forked
                # DataLoader workers already see these pages copy-on-write,
                # and the cache is only ever read, so no copy is triggered.
                print(f"  cached in {time.time() - t0:.0f}s "
                      f"({gb / (time.time() - t0):.2f} GB/s)", flush=True)

    def __len__(self) -> int:
        return self.length

    def _file(self):
        if self._f is None:  # opened per worker process
            self._f = h5py.File(self.path, "r", swmr=True)
        return self._f

    def __getitem__(self, idx: int):
        if self.cache is not None:
            return self.cache[idx].permute(2, 0, 1), int(self.labels[idx])
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
        self.single_stream = cfg.get("single_stream", False)

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
        if self.single_stream:
            # One stream, both channels stacked. x_2=None selects the
            # TwoStreamNetwork baseline forward path.
            return torch.cat([lum, rg], dim=1), None
        if self.parvo_highsf:
            # parvo = high-acuity luminance + red-green (untransformed lum)
            return lum, torch.cat([x[:, 0:1], x[:, 1:2]], dim=1)
        return lum, rg


def in_channels_for(cfg: dict) -> tuple[int, int]:
    if cfg.get("single_stream"):
        # Stream 2 does not exist; 0 records that in config.json rather than
        # leaving a misleading 1 for a stream that was never built.
        return 2, 0
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
def evaluate(model, builder, loader, device, max_batches=None, amp=False):
    model.eval()
    correct = total = 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(device, non_blocking=True).float() / 255.0
        y = y.to(device, non_blocking=True)
        x1, x2 = builder(x)
        if amp:
            x1 = x1.contiguous(memory_format=torch.channels_last)
            if x2 is not None:
                x2 = x2.contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
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
    # Default None, not "resnet18": the condition supplies the architecture
    # for arms that pin one (dkl_bcs_single34), and a truthy default here
    # would silently win over it. Falls back to resnet18 when neither sets it.
    p.add_argument("--like-resnet", default=None)
    p.add_argument("--age-months", type=int, default=48)
    p.add_argument("--max-steps", type=int, default=None,
                   help="stop early (smoke test)")
    p.add_argument("--val-batches", type=int, default=None)
    p.add_argument("--ckpt-every", type=int, default=10)
    # Measured on an A5000 (tmp_analysis/profile_opt.py): bf16 autocast plus
    # channels_last takes the model's fwd/bwd from 314 to 670 img/s and peak
    # memory from 20.5 to 11.2 GB at bs=256, cutting a 150-epoch run from
    # ~35 h to ~16 h. Adam keeps fp32 master weights, so this changes speed,
    # not the optimization. --no-amp restores the fp32 path.
    p.add_argument("--no-amp", action="store_true",
                   help="disable bf16 autocast + channels_last")
    p.add_argument("--resume", action="store_true",
                   help="continue from checkpoint_last.pt if present")
    p.add_argument("--cache-ram", action="store_true",
                   help="load the whole dataset into RAM (~48 GB train); "
                        "big-memory nodes only")
    args = p.parse_args()

    cfg = CONDITIONS[args.condition]
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Explicit --like-resnet wins (one-off probes), then the condition's own
    # architecture, then the historical default. Resolved before the run
    # directory is created so config.json records what was actually built.
    arch = args.like_resnet or cfg.get("like_resnet", "resnet18")

    run = f"{args.condition}_seed{args.seed}"
    out_dir = os.path.join(args.out_dir, run)
    os.makedirs(out_dir, exist_ok=True)

    train_ds = H5ImageDataset(args.dataset, "train",
                              cache_ram=args.cache_ram)
    val_ds = H5ImageDataset(args.dataset, "val",
                            cache_ram=args.cache_ram)
    # With the dataset already resident, worker processes buy nothing: there
    # is no I/O left to overlap, only a ~50 MB per-batch memcpy against a
    # ~380 ms GPU step. They also cost a great deal -- forking 24 children
    # off a 48 GB parent fails outright with OSError(12) under the job's
    # memory cgroup, even though the pages are copy-on-write.
    n_workers = args.num_workers
    if args.cache_ram and n_workers > 0:
        print(f"  cache_ram: forcing num_workers {n_workers} -> 0", flush=True)
        n_workers = 0
    common = dict(num_workers=n_workers, pin_memory=True,
                  persistent_workers=n_workers > 0)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, **common)

    c1, c2 = in_channels_for(cfg)
    single = cfg.get("single_stream", False)
    model = TwoStreamNetwork(
        num_classes=args.num_classes,
        like_resnet=arch,
        seperate_from_layer=0,
        # baseline=True builds ONE trunk (stream2_block1 = None) and takes the
        # x_2=None forward path; combined_color makes its stem 2-channel so the
        # single stream can receive both DKL channels.
        baseline=single,
        combined_color=single,
        dvd_strategy="dkl" if cfg["input_mode"] == "dkl" else "dvd",
        stream_concat=cfg.get("stream_concat", False),
        stream1_weight=cfg.get("stream1_weight", 0.5),
        legacy_asymmetric_mixing=False,  # corrected symmetric 75/25
    )
    patch_stream_input_channels(model, c1, c2)
    model = model.to(device)

    amp = (not args.no_amp) and device == "cuda" and torch.cuda.is_bf16_supported()
    if amp:
        model = model.to(memory_format=torch.channels_last)

    builder = StreamInputBuilder(cfg, age_months=args.age_months)
    augment = build_augmentations(cfg["input_mode"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss()

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        # `like_resnet` in vars(args) is None whenever the condition supplied
        # the architecture, so record what was actually built rather than
        # leaving the resolved value inferable only from condition_cfg.
        json.dump({**vars(args), "condition_cfg": cfg,
                   "like_resnet_resolved": arch,
                   "n_trunk_params": sum(p.numel() for p in model.parameters()),
                   "in_channels": [c1, c2]}, f, indent=2)

    print(f"[{run}] train={len(train_ds)} val={len(val_ds)} device={device} "
          f"arch={arch} params={sum(p.numel() for p in model.parameters()):,} "
          f"in_channels={c1},{c2} amp={amp} workers={n_workers}",
          flush=True)

    # Resume: these runs are ~3 days each, so a node failure or a
    # walltime kill must not cost the whole run. checkpoint_last.pt carries
    # the optimizer state too -- restoring weights alone would silently
    # restart Adam's moments and change the optimization.
    history, step, start_epoch = [], 0, 0
    last_path = os.path.join(out_dir, "checkpoint_last.pt")
    if args.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["state_dict"])
        optimizer.load_state_dict(ck["optimizer"])
        start_epoch = ck["epoch"] + 1
        history = ck.get("history", [])
        step = ck.get("step", 0)
        print(f"[{run}] resumed from epoch {ck['epoch']} "
              f"-> starting at {start_epoch}", flush=True)

    for epoch in range(start_epoch, args.epochs):
        t0, running, seen = time.time(), 0.0, 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True).float() / 255.0
            y = y.to(device, non_blocking=True)
            with torch.no_grad():
                x = augment(x)
            x1, x2 = builder(x)
            if amp:
                x1 = x1.contiguous(memory_format=torch.channels_last)
                if x2 is not None:
                    x2 = x2.contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                loss = criterion(model(x1, x2), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += loss.item() * y.numel()
            seen += y.numel()
            step += 1
            if args.max_steps is not None and step >= args.max_steps:
                break
        val_acc = evaluate(model, builder, val_loader, device,
                           max_batches=args.val_batches, amp=amp)
        rec = {"epoch": epoch, "train_loss": running / max(seen, 1),
               "val_top1": val_acc, "sec": round(time.time() - t0, 1)}
        history.append(rec)
        print(f"[{run}] {rec}", flush=True)
        with open(os.path.join(out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)
        # Resume point, rewritten every epoch: cheap next to an epoch, and
        # it bounds the loss from an unexpected kill to one epoch rather
        # than to ckpt_every. Written to a temp path first so a kill during
        # the write cannot leave a truncated file as the only resume point.
        tmp_path = last_path + ".tmp"
        torch.save({"epoch": epoch, "step": step, "history": history,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": {**cfg, "in_channels": [c1, c2]}}, tmp_path)
        os.replace(tmp_path, last_path)
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
