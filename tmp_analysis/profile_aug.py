"""Is the kornia augmentation pipeline the remaining bottleneck?

With the dataset resident in RAM and bf16+channels_last enabled, the full
step still runs at 311 img/s while the model alone manages 670. Something
between the two costs ~1.7 ms/image. The augmentation stack is the only
substantial GPU work left, so time each transform separately.

This is a measurement, not a proposal to change the augmentations: they are
part of the thesis replication, so any change to them is a change to the
science and has to be argued separately.
"""

from __future__ import annotations

import argparse
import time

import torch
import kornia.augmentation as K

from src.training.standalone_trainer import (
    CONDITIONS, StreamInputBuilder, build_augmentations,
)

p = argparse.ArgumentParser()
p.add_argument("--batch-size", type=int, default=256)
p.add_argument("--steps", type=int, default=20)
a = p.parse_args()

dev = "cuda"
bs = a.batch_size
x0 = torch.rand(bs, 3, 256, 256, device=dev)


def timeit(fn, n=None):
    n = n or a.steps
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    dt = time.time() - t0
    return n * bs / dt, dt / n * 1000


print(f"batch={bs} on {torch.cuda.get_device_name(0)}")
print(f"\n{'transform':<34}{'img/s':>10}{'ms/step':>10}")
each = [
    ("RandomHorizontalFlip(p=.25)", K.RandomHorizontalFlip(p=0.25)),
    ("RandomRotation(15, p=.25)", K.RandomRotation(degrees=15.0, p=0.25)),
    ("RandomGrayscale(p=.5)", K.RandomGrayscale(p=0.5)),
    ("ColorJitter(brightness=.4, p=.5)", K.ColorJitter(brightness=0.4, p=0.5)),
    ("RandomEqualize(p=.5)", K.RandomEqualize(p=0.5)),
    ("RandomPerspective(.3, p=.5)", K.RandomPerspective(0.3, p=0.5)),
    ("RandomSharpness(.5, p=.5)", K.RandomSharpness(0.5, p=0.5)),
    ("RandomGaussianBlur(3, p=.5)",
     K.RandomGaussianBlur((3, 3), (0.1, 2.0), p=0.5)),
]
for name, t in each:
    # Wrap in Sequential to match how the trainer builds and moves these;
    # instantiated bare, RandomPerspective trips a kornia dtype bug.
    t = torch.nn.Sequential(t).to(dev)
    try:
        with torch.no_grad():
            r, ms = timeit(lambda t=t: t(x0))
        print(f"{name:<34}{r:>10.0f}{ms:>10.1f}")
    except Exception as e:
        print(f"{name:<34}{'FAILED':>10}   {type(e).__name__}: {e}")

for mode in ("dkl", "rgb"):
    aug = build_augmentations(mode).to(dev)
    with torch.no_grad():
        r, ms = timeit(lambda: aug(x0))
    print(f"{'FULL STACK (' + mode + ')':<34}{r:>10.0f}{ms:>10.1f}")

# What would the stack cost without the one slow transform? Reported to
# size the opportunity, NOT as a proposal to drop it: RandomEqualize is
# part of the thesis's DVD-B augmentation set, so removing it changes the
# science. A faster reimplementation with identical semantics would not.
aug_noeq = torch.nn.Sequential(*[
    m for m in build_augmentations("dkl")
    if not isinstance(m, K.RandomEqualize)]).to(dev)
with torch.no_grad():
    r, ms = timeit(lambda: aug_noeq(x0))
print(f"{'FULL STACK minus RandomEqualize':<34}{r:>10.0f}{ms:>10.1f}")

# the DVD front end, for completeness
cfg = CONDITIONS["dkl_bcs"]
builder = StreamInputBuilder(cfg, age_months=48)
with torch.no_grad():
    r, ms = timeit(lambda: builder(x0))
print(f"{'DVD front end (dkl_bcs)':<34}{r:>10.0f}{ms:>10.1f}")

# host->device copy of one uint8 batch, the cost workers=0 no longer hides
cpu = torch.randint(0, 255, (bs, 3, 256, 256), dtype=torch.uint8).pin_memory()
r, ms = timeit(lambda: cpu.to(dev, non_blocking=True).float().div_(255.0))
print(f"{'H2D copy + float/255':<34}{r:>10.0f}{ms:>10.1f}")
