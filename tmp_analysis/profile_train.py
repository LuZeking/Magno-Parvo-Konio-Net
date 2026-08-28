"""Where does training time actually go?

The timing probe measured ~57 images/s for two ResNet18s at 256px on an
A5000, which is ~10x slower than that hardware should manage. This splits
the step into its three candidate bottlenecks -- h5 dataloading, the
per-batch DVD front end, and the model's forward/backward -- so the fix
targets the real one instead of a guess.
"""

from __future__ import annotations

import argparse
import time

import torch
from torch.utils.data import DataLoader

from src.training.standalone_trainer import (
    CONDITIONS, H5ImageDataset, StreamInputBuilder, build_augmentations,
    in_channels_for, patch_stream_input_channels,
)
from src.models.pytorch_models import TwoStreamNetwork

p = argparse.ArgumentParser()
p.add_argument("--dataset", required=True)
p.add_argument("--condition", default="dkl_bcs")
p.add_argument("--batch-size", type=int, default=256)
p.add_argument("--num-workers", type=int, default=8)
p.add_argument("--steps", type=int, default=30)
a = p.parse_args()

dev = "cuda"
cfg = CONDITIONS[a.condition]
print(f"device={torch.cuda.get_device_name(0)}  condition={a.condition}  "
      f"bs={a.batch_size}  workers={a.num_workers}", flush=True)


def sync():
    torch.cuda.synchronize()


ds = H5ImageDataset(a.dataset, "train")
loader = DataLoader(ds, batch_size=a.batch_size, shuffle=True,
                    num_workers=a.num_workers, pin_memory=True,
                    persistent_workers=a.num_workers > 0, drop_last=True)

c1, c2 = in_channels_for(cfg)
model = TwoStreamNetwork(num_classes=112, legacy_asymmetric_mixing=False)
patch_stream_input_channels(model, c1, c2)
model = model.to(dev)
opt = torch.optim.Adam(model.parameters(), lr=1e-4)
builder = StreamInputBuilder(cfg, age_months=48)
augment = build_augmentations(cfg["input_mode"]).to(dev)
lossf = torch.nn.CrossEntropyLoss()

# ---- 1. dataloader only ------------------------------------------------
it = iter(loader)
next(it)  # pay worker startup outside the timer
sync(); t0 = time.time()
n = 0
for _ in range(a.steps):
    x, y = next(it)
    n += x.shape[0]
t_load = time.time() - t0
print(f"[1] dataloader only          {t_load:7.2f}s  "
      f"{n / t_load:8.1f} img/s", flush=True)

# ---- 2. + H2D copy and DVD front end -----------------------------------
it = iter(loader)
next(it)
sync(); t0 = time.time()
n = 0
for _ in range(a.steps):
    x, y = next(it)
    x = x.to(dev, non_blocking=True).float() / 255.0
    x = augment(x)
    x1, x2 = builder(x)
    n += x.shape[0]
sync(); t_front = time.time() - t0
print(f"[2] + H2D + augment + DVD    {t_front:7.2f}s  "
      f"{n / t_front:8.1f} img/s   (front end adds "
      f"{t_front - t_load:.2f}s)", flush=True)

# ---- 3. model fwd/bwd on a FIXED batch (no data, no front end) ---------
x1f, x2f = x1.detach().clone(), x2.detach().clone()
yf = y.to(dev)
for _ in range(3):  # warm up cudnn autotuning
    opt.zero_grad(set_to_none=True)
    lossf(model(x1f, x2f), yf).backward()
    opt.step()
sync(); t0 = time.time()
for _ in range(a.steps):
    opt.zero_grad(set_to_none=True)
    lossf(model(x1f, x2f), yf).backward()
    opt.step()
sync(); t_model = time.time() - t0
print(f"[3] model fwd/bwd only       {t_model:7.2f}s  "
      f"{a.steps * a.batch_size / t_model:8.1f} img/s", flush=True)

# ---- 4. full step ------------------------------------------------------
it = iter(loader)
next(it)
sync(); t0 = time.time()
n = 0
for _ in range(a.steps):
    x, y = next(it)
    x = x.to(dev, non_blocking=True).float() / 255.0
    y = y.to(dev, non_blocking=True)
    x = augment(x)
    x1, x2 = builder(x)
    opt.zero_grad(set_to_none=True)
    lossf(model(x1, x2), y).backward()
    opt.step()
    n += x.shape[0]
sync(); t_full = time.time() - t0
print(f"[4] full training step       {t_full:7.2f}s  "
      f"{n / t_full:8.1f} img/s", flush=True)

print(f"\nbudget: load {t_load:.1f}s | front end {t_front - t_load:.1f}s | "
      f"model {t_model:.1f}s | measured full {t_full:.1f}s")
print(f"epoch (263200 img) at full rate = {263200 / (n / t_full) / 60:.1f} min"
      f"  -> 150 epochs = {263200 / (n / t_full) * 150 / 3600:.1f} h")
print(f"input shapes: stream1 {tuple(x1.shape)}  stream2 {tuple(x2.shape)}")
