"""Can the model floor be raised? bf16 autocast / channels_last / batch size.

The step profile showed the model's fwd/bwd caps throughput at ~320 img/s on
an A5000, so even a perfect dataloader leaves ~34 h per 150-epoch run. This
measures the model in isolation (fixed batch, no data) under the standard
throughput levers, and separately checks how far worker scaling goes.
"""

from __future__ import annotations

import argparse
import time

import torch
from torch.utils.data import DataLoader

from src.models.pytorch_models import TwoStreamNetwork
from src.training.standalone_trainer import (
    CONDITIONS, H5ImageDataset, in_channels_for, patch_stream_input_channels,
)

p = argparse.ArgumentParser()
p.add_argument("--dataset", required=True)
p.add_argument("--steps", type=int, default=25)
a = p.parse_args()

dev = "cuda"
cfg = CONDITIONS["dkl_bcs"]
c1, c2 = in_channels_for(cfg)
lossf = torch.nn.CrossEntropyLoss()
print(f"device={torch.cuda.get_device_name(0)}  bf16_supported="
      f"{torch.cuda.is_bf16_supported()}\n", flush=True)


def bench(bs: int, amp: bool, chlast: bool):
    torch.cuda.empty_cache()
    m = TwoStreamNetwork(num_classes=112, legacy_asymmetric_mixing=False)
    patch_stream_input_channels(m, c1, c2)
    m = m.to(dev)
    if chlast:
        m = m.to(memory_format=torch.channels_last)
    opt = torch.optim.Adam(m.parameters(), lr=1e-4)
    x1 = torch.randn(bs, c1, 256, 256, device=dev)
    x2 = torch.randn(bs, c2, 256, 256, device=dev)
    if chlast:
        x1 = x1.to(memory_format=torch.channels_last)
        x2 = x2.to(memory_format=torch.channels_last)
    y = torch.randint(0, 112, (bs,), device=dev)

    def step():
        opt.zero_grad(set_to_none=True)
        if amp:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = lossf(m(x1, x2), y)
        else:
            loss = lossf(m(x1, x2), y)
        loss.backward()
        opt.step()

    for _ in range(5):
        step()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(a.steps):
        step()
    torch.cuda.synchronize()
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30
    torch.cuda.reset_peak_memory_stats()
    del m, opt, x1, x2
    return a.steps * bs / dt, peak


print(f"{'batch':>6}{'bf16':>7}{'chan_last':>11}{'img/s':>10}"
      f"{'peakGB':>9}{'h/150ep':>10}")
best = 0.0
for bs in (128, 256, 384):
    for amp in (False, True):
        for chlast in (False, True):
            try:
                r, peak = bench(bs, amp, chlast)
            except torch.OutOfMemoryError:
                print(f"{bs:>6}{str(amp):>7}{str(chlast):>11}{'OOM':>10}")
                torch.cuda.empty_cache()
                continue
            best = max(best, r)
            print(f"{bs:>6}{str(amp):>7}{str(chlast):>11}{r:>10.1f}"
                  f"{peak:>9.1f}{263200 * 150 / r / 3600:>10.1f}", flush=True)

print(f"\nbest model-only throughput = {best:.0f} img/s "
      f"-> {263200 * 150 / best / 3600:.1f} h per 150-epoch run (model floor)")

# ---- how far does worker scaling go? -----------------------------------
print("\ndataloader scaling (h5 read only)")
ds = H5ImageDataset(a.dataset, "train")
for w in (16, 24, 32):
    dl = DataLoader(ds, batch_size=256, shuffle=True, num_workers=w,
                    pin_memory=True, persistent_workers=True, drop_last=True)
    it = iter(dl)
    next(it)
    t0 = time.time()
    n = 0
    for _ in range(20):
        x, _ = next(it)
        n += x.shape[0]
    dt = time.time() - t0
    print(f"  workers={w:>3}  {n / dt:>7.1f} img/s", flush=True)
    del it, dl
