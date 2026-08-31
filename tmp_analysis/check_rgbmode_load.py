"""Pre-flight for the rgb_tsn architecture-only control.

Two things must be true before this condition is worth running:
  1. build_model(in_channels=(3, 3)) accepts the rgb_tsn checkpoint with every
     stem weight actually loaded -- a mismatch would leave a random stem and
     manufacture a fake "no specialization" result, which is exactly the
     answer this control is supposed to be able to give honestly.
  2. The front end hands BOTH streams the identical tensor, because that
     identity is the whole content of the control.
"""

from __future__ import annotations

import numpy as np
import torch

from src.analysis.encoding import dkl_frontend, features

CK = "/scratch/byrong/VST_data/mpk_runs/rgb_tsn_seed0/checkpoint_last.pt"

state = torch.load(CK, map_location="cpu", weights_only=False)
if "state_dict" in state:
    state = state["state_dict"]

model = features.build_model(checkpoint=CK, in_channels=(3, 3))
sd = model.state_dict()
print("stem shapes and whether the checkpoint weights actually arrived:")
ok = True
for k, v in state.items():
    if "conv1.weight" in k and "layer" not in k:
        got = sd[k]
        same = got.shape == v.shape and torch.allclose(got, v)
        ok &= bool(same)
        print(f"  {k:<34} {tuple(got.shape)}  loaded={same}")

n_loaded = sum(
    1 for k, v in state.items()
    if k in sd and hasattr(v, "shape") and sd[k].shape == v.shape
    and torch.allclose(sd[k].float(), v.float())
)
n_ck = sum(1 for v in state.values() if hasattr(v, "shape"))
print(f"\ntensors matching checkpoint: {n_loaded}/{n_ck}")

ch_min, ch_max = dkl_frontend.compute_dkl_channel_minmax()
rgb = np.random.default_rng(0).integers(0, 256, (4, 425, 425, 3), dtype=np.uint8)
x1, x2 = dkl_frontend.magno_parvo_inputs(rgb, ch_min, ch_max, rgb_mode=True)
identical = torch.equal(x1, x2)
print(f"stream inputs: {tuple(x1.shape)} / {tuple(x2.shape)}  identical={identical}")

print("\nVERDICT:", "SAFE" if (ok and n_loaded == n_ck and identical
                              and x1.shape[1] == 3) else "DO NOT RUN")
