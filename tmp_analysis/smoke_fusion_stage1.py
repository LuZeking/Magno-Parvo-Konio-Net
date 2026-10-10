"""Model-level checks for the fusion Stage-1 readouts (CPU, seconds).

1. `average` reproduces TwoStreamNetwork's own forward with the same weights.
2. `additive` joint == w z_M + (1 - w) z_P; `hinge` leaves magno's top-k
   untouched and only lowers the rest.
3. The 64-px magno stem hands layer1 the same 64x64 grid, so the trunk output
   keeps its 512 x 8 x 8 shape.
4. Every fs1_ condition builds, with its parameter count.
"""

import torch

from src.models.late_fusion import (LateFusionNetwork, hinge_penalty,
                                    make_magno_stem64)
from src.models.pytorch_models import TwoStreamNetwork
from src.training.standalone_trainer import (CONDITIONS, STAGE1_CONDITIONS,
                                             in_channels_for,
                                             patch_stream_input_channels)


def trunk(c1=1, c2=1, seed=0):
    torch.manual_seed(seed)
    m = TwoStreamNetwork(num_classes=112, like_resnet="resnet18",
                         dvd_strategy="dkl", legacy_asymmetric_mixing=False)
    patch_stream_input_channels(m, c1, c2)
    return m


x1, x2 = torch.rand(4, 1, 256, 256), torch.rand(4, 1, 256, 256)

t = trunk().eval()
ref = t(x1, x2)
lf = LateFusionNetwork(t, "average").eval()
got = lf.forward_heads(x1, x2)
d = float((ref - got["joint"]).abs().max())
print(f"1. average == TwoStreamNetwork forward: max |diff| {d:.2e}")
assert d < 1e-4

lf = LateFusionNetwork(trunk(), "additive", w=0.7).eval()
o = lf.forward_heads(x1, x2)
d = float((o["joint"] - (0.7 * o["magno"] + 0.3 * o["parvo"])).abs().max())
print(f"2a. additive w=0.7 identity: {d:.2e}")
assert d < 1e-5
lf = LateFusionNetwork(trunk(), "hinge", k=3, lam=4.0).eval()
o = lf.forward_heads(x1, x2)
pen = o["joint"] - o["parvo"]
top3 = torch.topk(o["magno"], 3, dim=1).indices
assert float(pen.gather(1, top3).abs().max()) == 0.0
assert float(pen.max()) <= 0.0
assert torch.allclose(pen, hinge_penalty(o["magno"], 3, 4.0))
print(f"2b. hinge: top-3 untouched, others lowered (min {float(pen.min()):.3f})")

t = trunk()
make_magno_stem64(t.stream1_block1, 1)
s = t.stream1_block1(x1).shape
print(f"3. magno stem64 trunk output {tuple(s)}")
assert tuple(s) == (4, 512, 8, 8)

for c in STAGE1_CONDITIONS:
    cfg = CONDITIONS[c]
    c1, c2 = in_channels_for(cfg)
    t = trunk(c1, c2)
    if cfg.get("magno_stem64"):
        make_magno_stem64(t.stream1_block1, c1)
    m = LateFusionNetwork(t, cfg["fusion"], w=cfg.get("w", 0.5),
                          k=cfg.get("k", 3), lam=cfg.get("lam", 0.0))
    n = sum(p.numel() for p in m.parameters())
    print(f"4. {c:<26} fusion={cfg['fusion']:<8} in={c1},{c2} "
          f"blur={cfg['blur']} sigma={cfg.get('blur_sigma', 4.0)} params={n:,}")
print("all model checks passed")
