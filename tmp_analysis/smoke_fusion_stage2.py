"""Model-level checks for the Stage-2 (single-objective) TSN (CPU, seconds).

1. `average` at w = 0.5 still reproduces TwoStreamNetwork's forward (Stage-1
   A0 runs are unchanged) and at w = 0.7 reproduces TwoStreamNetwork with
   stream1_weight = 0.7 (the thesis's M-heavy forward).
2. detach_aux: the per-stream readout heads' loss sends no gradient into
   either trunk or the tied head; the joint loss does.
3. Every fs2_ condition builds, with its recipe and parameter count.
"""

import torch

from src.models.late_fusion import LateFusionNetwork
from src.models.pytorch_models import TwoStreamNetwork
from src.training.standalone_trainer import (CONDITIONS, STAGE2_CONDITIONS,
                                             in_channels_for,
                                             patch_stream_input_channels)


def trunk(c1=1, c2=1, seed=0, w=0.5):
    torch.manual_seed(seed)
    m = TwoStreamNetwork(num_classes=112, like_resnet="resnet18",
                         dvd_strategy="dkl", stream1_weight=w,
                         legacy_asymmetric_mixing=False)
    patch_stream_input_channels(m, c1, c2)
    return m


x1, x2 = torch.rand(4, 1, 256, 256), torch.rand(4, 1, 256, 256)
y = torch.tensor([0, 1, 2, 3])

for w in (0.5, 0.7):
    t = trunk(w=w).eval()
    ref = t(x1, x2)
    got = LateFusionNetwork(t, "average", w=w).eval().forward_heads(x1, x2)["joint"]
    d = float((ref - got).abs().max())
    print(f"1. average w={w} == TwoStreamNetwork(stream1_weight={w}): max |diff| {d:.2e}")
    assert d < 1e-4

net = LateFusionNetwork(trunk(), "average", w=0.5, detach_aux=True).train()
ce = torch.nn.CrossEntropyLoss()
out = net.forward_heads(x1, x2)
(ce(out["magno"], y) + ce(out["parvo"], y)).backward()
body = [p for n, p in net.named_parameters() if not n.startswith(("head_m", "head_p"))]
leak = max(float(p.grad.abs().max()) if p.grad is not None else 0.0 for p in body)
heads = min(float(p.grad.abs().sum()) for n, p in net.named_parameters()
            if n.startswith(("head_m", "head_p")) and p.ndim > 1)
print(f"2a. readout-head loss: max |grad| on trunks + tied head {leak:.1e}, heads {heads:.2e}")
assert leak == 0.0 and heads > 0
net.zero_grad()
out = net.forward_heads(x1, x2)
ce(out["joint"], y).backward()
g1 = float(net.stream1_block1[0].weight.grad.abs().sum())
g2 = float(net.stream2_block1[0].weight.grad.abs().sum())
print(f"2b. joint loss reaches both trunks: |grad| magno stem {g1:.2e}, parvo stem {g2:.2e}")
assert g1 > 0 and g2 > 0

for c in STAGE2_CONDITIONS:
    cfg = CONDITIONS[c]
    c1, c2 = in_channels_for(cfg)
    m = LateFusionNetwork(trunk(c1, c2, w=cfg["stream1_weight"]), cfg["fusion"],
                          w=cfg["w"], detach_aux=cfg["detach_aux"])
    n = sum(p.numel() for p in m.parameters())
    r = cfg["recipe"]
    print(f"3. {c:<22} w={cfg['w']} in={c1},{c2} aux={r['aux_weight']} detach={cfg['detach_aux']} "
          f"ls={r['label_smoothing']} lr={r['lr']} params={n:,}")
print("all stage-2 model checks passed")
