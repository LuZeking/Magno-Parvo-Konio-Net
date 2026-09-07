"""Verify the premise of §64.5's parameter-matched control before spending 78 GPU-h.

§72.6 asserts one ResNet34 trunk is ~21.3M against ~22.4M for two ResNet18
trunks. That number decides whether this arm is actually parameter-matched, so
it is measured here rather than trusted, together with a forward pass on the
real input shape.
"""
import sys

import torch

sys.path.insert(0, "/home/byrong/projects/TSN/Magno-Parvo-Konio-Net/"
                   ".claude/worktrees/mpk-dev")

from src.models.pytorch_models import TwoStreamNetwork  # noqa: E402
from src.training.standalone_trainer import (  # noqa: E402
    CONDITIONS, in_channels_for, patch_stream_input_channels,
)


def build(condition, arch=None):
    cfg = CONDITIONS[condition]
    single = cfg.get("single_stream", False)
    a = arch or cfg.get("like_resnet", "resnet18")
    m = TwoStreamNetwork(
        num_classes=112, like_resnet=a, seperate_from_layer=0,
        baseline=single, combined_color=single,
        dvd_strategy="dkl" if cfg["input_mode"] == "dkl" else "dvd",
        stream_concat=cfg.get("stream_concat", False),
        stream1_weight=cfg.get("stream1_weight", 0.5),
        legacy_asymmetric_mixing=False,
    )
    c1, c2 = in_channels_for(cfg)
    patch_stream_input_channels(m, c1, c2)
    return m, a, (c1, c2)


print(f"{'condition':22s} {'arch':9s} {'in_ch':>7s} {'params':>12s}")
counts = {}
for cond in ("dkl_bcs", "dkl_bcs_single", "dkl_bcs_single34"):
    m, a, ch = build(cond)
    n = sum(p.numel() for p in m.parameters())
    counts[cond] = n
    print(f"{cond:22s} {a:9s} {str(ch):>7s} {n:12,d}")

two = counts["dkl_bcs"]
one18 = counts["dkl_bcs_single"]
one34 = counts["dkl_bcs_single34"]
print(f"\ntwo ResNet18 trunks   {two:,}")
print(f"one ResNet18 trunk    {one18:,}   ({one18 / two:.1%} of two-stream)")
print(f"one ResNet34 trunk    {one34:,}   ({one34 / two:.1%} of two-stream)")
print(f"\n§72.6 claims ~21.3M vs ~22.4M -> ratio {21.3 / 22.4:.3f}")
print(f"measured ratio resnet34-single / two-stream = {one34 / two:.3f}")
gap = abs(one34 - two) / two
print(f"parameter mismatch {gap:.1%} "
      f"({'MATCHED, control is valid' if gap < 0.10 else 'NOT MATCHED -- the control does not do its job'})")

# forward pass on the real input: one stream, both DKL channels stacked
print("\nforward pass, batch 2 x 2ch x 256 x 256:")
for cond in ("dkl_bcs_single", "dkl_bcs_single34"):
    m, a, (c1, c2) = build(cond)
    m.eval()
    x = torch.randn(2, c1, 256, 256)
    with torch.no_grad():
        y = m(x, None)
    print(f"  {cond:20s} {a:9s} -> logits {tuple(y.shape)}")
    assert y.shape == (2, 112), f"unexpected logits shape {y.shape}"
print("\nOK")
