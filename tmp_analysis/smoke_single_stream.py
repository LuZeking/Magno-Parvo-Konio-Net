"""Smoke test for the `dkl_bcs_single` condition, before any GPU time.

Checks, in order:
  1. in_channels_for and the CONDITIONS entry agree.
  2. StreamInputBuilder returns a 2-channel x1 and x2=None, and applies the
     diet to the luminance channel only.
  3. The model builds with ONE trunk (stream2_block1 is None) and a 2-channel
     stem, and forward(x1, None) runs forward AND backward.
  4. Parameter counts, so the capacity confound is a number rather than a
     hand-wave.
  5. The two-stream arm still builds and runs unchanged (no regression).
"""
import torch
import torch.nn as nn

from src.training.standalone_trainer import (
    CONDITIONS, StreamInputBuilder, in_channels_for,
    patch_stream_input_channels,
)
from src.models.pytorch_models import TwoStreamNetwork


def build(cond):
    cfg = CONDITIONS[cond]
    c1, c2 = in_channels_for(cfg)
    single = cfg.get("single_stream", False)
    m = TwoStreamNetwork(num_classes=112, like_resnet="resnet18",
                         seperate_from_layer=0, baseline=single,
                         combined_color=single,
                         dvd_strategy="dkl" if cfg["input_mode"] == "dkl" else "dvd",
                         stream_concat=cfg.get("stream_concat", False),
                         stream1_weight=cfg.get("stream1_weight", 0.5),
                         legacy_asymmetric_mixing=False)
    patch_stream_input_channels(m, c1, c2)
    return cfg, m, (c1, c2)


print("=" * 78)
print("1-2. condition, channels, and the input builder")
print("=" * 78)
cfg, m_s, (c1, c2) = build("dkl_bcs_single")
print(f"  CONDITIONS['dkl_bcs_single'] = {cfg}")
print(f"  in_channels_for -> ({c1}, {c2})")
assert (c1, c2) == (2, 0), (c1, c2)

x = torch.rand(4, 3, 256, 256)
b = StreamInputBuilder(cfg, age_months=48)
x1, x2 = b(x)
print(f"  builder: x1 {tuple(x1.shape)}   x2 = {x2}")
assert x1.shape == (4, 2, 256, 256), x1.shape
assert x2 is None

# The diet must be on luminance only: channel 0 changed, channel 1 untouched.
d_lum = (x1[:, 0] - x[:, 0]).abs().max().item()
d_rg = (x1[:, 1] - x[:, 1]).abs().max().item()
print(f"  max |change| channel 0 (luminance) = {d_lum:.4f}  (expect > 0)")
print(f"  max |change| channel 1 (red-green) = {d_rg:.4e}  (expect 0)")
assert d_lum > 1e-3, "diet was not applied to luminance"
assert d_rg < 1e-6, "red-green channel was modified; it must not be"

print("\n" + "=" * 78)
print("3. one trunk, 2-channel stem, forward AND backward")
print("=" * 78)
print(f"  stream1_block1 is None? {m_s.stream1_block1 is None}")
print(f"  stream2_block1 is None? {m_s.stream2_block1 is None}")
assert m_s.stream2_block1 is None, "stream 2 was built; this is not single-stream"
stem = m_s.stream1_block1[0]
print(f"  stem conv: in_channels={stem.in_channels} out={stem.out_channels}")
assert stem.in_channels == 2, stem.in_channels

out = m_s(x1, x2)
print(f"  forward(x1, None) -> {tuple(out.shape)}")
assert out.shape == (4, 112), out.shape
loss = nn.CrossEntropyLoss()(out, torch.randint(0, 112, (4,)))
loss.backward()
g = sum(p.grad.abs().sum().item() for p in m_s.parameters() if p.grad is not None)
print(f"  backward ok, total |grad| = {g:.3f}")
assert g > 0

print("\n" + "=" * 78)
print("4. the capacity confound, as a number")
print("=" * 78)
_, m_t, _ = build("dkl_bcs")
n = lambda mod: sum(p.numel() for p in mod.parameters() if p.requires_grad)
print(f"  dkl_bcs        (two streams): {n(m_t):>11,} params")
print(f"  dkl_bcs_single (one stream) : {n(m_s):>11,} params"
      f"   = {n(m_s)/n(m_t):.3f}x")
print(f"  trunk only, stream1         : {n(m_t.stream1_block1):>11,}")
print(f"  classifier (shared)         : {n(m_t.classifier):>11,}")
print("\n  So the single-stream arm has roughly half the trunk capacity. That")
print("  confound only bites if it LOSES; a win or a tie is decisive as-is.")

print("\n" + "=" * 78)
print("5. regression check -- the two-stream path is unchanged")
print("=" * 78)
bt = StreamInputBuilder(CONDITIONS["dkl_bcs"], age_months=48)
t1, t2 = bt(x)
print(f"  dkl_bcs builder: x1 {tuple(t1.shape)}  x2 {tuple(t2.shape)}")
assert t1.shape == (4, 1, 256, 256) and t2.shape == (4, 1, 256, 256)
o = m_t(t1, t2)
print(f"  forward -> {tuple(o.shape)}")
assert o.shape == (4, 112)

for cond in ("dkl_base", "dkl_bcs_swap", "dkl_bcs_parvo_highsf", "rgb_tsn"):
    c, mm, ch = build(cond)
    bb = StreamInputBuilder(c, age_months=48)
    a1, a2 = bb(x)
    oo = mm(a1, a2)
    print(f"  {cond:24s} in_channels={ch}  x1={tuple(a1.shape)[1]}ch "
          f"x2={tuple(a2.shape)[1]}ch  out={tuple(oo.shape)}  OK")

print("\nALL CHECKS PASSED")
