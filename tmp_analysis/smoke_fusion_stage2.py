"""Model-level checks for the Stage-2 (single-objective) TSN (CPU, seconds).

1. `average` at w = 0.5 still reproduces TwoStreamNetwork's forward (Stage-1
   A0 runs are unchanged) and at w = 0.7 reproduces TwoStreamNetwork with
   stream1_weight = 0.7 (the thesis's M-heavy forward).
2. detach_aux: the per-stream readout heads' loss sends no gradient into
   either trunk or the tied head; the joint loss does.
3. Every fs2_ condition builds, with its recipe and parameter count.
4-7. OGM-GE (Stage 2b).
8. norm_scale (Stage 2d): both stream features have norm s, and scaling a
   trunk's output leaves the decision unchanged (it does change it without).
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
                          w=cfg["w"], detach_aux=cfg["detach_aux"],
                          norm_scale=cfg.get("norm_scale"))
    n = sum(p.numel() for p in m.parameters())
    r = cfg["recipe"]
    print(f"3. {c:<24} w={cfg['w']} in={c1},{c2} aux={r['aux_weight']} detach={cfg['detach_aux']} "
          f"ls={r['label_smoothing']} lr={r['lr']} epochs={r.get('epochs', '-')} "
          f"norm={cfg.get('norm_scale')} params={n:,}")

# --- Stage 2b: OGM-GE (fs2_bcs_avg_w050_ogm) -------------------------------
import copy  # noqa: E402

from src.training.standalone_trainer import OGMGE, build_stage1_optimizer  # noqa: E402

cfg = CONDITIONS["fs2_bcs_avg_w050_ogm"]
net = LateFusionNetwork(trunk(w=0.5), "average", w=0.5, detach_aux=True).train()
out = net.forward_heads(x1, x2)
c_m, c_p = net.stream_contributions(out)
d = float((c_m + c_p - out["joint"]).abs().max())
print(f"4. stream contributions sum to the joint logit: max |diff| {d:.2e}")
assert d < 1e-5

opt_split, _ = build_stage1_optimizer(net, cfg["recipe"], 10, 2)
conv_ids = {id(p) for b in (net.stream1_block1, net.stream2_block1)
            for p in b.parameters() if p.ndim == 4}
named = {g.get("name"): {id(p) for p in g["params"]} for g in opt_split.param_groups}
all_ids = [id(p) for g in opt_split.param_groups for p in g["params"]]
print(f"5a. groups {[ (g.get('name'), len(g['params'])) for g in opt_split.param_groups]}")
assert named["magno_conv"] | named["parvo_conv"] == conv_ids
assert len(all_ids) == len(set(all_ids)) == len(list(net.parameters()))

# split groups at k = 1 (no OGM applied) == the unsplit Stage-2 optimiser
a, b = copy.deepcopy(net), copy.deepcopy(net)
oa, _ = build_stage1_optimizer(a, cfg["recipe"], 10, 2)
ob, _ = build_stage1_optimizer(b, CONDITIONS["fs2_bcs_avg_w050"]["recipe"], 10, 2)
for m_, o_ in ((a, oa), (b, ob)):
    for _ in range(2):
        o_.zero_grad()
        ce(m_.forward_heads(x1, x2)["joint"], y).backward()
        o_.step()
d = max(float((pa - pb).abs().max()) for pa, pb in zip(a.parameters(), b.parameters()))
print(f"5b. split vs unsplit AdamW, 2 steps without modulation: max |diff| {d:.2e}")
assert d == 0.0

ogm = OGMGE(opt_split, cfg["recipe"]["ogm"])
opt_split.zero_grad()
out = net.forward_heads(x1, x2)
loss = ce(out["joint"], y)
loss.backward()
g_before = {id(p): p.grad.clone() for p in net.parameters() if p.grad is not None}
lr0 = {g.get("name"): g["lr"] for g in opt_split.param_groups}
saved = ogm.modulate(net, out, y, epoch=0)
s = ogm.summary()
r = s["ratio_parvo_over_magno"]
k_exp = 1 - torch.tanh(torch.tensor(0.1 * (r if r > 1 else 1 / r))).item()
dom, other = ("parvo_conv", "magno_conv") if r > 1 else ("magno_conv", "parvo_conv")
lr_now = {g.get("name"): g["lr"] for g in opt_split.param_groups}
print(f"6a. r = s_P/s_M = {r:.3f}: {dom} lr x {lr_now[dom] / lr0[dom]:.4f} (expect {k_exp:.4f}), "
      f"{other} x {lr_now[other] / lr0[other]:.4f}, others unchanged")
assert abs(lr_now[dom] / lr0[dom] - k_exp) < 1e-6 and lr_now[other] == lr0[other]
assert lr_now[None] == lr0[None]
changed = {id(p) for p in net.parameters()
           if p.grad is not None and not torch.equal(p.grad, g_before[id(p)])}
print(f"6b. GE noise touched {len(changed)} tensors = all {len(conv_ids)} trunk conv weights: "
      f"{changed == conv_ids}")
assert changed == conv_ids
ogm.restore(saved)
assert {g.get("name"): g["lr"] for g in opt_split.param_groups} == lr0
opt_split.zero_grad()
out = net.forward_heads(x1, x2)
ce(out["joint"], y).backward()
g_before = {id(p): p.grad.clone() for p in net.parameters() if p.grad is not None}
assert ogm.modulate(net, out, y, epoch=26) is None
assert all(torch.equal(p.grad, g_before[id(p)]) for p in net.parameters() if p.grad is not None)
print("6c. lr restored after the step; epoch 26 (outside 0-25): no lr change, no noise")
print(f"7. {'fs2_bcs_avg_w050_ogm'} recipe ogm={cfg['recipe']['ogm']} aux={cfg['recipe']['aux_weight']} "
      f"detach={cfg['detach_aux']} (the trunks see only the joint CE)")

# --- Stage 2d: scale-pinned average (fs2_bcs_navg_*) -------------------------
cfg = CONDITIONS["fs2_bcs_navg_w070_ls_e25"]
for s in (cfg["norm_scale"], None):
    net = LateFusionNetwork(trunk(w=0.7), "average", w=0.7, detach_aux=True,
                            norm_scale=s).eval()
    with torch.no_grad():
        h_m, h_p = net.features(x1, x2)
        ref = net.forward_heads(x1, x2)["joint"]
        hook = net.stream2_block1.register_forward_hook(lambda mod, i, o: 3.0 * o)
        louder = net.forward_heads(x1, x2)["joint"]
        hook.remove()
    d = float((louder - ref).abs().max())
    if s:
        nm, npv = float(h_m.norm(dim=1).sub(s).abs().max()), float(h_p.norm(dim=1).sub(s).abs().max())
        print(f"8a. norm_scale {s:.3f}: | |h| - s | magno {nm:.1e}, parvo {npv:.1e}; "
              f"parvo trunk x3 changes the joint by {d:.1e}")
        assert nm < 1e-3 and npv < 1e-3 and d < 1e-4
    else:
        print(f"8b. without norm_scale, parvo trunk x3 changes the joint by {d:.2e}")
        assert d > 1e-3
print("all stage-2 model checks passed")
