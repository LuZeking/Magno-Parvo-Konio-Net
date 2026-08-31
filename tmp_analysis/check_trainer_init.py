"""Reproduce the trainer's own model construction for rgb_tsn and dkl_bcs."""
import torch
from src.models.pytorch_models import TwoStreamNetwork
from src.training.standalone_trainer import (
    CONDITIONS, in_channels_for, patch_stream_input_channels,
)


def stream_diffs(sd):
    out = {}
    for k in sd:
        k2 = k.replace("stream1", "stream2")
        if not k.startswith("stream1") or k2 not in sd:
            continue
        if sd[k].shape != sd[k2].shape:
            out[k] = float("inf")
            continue
        out[k] = (sd[k].float() - sd[k2].float()).abs().max().item()
    return out


for cond in ("rgb_tsn", "dkl_bcs"):
    cfg = CONDITIONS[cond]
    torch.manual_seed(0)
    m = TwoStreamNetwork(
        num_classes=112, like_resnet="resnet18", seperate_from_layer=0,
        baseline=False,
        dvd_strategy="dkl" if cfg["input_mode"] == "dkl" else "dvd",
        stream_concat=cfg.get("stream_concat", False),
        stream1_weight=cfg.get("stream1_weight", 0.5),
        legacy_asymmetric_mixing=False,
    )
    c1, c2 = in_channels_for(cfg)
    pre = stream_diffs(m.state_dict())
    patch_stream_input_channels(m, c1, c2)
    post = stream_diffs(m.state_dict())
    nz_pre = [k for k, v in pre.items() if v > 0]
    nz_post = [k for k, v in post.items() if v > 0]
    print(f"{cond}: in_channels={c1},{c2}")
    print(f"  before patch: {len(nz_pre)}/{len(pre)} pairs differ  {nz_pre}")
    print(f"  after  patch: {len(nz_post)}/{len(post)} pairs differ  {nz_post}")
