"""Unit-level layer-4 features for the 1-to-1 test (plan §3-§4).

One job per subject loads that subject's NSD images once and runs every network,
so ~5 GB of stimuli are read 8 times rather than 8 x 17 times.

TSN units: output of each trunk's `layer4` (child 7 of `stream*_block1`),
512 x 8 x 8 at 256 px, restricted to a fixed superset of 128 channels per grid
position per stream (RNG seed 0, identical for every TSN network) -> 16,384
units: [stream 1 (magno) superset, stream 2 (parvo) superset].

TDANN units: `base_model.layer4[1]` output, all 512 x 7 x 7 = 25,088 units,
NSD images resized to 224 with ImageNet normalisation (spacestream
`NSD_TRANSFORMS`: Resize(224) -> ToTensor -> Normalize).

    python -m src.analysis.finzi_1to1.extract --subject 1 [--networks a,b]
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from src.analysis.encoding import dkl_frontend, features, nsd_data
from src.analysis.finzi_1to1 import OUT_ROOT, features_path

RUNS = "/scratch/byrong/VST_data/mpk_runs"
TDANN_DIR = "/scratch/byrong/TSN/finzi_1to1/cache/tdann_ckpt"  # OSF 64qv3 / qy32x
LAYER4_CHILD = 7
GRID_TSN = 8
N_SUPERSET = 128  # channels per position per stream

FRONT_ENDS = {
    "diet": dict(lum_blur=True, lum_cs=True, transform_target="magno"),
    "base": dict(lum_blur=False, lum_cs=False, transform_target="magno"),
    "swap": dict(lum_blur=True, lum_cs=True, transform_target="parvo"),
}


def network_registry() -> dict[str, dict]:
    reg = {}
    for k in range(3):
        reg[f"diet_s{k}"] = dict(kind="tsn", run=f"dkl_bcs_seed{k}", fe="diet")
        reg[f"nodiet_s{k}"] = dict(kind="tsn", run=f"dkl_base_seed{k}", fe="base")
        reg[f"swap_s{k}"] = dict(kind="tsn", run=f"dkl_bcs_swap_seed{k}", fe="swap")
        reg[f"untrained_diet_s{k}"] = dict(kind="tsn", run=None, seed=k, fe="diet")
        reg[f"untrained_nodiet_s{k}"] = dict(kind="tsn", run=None, seed=k, fe="base")
    reg["tdann_a025_s0"] = dict(kind="tdann", ckpt="simclr_a025_seed0.torch")
    reg["tdann_a0_s0"] = dict(kind="tdann", ckpt="simclr_a0_seed0.torch")
    # Seeds 1-4 of the same alpha = 0.25 SimCLR TDANN (Margalit et al. OSF 64qv3,
    # isoswap_3_seed_<k>), for the magno-likeness probe (PLAN_TDANN_MAGNO_PROBE).
    for k in range(1, 5):
        reg[f"tdann_a025_s{k}"] = dict(kind="tdann", ckpt=f"simclr_a025_seed{k}.torch")
    return reg


def tsn_superset() -> np.ndarray:
    """(2, 64, 128) channel ids: stream x grid position x superset slot."""
    rng = np.random.default_rng(0)
    return np.stack([
        np.stack([np.sort(rng.choice(512, N_SUPERSET, replace=False))
                  for _ in range(GRID_TSN * GRID_TSN)])
        for _ in range(2)
    ])


def tsn_unit_table() -> dict[str, np.ndarray]:
    """Per-unit stream, channel, grid position and flat (C,H,W) index."""
    ch = tsn_superset()
    stream = np.repeat([0, 1], GRID_TSN * GRID_TSN * N_SUPERSET)
    pos = np.tile(np.repeat(np.arange(GRID_TSN * GRID_TSN), N_SUPERSET), 2)
    chan = ch.reshape(-1)
    flat = chan * GRID_TSN * GRID_TSN + pos  # index into layer4.flatten(1)
    return dict(stream=stream.astype(np.int8), pos=pos.astype(np.int16),
                chan=chan.astype(np.int16), flat=flat.astype(np.int64))


def tdann_unit_table() -> dict[str, np.ndarray]:
    n = 512 * 49
    flat = np.arange(n)
    return dict(stream=np.zeros(n, np.int8), pos=(flat % 49).astype(np.int16),
                chan=(flat // 49).astype(np.int16), flat=flat)


def build_tdann(path: str, device: str) -> torch.nn.Module:
    from torchvision.models.resnet import resnet18

    ck = torch.load(path, map_location="cpu", weights_only=False)
    trunk = ck["classy_state_dict"]["base_model"]["model"]["trunk"]
    base = {k.removeprefix("base_model."): v for k, v in trunk.items()
            if k.startswith("base_model.")}
    m = resnet18(weights=None)
    missing, unexpected = m.load_state_dict(base, strict=False)
    missing = [k for k in missing if not k.startswith("fc.")]
    if missing or unexpected:
        raise RuntimeError(f"TDANN {path}: missing={missing} unexpected={unexpected}")
    return m.eval().to(device)


@torch.no_grad()
def tsn_layer4(block: torch.nn.Sequential, x: torch.Tensor) -> torch.Tensor:
    for i, layer in enumerate(block):
        x = layer(x)
        if i == LAYER4_CHILD:
            return x
    raise RuntimeError("stream block shorter than layer4")


@torch.no_grad()
def tdann_layer4(m, rgb_uint8: np.ndarray, device: str) -> torch.Tensor:
    t = torch.from_numpy(np.ascontiguousarray(rgb_uint8)).to(device)
    t = t.permute(0, 3, 1, 2).float() / 255.0
    t = F.interpolate(t, size=(224, 224), mode="bilinear", antialias=True,
                      align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
    x = (t - mean) / std
    x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
    x = m.layer3(m.layer2(m.layer1(x)))
    return m.layer4(x)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--subject", type=int, required=True)
    p.add_argument("--networks", default="all")
    p.add_argument("--chunk", type=int, default=250)
    p.add_argument("--limit", type=int, default=0, help="smoke test: first N images")
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()

    reg = network_registry()
    names = list(reg) if args.networks == "all" else args.networks.split(",")
    todo = [n for n in names if not os.path.exists(features_path(n, args.subject))]
    print(f"[extract] subj{args.subject:02d} device={device} networks={todo}", flush=True)
    if not todo:
        return

    stim_info = nsd_data.load_stim_info()
    split = nsd_data.build_train_test_split(args.subject, stim_info)
    train_ids = np.unique(split["train_image_ids"])
    test_ids = split["test_image_ids"]
    all_ids = np.concatenate([train_ids, test_ids])
    if args.limit:
        all_ids = np.concatenate([train_ids[: args.limit], test_ids[: args.limit]])
        train_ids, test_ids = train_ids[: args.limit], test_ids[: args.limit]
    n_train = len(train_ids)

    ch_min, ch_max = dkl_frontend.compute_dkl_channel_minmax()
    tsn_units, tdann_units = tsn_unit_table(), tdann_unit_table()
    gather = torch.from_numpy(tsn_units["flat"]).to(device)
    s1 = torch.from_numpy(tsn_units["stream"] == 0).to(device)

    models, feats, prov = {}, {}, {}
    for n in todo:
        spec = reg[n]
        if spec["kind"] == "tsn":
            ckpt = None
            if spec["run"] is not None:
                ckpt = os.path.join(RUNS, spec["run"], "checkpoint_last.pt")
                features.assert_converged(ckpt)
            models[n] = features.build_model(checkpoint=ckpt, seed=spec.get("seed", 0),
                                             device=device)
            n_units = len(tsn_units["flat"])
            prov[n] = dict(spec, checkpoint=ckpt)
        else:
            ckpt = os.path.join(TDANN_DIR, spec["ckpt"])
            models[n] = build_tdann(ckpt, device)
            n_units = len(tdann_units["flat"])
            prov[n] = dict(spec, checkpoint=ckpt)
        feats[n] = np.zeros((len(all_ids), n_units), dtype=np.float16)
    print(f"[extract] models built in {time.time() - t0:.0f}s", flush=True)

    fes = sorted({reg[n]["fe"] for n in todo if reg[n]["kind"] == "tsn"})
    for start in range(0, len(all_ids), args.chunk):
        stop = min(start + args.chunk, len(all_ids))
        rgb = dkl_frontend.load_nsd_images(all_ids[start:stop])
        for fe in fes:
            x_m, x_p = dkl_frontend.magno_parvo_inputs(
                rgb, ch_min, ch_max, device=device, **FRONT_ENDS[fe])
            for n in todo:
                if reg[n]["kind"] != "tsn" or reg[n]["fe"] != fe:
                    continue
                m = models[n]
                y1 = tsn_layer4(m.stream1_block1, x_m)
                y2 = tsn_layer4(m.stream2_block1, x_p)
                if y1.shape[1:] != (512, GRID_TSN, GRID_TSN):
                    raise RuntimeError(f"layer4 shape {tuple(y1.shape)}")
                y1f, y2f = y1.flatten(1), y2.flatten(1)
                out = torch.where(s1[None, :], y1f[:, gather], y2f[:, gather])
                feats[n][start:stop] = out.half().cpu().numpy()
        for n in todo:
            if reg[n]["kind"] == "tdann":
                y = tdann_layer4(models[n], rgb, device)
                if y.shape[1:] != (512, 7, 7):
                    raise RuntimeError(f"TDANN layer4 shape {tuple(y.shape)}")
                feats[n][start:stop] = y.flatten(1).half().cpu().numpy()
        print(f"[extract] {stop}/{len(all_ids)} images {time.time() - t0:.0f}s", flush=True)

    for n in todo:
        units = tdann_units if reg[n]["kind"] == "tdann" else tsn_units
        f = feats[n]
        if not np.isfinite(f).all():
            raise RuntimeError(f"{n}: non-finite features")
        out = features_path(n, args.subject)
        tmp = out + ".tmp.npz"
        np.savez(tmp, train_feats=f[:n_train], test_feats=f[n_train:],
                 train_image_ids=train_ids, test_image_ids=test_ids,
                 unit_stream=units["stream"], unit_pos=units["pos"],
                 unit_chan=units["chan"],
                 provenance=json.dumps(prov[n]))
        os.replace(tmp, out)
        print(f"[extract] wrote {out} {f.shape}", flush=True)
    print(f"[extract] done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
