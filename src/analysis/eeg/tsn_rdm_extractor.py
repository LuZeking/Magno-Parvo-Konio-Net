"""TSN stream-wise RDMs on the developmental-EEG stimuli.

Feeds the 4 x 12 EEG stimulus subsets through the TSN's DKL front end
(identical to training; see src/analysis/encoding/dkl_frontend.py) and writes
per-subset model RDMs in the SAME npz schema as the infant-project RSA
campaign extractors (rsa2026), so the caches are interchangeable:

    <out_root>/<subset>/<model_name>/rdms_epoch_<NNN>.npz
    keys: {layer}_rdm_no_zscore, {layer}_rdm_zscore, + metadata

Layers emitted per stream: conv1, layer1..layer4, avgpool (global average
pool of layer4), prefixed magno_/parvo_, plus combined_avgpool =
w*magno + (1-w)*parvo — the actual pre-classifier representation.

RDM distances are computed with the infant project's own compute_rdm
(rdm_distance_metric.py) for methodological identity with the other model
families in the campaign.

Run (untrained control):
    python -m src.analysis.eeg.tsn_rdm_extractor \
        --model-name tsn_untrained_seed0 \
        --out-root /scratch/byrong/infant/tsn_rdms_orig_input_v1 \
        --out-root-spearman /scratch/byrong/infant/tsn_rdms_orig_input_v1_spearman
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

INFANT_PROJECT = os.environ.get(
    "INFANT_PROJECT", "/home/byrong/projects/infant"
)
sys.path.insert(0, INFANT_PROJECT)

from rdm_distance_metric import compute_rdm  # noqa: E402  (infant project)

from src.analysis.encoding import dkl_frontend, features  # noqa: E402

EEG_STIMULI_DIR = os.environ.get(
    "EEG_STIMULI_DIR", "/scratch/byrong/infant/infantEEG/stimuli"
)
# Canonical EEG stimulus subsets (must match the campaign / neural RDMs).
SUBSETS = {
    "subset1": [3, 8, 13, 21, 33, 39, 52, 53, 60, 65, 69, 72],
    "subset2": [5, 14, 18, 22, 26, 34, 41, 47, 51, 61, 67, 79],
    "subset3": [1, 7, 12, 23, 32, 35, 43, 48, 55, 63, 70, 76],
    "subset4": [6, 10, 17, 25, 30, 36, 45, 50, 56, 66, 68, 71],
}
# Stage layers inside stream{1,2}_block1 (torchvision resnet children):
STAGE_NAMES = {0: "conv1", 4: "layer1", 5: "layer2", 6: "layer3", 7: "layer4"}


def load_subset_images(subset: str) -> np.ndarray:
    """(12, H, W, 3) uint8 RGB, in canonical subset order."""
    from PIL import Image

    imgs = []
    for sid in SUBSETS[subset]:
        path = os.path.join(EEG_STIMULI_DIR, f"{sid}.jpg")
        img = Image.open(path).convert("RGB")
        imgs.append(np.asarray(img, dtype=np.uint8))
    shapes = {im.shape for im in imgs}
    if len(shapes) > 1:  # resize handled downstream at 256 anyway
        from PIL import Image as I

        target = max(shapes)
        imgs = [
            np.asarray(
                I.fromarray(im).resize(target[1::-1], I.BILINEAR), np.uint8
            )
            for im in imgs
        ]
    return np.stack(imgs)


@torch.no_grad()
def stream_layer_activations(
    model, x_magno: torch.Tensor, x_parvo: torch.Tensor
) -> dict[str, np.ndarray]:
    """Flattened activations per emitted layer for a (12, 1, S, S) batch."""
    out: dict[str, np.ndarray] = {}
    pooled = {}
    for stream_name, block, x in (
        ("magno", model.stream1_block1, x_magno),
        ("parvo", model.stream2_block1, x_parvo),
    ):
        h = x
        for i, layer in enumerate(block):
            h = layer(h)
            if i in STAGE_NAMES:
                out[f"{stream_name}_{STAGE_NAMES[i]}"] = (
                    h.flatten(1).cpu().numpy().astype(np.float64)
                )
        gap = torch.nn.functional.adaptive_avg_pool2d(h, 1).flatten(1)
        out[f"{stream_name}_avgpool"] = gap.cpu().numpy().astype(np.float64)
        pooled[stream_name] = gap
    w = float(model.stream1_weight)
    combined = w * pooled["magno"] + (1.0 - w) * pooled["parvo"]
    out["combined_avgpool"] = combined.cpu().numpy().astype(np.float64)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--epoch", type=int, default=0,
                   help="epoch label for the npz filename/metadata")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stream-concat", action="store_true")
    p.add_argument("--stream1-weight", type=float, default=0.5)
    p.add_argument("--no-lum-blur", action="store_true")
    p.add_argument("--no-lum-cs", action="store_true")
    p.add_argument("--age-months", type=int, default=48,
                   help="DVD CS transform age for the magno input")
    p.add_argument("--out-root", required=True, help="pearson cache root")
    p.add_argument("--out-root-spearman", default=None)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ch_min, ch_max = dkl_frontend.compute_dkl_channel_minmax()
    model = features.build_model(
        checkpoint=args.checkpoint,
        stream_concat=args.stream_concat,
        stream1_weight=args.stream1_weight,
        seed=args.seed,
        device=device,
    )

    out_roots = {"pearson": args.out_root}
    if args.out_root_spearman:
        out_roots["spearman"] = args.out_root_spearman

    for subset in SUBSETS:
        rgb = load_subset_images(subset)
        x_m, x_p = dkl_frontend.magno_parvo_inputs(
            rgb, ch_min, ch_max,
            lum_blur=not args.no_lum_blur,
            lum_cs=not args.no_lum_cs,
            age_months=args.age_months,
            device=device,
        )
        acts = stream_layer_activations(model, x_m, x_p)
        layers = list(acts.keys())

        for distance, root in out_roots.items():
            payload: dict = {
                "epoch": args.epoch,
                "model_name": args.model_name,
                "subset": subset,
                "developmental_age_months": float(args.age_months),
                "layers": np.array(layers),
                "metric": "correlation",
                "n_stimuli": len(SUBSETS[subset]),
                "stimuli_mode": "original_static",
                "checkpoint_path": str(args.checkpoint),
                "checkpoint_kind": (
                    "untrained" if args.checkpoint is None else "supervised_tsn"
                ),
                "distance": distance,
            }
            for layer, feats in acts.items():
                for zs, suffix in ((False, "no_zscore"), (True, "zscore")):
                    payload[f"{layer}_rdm_{suffix}"] = compute_rdm(
                        feats, zscore_feats=zs, distance=distance
                    )
            out_dir = os.path.join(root, subset, args.model_name)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(
                out_dir, f"rdms_epoch_{args.epoch:03d}.npz"
            )
            np.savez(out_path, **payload)
            meta = {
                k: (payload[k].tolist() if isinstance(payload[k], np.ndarray)
                    else payload[k])
                for k in ("epoch", "model_name", "subset", "layers",
                          "distance", "checkpoint_path")
            }
            with open(
                os.path.join(out_dir, f"metadata_epoch_{args.epoch:03d}.json"),
                "w",
            ) as f:
                json.dump(meta, f, indent=2)
            print(f"[{subset}] {distance} -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
