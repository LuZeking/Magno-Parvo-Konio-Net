"""Cue-conflict shape-bias evaluation, with per-stream decomposition.

Behavioural shape bias (Geirhos et al. 2018; Lu et al. 2026):
    B_shape = N_shape / (N_shape + N_texture)
restricted to the 16 cue-conflict categories, over the canonical 1280
style-transfer images. Filenames follow the Geirhos convention
`<shape><i>-<texture><j>.png`, so shape/texture labels come from the name.

Beyond the aggregate number this module answers the thesis's open question
(§5.2: why does behaviour follow the shape-biased magno stream when the
parvo stream is the better classifier?) by decomposing the decision:

  * ``full``        both streams, as trained
  * ``magno_only``  parvo stream features zeroed before the weighted sum
  * ``parvo_only``  magno stream features zeroed
  * plus each stream's logit contribution through the shared classifier,
    so shape/texture preference can be attributed per stream on the very
    same images.

Because the classifier is linear on the weighted feature sum, zeroing one
stream isolates the other's contribution exactly (up to the bias term).

Run:
    python -m src.analysis.behavior.cue_conflict \
        --checkpoint <ckpt.pt> --model-name dkl_bcs_seed0 \
        --out-dir results/cue_conflict
    python -m src.analysis.behavior.cue_conflict \
        --checkpoint fs2_dvd300_avg_w050_ls_seed0 --dvd-age-months 300 \
        --model-name fs2_dvd300_avg_w050_ls_seed0 --out-dir results/cue_conflict

(--checkpoint takes a Stage-2 run name; a DVD-diet run is refused without
--dvd-age-months, see dkl_frontend.check_frontend.)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np
import torch
import torch.nn.functional as F

# dkl_frontend and features pull in the training stack (dvd_scale_free); they are imported where the
# model is run, so the scoring helpers (shape_bias, load_class_index, GEIRHOS_CLASSES) import without it.


def stream_in_channels(args) -> tuple[int, int]:
    """Per-stream stem channel counts implied by the front-end flags.

    Mirrors ``standalone_trainer.in_channels_for``; kept here because the
    analysis entry points select the condition by CLI flag rather than by
    the trainer's condition dict. A wrong answer is loud, not silent --
    ``load_state_dict`` raises on a stem shape mismatch regardless of
    ``strict`` -- so this cannot quietly produce a randomly-initialized stem.
    """
    if getattr(args, "rgb_mode", False):
        return 3, 3
    return 1, (2 if getattr(args, "parvo_highsf", False) else 1)

CUE_CONFLICT_DIR = os.environ.get(
    "CUE_CONFLICT_DIR",
    "/scratch/byrong/encoding/datasets/geirhos_cue_conflict/stimuli/"
    "style-transfer-preprocessed-512",
)
MINIECOSET_H5 = os.environ.get(   # the same default as dkl_frontend.MINIECOSET_H5
    "MINIECOSET_H5", "/scratch/byrong/VST_data/VST_datasets/texture2shape_miniecoset_256px.h5")
GEIRHOS_CLASSES = [
    "airplane", "bear", "bicycle", "bird", "boat", "bottle", "car", "cat",
    "chair", "clock", "dog", "elephant", "keyboard", "knife", "oven", "truck",
]


def load_class_index() -> dict[str, int]:
    """Map each Geirhos category to its MiniEcoset class index."""
    import h5py

    with h5py.File(MINIECOSET_H5, "r") as f:
        cats = [
            c.decode() if isinstance(c, (bytes, np.bytes_)) else str(c)
            for c in f["categories"][:]
        ]
    lower = [c.lower() for c in cats]
    mapping = {}
    for g in GEIRHOS_CLASSES:
        if g not in lower:
            raise KeyError(f"category {g!r} missing from MiniEcoset")
        mapping[g] = lower.index(g)
    return mapping


def list_cue_conflict_images() -> list[tuple[str, str, str]]:
    """[(path, shape_class, texture_class)] excluding same-cue images."""
    out = []
    paths = sorted(
        glob.glob(os.path.join(CUE_CONFLICT_DIR, "**", "*.png"), recursive=True)
        + glob.glob(os.path.join(CUE_CONFLICT_DIR, "**", "*.jpg"), recursive=True)
    )
    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        m = re.match(r"([a-z]+)\d+-([a-z]+)\d+$", stem)
        if not m:
            continue
        shape, texture = m.group(1), m.group(2)
        if shape == texture:  # Geirhos excluded these
            continue
        if shape in GEIRHOS_CLASSES and texture in GEIRHOS_CLASSES:
            out.append((path, shape, texture))
    return out


def load_images(paths: list[str]) -> np.ndarray:
    from PIL import Image

    return np.stack(
        [np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
         for p in paths]
    )


@torch.no_grad()
def stream_logits(
    model, x_magno: torch.Tensor, x_parvo: torch.Tensor, batch_size: int = 32,
    device: str = "cpu",
) -> dict[str, np.ndarray]:
    """Logits for the full model and for each stream in isolation."""
    outs: dict[str, list[np.ndarray]] = {
        "full": [], "magno_only": [], "parvo_only": []
    }
    w = float(model.stream1_weight)

    for start in range(0, x_magno.shape[0], batch_size):
        xm = x_magno[start : start + batch_size].to(device)
        xp = x_parvo[start : start + batch_size].to(device)

        y1 = model.stream1_block1(xm)
        y2 = model.stream2_block1(xp)
        if model.stream_concat:
            y1, y2 = model._mix_streams(y1, y2)
            y1 = model.stream1_block2(y1)
            y2 = model.stream2_block2(y2)

        combos = {
            "full": w * y1 + (1 - w) * y2,
            "magno_only": w * y1,
            "parvo_only": (1 - w) * y2,
        }
        for key, feat in combos.items():
            outs[key].append(model.classifier(feat).float().cpu().numpy())

    return {k: np.concatenate(v) for k, v in outs.items()}



def build_baseline(
    checkpoint: str | None,
    like_resnet: str = "resnet50",
    num_classes: int = 112,
    seed: int = 0,
    device: str = "cpu",
):
    """Plain single-stream ResNet baseline (RGB Baseline / DVD Baseline).

    These are the anchors for the paper's comparative claim ("more
    shape-biased than standard CNNs, approaching DVD-trained models"), so
    they are scored with the identical decision rule as the TSN.
    """
    import torch.nn as nn
    import torchvision.models as tvm

    torch.manual_seed(seed)
    model = getattr(tvm, like_resnet)(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        cleaned = {}
        for k, v in state.items():
            for prefix in ("network.", "model.", "module."):
                if k.startswith(prefix):
                    k = k[len(prefix):]
            cleaned[k] = v
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if missing or unexpected:
            print(
                f"[cue_conflict] baseline load_state_dict: {len(missing)} "
                f"missing, {len(unexpected)} unexpected keys",
                flush=True,
            )
    return model.eval().to(device)


@torch.no_grad()
def baseline_logits(
    model, rgb_uint8: np.ndarray, batch_size: int = 32, device: str = "cpu",
    image_size: int = 224,
) -> dict[str, np.ndarray]:
    """Logits for a single-stream baseline on RGB cue-conflict images."""
    x = torch.from_numpy(np.ascontiguousarray(rgb_uint8)).permute(0, 3, 1, 2)
    x = x.float() / 255.0
    if x.shape[-1] != image_size:
        x = F.interpolate(
            x, size=(image_size, image_size), mode="bilinear",
            antialias=True, align_corners=False,
        )
    out = []
    for start in range(0, x.shape[0], batch_size):
        out.append(
            model(x[start:start + batch_size].to(device)).float().cpu().numpy()
        )
    return {"full": np.concatenate(out)}


def shape_bias(
    logits: np.ndarray, shapes: list[str], textures: list[str],
    class_index: dict[str, int],
) -> dict[str, float]:
    """Restrict decisions to the 16 categories, then score shape vs texture."""
    cols = [class_index[c] for c in GEIRHOS_CLASSES]
    restricted = logits[:, cols]
    decided = np.asarray(GEIRHOS_CLASSES, dtype=object)[
        restricted.argmax(axis=1)
    ]
    n_shape = int(np.sum(decided == np.asarray(shapes, dtype=object)))
    n_texture = int(np.sum(decided == np.asarray(textures, dtype=object)))
    denom = n_shape + n_texture
    return {
        "n_images": int(len(shapes)),
        "n_shape": n_shape,
        "n_texture": n_texture,
        "n_other": int(len(shapes) - denom),
        "shape_bias": (n_shape / denom) if denom else float("nan"),
        "shape_accuracy": n_shape / len(shapes),
        "texture_accuracy": n_texture / len(shapes),
    }


def bootstrap_ci(
    logits: np.ndarray, shapes: list[str], textures: list[str],
    class_index: dict[str, int], n_boot: int = 2000, seed: int = 0,
) -> tuple[float, float]:
    """Percentile CI for shape bias, resampling cue-conflict items."""
    rng = np.random.default_rng(seed)
    shapes_a = np.asarray(shapes, dtype=object)
    textures_a = np.asarray(textures, dtype=object)
    n = len(shapes)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        res = shape_bias(
            logits[idx], list(shapes_a[idx]), list(textures_a[idx]),
            class_index,
        )
        if not np.isnan(res["shape_bias"]):
            vals.append(res["shape_bias"])
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))



def write_report(logits, shapes, textures, class_index, args) -> None:
    """Score every condition, print it, and persist JSON + raw logits."""
    report = {"model_name": args.model_name,
              "checkpoint": str(args.checkpoint), "conditions": {}}
    for cond, lg in logits.items():
        res = shape_bias(lg, shapes, textures, class_index)
        lo, hi = bootstrap_ci(lg, shapes, textures, class_index,
                              n_boot=args.n_boot, seed=args.seed)
        res["shape_bias_ci95"] = [lo, hi]
        report["conditions"][cond] = res
        print(f"  {cond:11s} shape_bias={res['shape_bias']:.3f} "
              f"[{lo:.3f}, {hi:.3f}]  shape_acc={res['shape_accuracy']:.3f}",
              flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.model_name}.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    np.savez(
        os.path.join(args.out_dir, f"{args.model_name}_logits.npz"),
        shapes=np.asarray(shapes), textures=np.asarray(textures),
        **logits,
    )
    print(f"saved -> {out_path}", flush=True)


def main():
    from src.analysis.encoding import dkl_frontend, features
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--model-name", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--stream-concat", action="store_true")
    p.add_argument("--stream1-weight", type=float, default=0.5)
    p.add_argument("--no-lum-blur", action="store_true")
    p.add_argument("--no-lum-cs", action="store_true")
    p.add_argument("--transform-target", default="magno",
                   choices=["magno", "parvo"])
    p.add_argument("--age-months", type=int, default=48)
    p.add_argument("--dvd-age-months", type=float, default=None,
                   help="DVD magno diet at this age instead of blur + CS "
                        "(fs2_dvd* runs: 300, their dvd_end_months)")
    p.add_argument("--baseline", action="store_true",
                   help="checkpoint is a single-stream ResNet baseline")
    # resnet18, not resnet50: every checkpoint in this project is resnet18
    # (see any run's config.json), and the two-stream path below does not pass
    # this through at all -- it takes features.build_model's own resnet18
    # default. So the old resnet50 default was inert on the main path and
    # wrong on --baseline, which is the worst combination: silently ignored
    # where you look, silently wrong where you don't.
    p.add_argument("--like-resnet", default="resnet18",
                   help="architecture for --baseline only; the two-stream "
                        "path always uses features.build_model's default")
    p.add_argument("--legacy-mixing", action="store_true",
                   help="reproduce pre-fix asymmetric mixing (old ckpts)")
    p.add_argument("--parvo-highsf", action="store_true",
                   help="biologically corrected parvo input: stream 2 gets "
                        "untransformed luminance + rg (2 channels). Sets "
                        "in_channels=(1,2); required for the "
                        "dkl_bcs_parvo_highsf checkpoints and wrong for "
                        "every other condition.")
    p.add_argument("--rgb-mode", action="store_true",
                   help="rgb_tsn architecture-only control: both streams get "
                        "the identical 3-channel image, in_channels=(3,3). "
                        "Despite the name the channels are DKL lum/rg/by -- "
                        "every training condition read the DKL h5.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--allow-partial", action="store_true",
                   help="probe a still-training checkpoint on purpose")
    args = p.parse_args()
    # An fs2 run name -> its checkpoint; refuses flags that are not the
    # checkpoint's own front end (blur + CS on a DVD-diet run).
    fe = dkl_frontend.frontend_from_args(args)

    # Same guard as stream_shape_texture.py: `checkpoint_last.pt` is
    # overwritten every epoch, so reading one mid-run yields a partially
    # trained network whose report is indistinguishable from a converged one.
    # That already cost the project a retracted result once (PAPER_PLAN §10.1)
    # and this entry point was missing the check.
    if args.checkpoint:
        features.assert_converged(args.checkpoint, args.allow_partial)

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    class_index = load_class_index()
    items = list_cue_conflict_images()
    if not items:
        raise RuntimeError(f"no cue-conflict images under {CUE_CONFLICT_DIR}")
    paths = [i[0] for i in items]
    shapes = [i[1] for i in items]
    textures = [i[2] for i in items]
    print(f"[{args.model_name}] {len(items)} cue-conflict images", flush=True)

    if args.baseline:
        # Single-stream anchor: no DKL front end, no stream decomposition.
        model = build_baseline(
            args.checkpoint, like_resnet=args.like_resnet,
            seed=args.seed, device=device,
        )
        parts: dict[str, list[np.ndarray]] = {}
        for start in range(0, len(paths), 128):
            rgb = load_images(paths[start:start + 128])
            for k, v in baseline_logits(model, rgb, device=device).items():
                parts.setdefault(k, []).append(v)
            print(f"  {min(start + 128, len(paths))}/{len(paths)}", flush=True)
        write_report(
            {k: np.concatenate(v) for k, v in parts.items()},
            shapes, textures, class_index, args,
        )
        return

    ch_min, ch_max = dkl_frontend.compute_dkl_channel_minmax()
    model = features.build_model(
        checkpoint=args.checkpoint,
        stream_concat=args.stream_concat,
        stream1_weight=args.stream1_weight,
        legacy_asymmetric_mixing=args.legacy_mixing,
        seed=args.seed,
        device=device,
        in_channels=stream_in_channels(args),
    )

    all_logits: dict[str, list[np.ndarray]] = {}
    chunk = 128
    for start in range(0, len(paths), chunk):
        rgb = load_images(paths[start : start + chunk])
        x_m, x_p = dkl_frontend.magno_parvo_inputs(
            rgb, ch_min, ch_max, device=device, **fe)
        got = stream_logits(model, x_m, x_p, device=device)
        for k, v in got.items():
            all_logits.setdefault(k, []).append(v)
        print(f"  {min(start + chunk, len(paths))}/{len(paths)}", flush=True)
    logits = {k: np.concatenate(v) for k, v in all_logits.items()}

    write_report(logits, shapes, textures, class_index, args)


if __name__ == "__main__":
    main()
