#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create *new* cue-conflict datasets by merging RGB images (.png) with their corresponding
single-channel depth maps (.tiff), using multiple "attention-guided" fusion strategies.

Assumptions (matches your screenshots):
- rgb_root has structure:  cue-conflict/<class_name>/*.png
- depth_root has structure: cue-conflict-depth_anything_v2_depth_only/<class_name>/*.tiff
- For each RGB file:   elephant9-airplane2.png
  the corresponding depth file is: elephant9-airplane2.tiff
  (same basename, different extension, same class subdir)

Depth convention:
- Your depth maps: black = near, white = far
- We invert to "attention/near-ness": near = 1, far = 0   via: depth_near = 1 - depth_norm

Outputs:
- out_root/<method>/<class_name>/<basename>.png  (most methods)
- out_root/rgbd_npz/<class_name>/<basename>.npz  (RGBD 4ch stored losslessly)
Optionally saves:
- out_root/<method>__mask/<class>/<basename>_mask.png
- out_root/<method>__depth/<class>/<basename>_depth.png

Usage examples:
1) Run a bunch of methods at once:
python scripts/generate_depth_attention_cue_conflict.py \
    --rgb_root datasets/cue-conflict \
    --depth_root datasets/cue-conflict-depth_anything_v2_depth_only \
    --methods binary_fg,weighted_mul,weighted_blend,log_weighted,soft_fg,depth_of_field,edge_overlay,depth_overlay,rgbd_npz \
    --bg_mode white \
    --alpha 0.6 --beta 8.0 --fg_percentile 80 --sigmoid_k 12  --blur_sigma 4 \
    # --save_mask --save_depth \
    #  --out_root datasets/cue-conflict_rgb_depth_fusions \

2) Only foreground masking (hard threshold using Otsu):
   python build_rgb_depth_datasets.py --rgb_root cue-conflict --depth_root cue-conflict-depth_anything_v2_depth_only \
     --out_root out --methods binary_fg --binary_mode otsu --bg_mode mean

Dependencies:
- numpy
- pillow
- imageio (for TIFF robustness)
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter

try:
    import imageio.v3 as iio  # robust TIFF read
except Exception as e:
    iio = None


# ----------------------------
# Utils: I/O
# ----------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def load_rgb01(path: Path) -> np.ndarray:
    """RGB in [0,1], float32, shape (H,W,3)."""
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def load_depth01(path: Path) -> np.ndarray:
    """
    Load depth as float32 in [0,1], shape (H,W).
    Supports common TIFF types (uint8/uint16/float).
    """
    # Prefer imageio for TIFF; fall back to PIL.
    if iio is not None:
        d = iio.imread(str(path))
        d = np.asarray(d)
    else:
        d = np.asarray(Image.open(path))

    # If depth is HxWxC, keep first channel (some TIFF writers store as 3ch)
    if d.ndim == 3:
        d = d[..., 0]

    d = d.astype(np.float32)

    # Robust normalization:
    # - if already [0,1] float: keep
    # - if integer: divide by max
    # - else: min-max
    if np.issubdtype(d.dtype, np.floating):
        # could be any scale; min-max
        dmin, dmax = float(np.nanmin(d)), float(np.nanmax(d))
        if dmax - dmin > 1e-12:
            d = (d - dmin) / (dmax - dmin)
        else:
            d = np.zeros_like(d, dtype=np.float32)
    else:
        # integer types: normalize by dtype max
        maxv = float(np.iinfo(d.dtype).max)
        d = d / maxv

    d = np.clip(d, 0.0, 1.0)
    return d.astype(np.float32)


def save_png01(path: Path, x01: np.ndarray) -> None:
    """Save float image [0,1] as uint8 PNG."""
    x = np.clip(x01 * 255.0 + 0.5, 0, 255).astype(np.uint8)
    if x.ndim == 2:
        img = Image.fromarray(x, mode="L")
    else:
        img = Image.fromarray(x, mode="RGB")
    ensure_dir(path.parent)
    img.save(path)


def pil_from_rgb01(rgb01: np.ndarray) -> Image.Image:
    x = np.clip(rgb01 * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return Image.fromarray(x, mode="RGB")


def rgb01_from_pil(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return arr


# ----------------------------
# Attention / Mask builders
# ----------------------------

def invert_depth_to_near(depth01: np.ndarray) -> np.ndarray:
    """Given black=near (0), white=far (1), convert to near-ness attention: near=1."""
    return np.clip(1.0 - depth01, 0.0, 1.0)


def otsu_threshold(x01: np.ndarray, nbins: int = 256) -> float:
    """Otsu threshold for x in [0,1]. Returns threshold in [0,1]."""
    x = np.clip(x01, 0.0, 1.0).ravel()
    hist, bin_edges = np.histogram(x, bins=nbins, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    p = hist / (hist.sum() + 1e-12)

    omega = np.cumsum(p)
    mu = np.cumsum(p * (bin_edges[:-1] + bin_edges[1:]) * 0.5)
    mu_t = mu[-1]

    sigma_b2 = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega) + 1e-12)
    k = int(np.nanargmax(sigma_b2))
    # threshold at bin center
    thr = float((bin_edges[k] + bin_edges[k + 1]) * 0.5)
    return float(np.clip(thr, 0.0, 1.0))


def make_attention_map(
    depth_near01: np.ndarray,
    method: str,
    *,
    beta: float = 8.0,
    fg_percentile: float = 80.0,
    binary_mode: str = "otsu",
    fixed_thresh: float = 0.5,
    sigmoid_k: float = 12.0,
) -> np.ndarray:
    """
    Build an attention map A in [0,1] from depth_near01 (near=1).
    """
    d = np.clip(depth_near01, 0.0, 1.0)

    if method == "linear":
        A = d

    elif method == "log":
        # compress far/boost near using log1p
        # A = log(1 + beta*d)/log(1+beta)
        A = np.log1p(beta * d) / np.log1p(beta)

    elif method == "gamma":
        # emphasize near with power law (gamma < 1 boosts near; gamma > 1 suppresses)
        # here we map beta -> gamma in a simple way: gamma = 1/beta (bounded)
        gamma = float(np.clip(1.0 / max(beta, 1e-6), 0.2, 3.0))
        A = np.power(d, gamma)

    elif method == "binary":
        if binary_mode == "otsu":
            thr = otsu_threshold(d)
        elif binary_mode == "percentile":
            thr = float(np.percentile(d, fg_percentile))
        elif binary_mode == "fixed":
            thr = float(fixed_thresh)
        else:
            raise ValueError(f"Unknown binary_mode: {binary_mode}")
        A = (d >= thr).astype(np.float32)

    elif method == "soft":
        # soft foreground around a threshold (sigmoid)
        thr = float(np.percentile(d, fg_percentile))
        A = 1.0 / (1.0 + np.exp(-sigmoid_k * (d - thr)))

    elif method == "bandpass":
        # “mid-depth focus”: highlights middle range, sometimes useful to test failures
        # center at percentile; width controlled by beta
        c = float(np.percentile(d, fg_percentile))
        sigma = 0.2 / max(beta, 1e-6) + 1e-3
        A = np.exp(-0.5 * ((d - c) / sigma) ** 2).astype(np.float32)

    else:
        raise ValueError(f"Unknown attention method: {method}")

    return np.clip(A.astype(np.float32), 0.0, 1.0)


# ----------------------------
# Fusion methods (RGB + A/depth)
# ----------------------------

def get_background_rgb01(
    rgb01: np.ndarray,
    *,
    bg_mode: str = "black",
    blur_sigma: float = 3.0,
) -> np.ndarray:
    if bg_mode == "black":
        return np.zeros_like(rgb01, dtype=np.float32)
    if bg_mode == "white":
        return np.ones_like(rgb01, dtype=np.float32)
    if bg_mode == "mean":
        m = float(rgb01.mean())
        return np.full_like(rgb01, m, dtype=np.float32)
    if bg_mode == "blur":
        # quick blur via PIL
        pil = pil_from_rgb01(rgb01)
        # PIL GaussianBlur uses radius roughly analogous to sigma
        blurred = pil.filter(ImageFilter.GaussianBlur(radius=float(blur_sigma)))
        return rgb01_from_pil(blurred)
    raise ValueError(f"Unknown bg_mode: {bg_mode}")


def fuse_rgb_depth(
    rgb01: np.ndarray,
    depth_near01: np.ndarray,
    *,
    fusion: str,
    alpha: float = 0.6,
    beta: float = 8.0,
    fg_percentile: float = 80.0,
    binary_mode: str = "otsu",
    fixed_thresh: float = 0.5,
    sigmoid_k: float = 12.0,
    bg_mode: str = "black",
    blur_sigma: float = 3.0,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Returns: (fused_rgb01, maskA01_or_None, depth_near01_or_None)
    """
    d = np.clip(depth_near01, 0.0, 1.0).astype(np.float32)

    # Choose the attention map builder depending on fusion method
    # (some fusions want linear vs log vs binary etc.)
    if fusion in {"weighted_mul", "weighted_blend", "depth_overlay", "edge_overlay", "depth_of_field", "foreground_keep"}:
        # Default attention is "linear" unless fusion implies a different one.
        A = make_attention_map(d, "linear", beta=beta, fg_percentile=fg_percentile,
                               binary_mode=binary_mode, fixed_thresh=fixed_thresh, sigmoid_k=sigmoid_k)
    elif fusion == "log_weighted":
        A = make_attention_map(d, "log", beta=beta, fg_percentile=fg_percentile,
                               binary_mode=binary_mode, fixed_thresh=fixed_thresh, sigmoid_k=sigmoid_k)
    elif fusion == "binary_fg":
        A = make_attention_map(d, "binary", beta=beta, fg_percentile=fg_percentile,
                               binary_mode=binary_mode, fixed_thresh=fixed_thresh, sigmoid_k=sigmoid_k)
    elif fusion == "soft_fg":
        A = make_attention_map(d, "soft", beta=beta, fg_percentile=fg_percentile,
                               binary_mode=binary_mode, fixed_thresh=fixed_thresh, sigmoid_k=sigmoid_k)
    else:
        A = None

    if fusion == "weighted_mul":
        # Multiply RGB by an attention gain: keep near bright, push far dark
        # out = rgb * ( (1-alpha) + alpha*A )
        gain = (1.0 - alpha) + alpha * A[..., None]
        out = np.clip(rgb01 * gain, 0.0, 1.0)

    elif fusion == "weighted_blend":
        # Alpha-composite between RGB and a background using A
        bg = get_background_rgb01(rgb01, bg_mode=bg_mode, blur_sigma=blur_sigma)
        # Use alpha*A as final alpha so you can dial strength
        w = np.clip(alpha * A, 0.0, 1.0)[..., None]
        out = np.clip(rgb01 * w + bg * (1.0 - w), 0.0, 1.0)

    elif fusion == "log_weighted":
        # Like weighted_mul but with log-shaped attention (more aggressive near emphasis)
        gain = (1.0 - alpha) + alpha * A[..., None]
        out = np.clip(rgb01 * gain, 0.0, 1.0)

    elif fusion == "binary_fg":
        # Hard mask foreground; background can be blurred/mean/black/white
        bg = get_background_rgb01(rgb01, bg_mode=bg_mode, blur_sigma=blur_sigma)
        m = A[..., None]
        out = np.clip(rgb01 * m + bg * (1.0 - m), 0.0, 1.0)

    elif fusion == "soft_fg":
        # Soft mask with sigmoid edges
        bg = get_background_rgb01(rgb01, bg_mode=bg_mode, blur_sigma=blur_sigma)
        m = np.clip(A, 0.0, 1.0)[..., None]
        out = np.clip(rgb01 * m + bg * (1.0 - m), 0.0, 1.0)

    elif fusion == "foreground_keep":
        # Keep only foreground content (mask multiply), background forced to black
        m = np.clip(alpha * A, 0.0, 1.0)[..., None]
        out = np.clip(rgb01 * m, 0.0, 1.0)

    elif fusion == "depth_overlay":
        # Overlay depth (grayscale) onto RGB: out = (1-alpha)*rgb + alpha*depth_gray
        depth_gray = d[..., None]
        out = np.clip((1.0 - alpha) * rgb01 + alpha * depth_gray, 0.0, 1.0)

    elif fusion == "edge_overlay":
        # Add depth edges as a luminance cue (top-down-ish “where boundaries are”)
        # Simple Sobel on depth; then add as grayscale
        gx = np.zeros_like(d)
        gy = np.zeros_like(d)
        gx[:, 1:-1] = d[:, 2:] - d[:, :-2]
        gy[1:-1, :] = d[2:, :] - d[:-2, :]
        g = np.sqrt(gx * gx + gy * gy)
        g = g / (g.max() + 1e-12)
        edge = g[..., None]
        out = np.clip(rgb01 + alpha * edge, 0.0, 1.0)
        A = g  # treat edge magnitude as "mask" for saving

    elif fusion == "depth_of_field":
        # Blur background more than foreground (classic “attention / DOF” trick)
        # We build a blurred image and composite with A.
        pil = pil_from_rgb01(rgb01)
        blurred = pil.filter(ImageFilter.GaussianBlur(radius=float(blur_sigma)))
        blur01 = rgb01_from_pil(blurred)
        m = np.clip(A, 0.0, 1.0)[..., None]
        out = np.clip(rgb01 * m + blur01 * (1.0 - m), 0.0, 1.0)

    elif fusion == "rgbd_npz":
        # Not an RGB fusion, handled elsewhere
        out = rgb01
        A = None

    else:
        raise ValueError(f"Unknown fusion method: {fusion}")

    return out.astype(np.float32), (None if A is None else A.astype(np.float32)), d.astype(np.float32)


# ----------------------------
# Dataset traversal + generation
# ----------------------------

@dataclass
class PairItem:
    cls: str
    rgb_path: Path
    depth_path: Path
    stem: str


def collect_pairs(rgb_root: Path, depth_root: Path) -> List[PairItem]:
    """
    Collect matched (rgb, depth) pairs using identical:
      - class subdir name
      - file basename (stem)
    """
    pairs: List[PairItem] = []
    missing_depth: List[Path] = []
    missing_rgb: List[Path] = []

    # index all depth by (cls, stem)
    depth_index: Dict[Tuple[str, str], Path] = {}
    for cls_dir in sorted([p for p in depth_root.iterdir() if p.is_dir()]):
        cls = cls_dir.name
        for dp in sorted(cls_dir.glob("*.tif*")):
            depth_index[(cls, dp.stem)] = dp

    # walk rgb and match
    for cls_dir in sorted([p for p in rgb_root.iterdir() if p.is_dir()]):
        cls = cls_dir.name
        for rp in sorted(cls_dir.iterdir()):
            if not rp.is_file() or rp.suffix.lower() != ".png":
                continue
            key = (cls, rp.stem)
            dp = depth_index.get(key, None)
            if dp is None:
                missing_depth.append(rp)
                continue
            pairs.append(PairItem(cls=cls, rgb_path=rp, depth_path=dp, stem=rp.stem))

    # also find depth without rgb (optional)
    rgb_index = {(p.parent.name, p.stem) for p in rgb_root.rglob("*.png")}
    for (cls, stem), dp in depth_index.items():
        if (cls, stem) not in rgb_index:
            missing_rgb.append(dp)

    if missing_depth:
        print(f"[WARN] {len(missing_depth)} RGB files missing depth counterpart. Showing first 5:")
        for p in missing_depth[:5]:
            print("  -", p)
    if missing_rgb:
        print(f"[WARN] {len(missing_rgb)} depth files missing RGB counterpart. Showing first 5:")
        for p in missing_rgb[:5]:
            print("  -", p)

    print(f"[INFO] Collected {len(pairs)} matched RGB+depth pairs.")
    return pairs


def save_rgbd_npz(out_path: Path, rgb01: np.ndarray, depth_near01: np.ndarray) -> None:
    """
    Save RGB (uint8) + depth_near (float16) into a compact .npz for training pipelines.
    """
    ensure_dir(out_path.parent)
    rgb_u8 = np.clip(rgb01 * 255.0 + 0.5, 0, 255).astype(np.uint8)
    depth_f16 = depth_near01.astype(np.float16)
    np.savez_compressed(str(out_path), rgb=rgb_u8, depth=depth_f16)


def build_datasets(
    pairs: List[PairItem],
    *,
    out_root: Path,
    methods: List[str],
    alpha: float,
    beta: float,
    fg_percentile: float,
    binary_mode: str,
    fixed_thresh: float,
    sigmoid_k: float,
    bg_mode: str,
    blur_sigma: float,
    invert_depth: bool,
    save_mask: bool,
    save_depth: bool,
) -> None:
    ensure_dir(out_root)

    index_rows: List[Dict[str, str]] = []

    for i, item in enumerate(pairs):
        rgb01 = load_rgb01(item.rgb_path)
        depth01 = load_depth01(item.depth_path)

        # Resize depth to match RGB if needed
        if depth01.shape[0] != rgb01.shape[0] or depth01.shape[1] != rgb01.shape[1]:
            depth_pil = Image.fromarray(np.clip(depth01 * 255.0 + 0.5, 0, 255).astype(np.uint8), mode="L")
            depth_pil = depth_pil.resize((rgb01.shape[1], rgb01.shape[0]), resample=Image.BILINEAR)
            depth01 = (np.asarray(depth_pil, dtype=np.float32) / 255.0)

        depth_near01 = invert_depth_to_near(depth01) if invert_depth else np.clip(depth01, 0.0, 1.0)

        for m in methods:
            if m == "rgbd_npz":
                out_path = out_root / "rgbd_npz" / item.cls / f"{item.stem}.npz"
                save_rgbd_npz(out_path, rgb01, depth_near01)
                index_rows.append({
                    "method": m,
                    "class": item.cls,
                    "stem": item.stem,
                    "rgb_path": str(item.rgb_path),
                    "depth_path": str(item.depth_path),
                    "out_path": str(out_path),
                })
                continue

            fused01, maskA01, depth_used01 = fuse_rgb_depth(
                rgb01,
                depth_near01,
                fusion=m,
                alpha=alpha,
                beta=beta,
                fg_percentile=fg_percentile,
                binary_mode=binary_mode,
                fixed_thresh=fixed_thresh,
                sigmoid_k=sigmoid_k,
                bg_mode=bg_mode,
                blur_sigma=blur_sigma,
            )

            out_path = out_root / m / item.cls / f"{item.stem}.png"
            save_png01(out_path, fused01)

            if save_mask and maskA01 is not None:
                mask_path = out_root / f"{m}__mask" / item.cls / f"{item.stem}_mask.png"
                save_png01(mask_path, maskA01)

            if save_depth and depth_used01 is not None:
                depth_path = out_root / f"{m}__depth" / item.cls / f"{item.stem}_depth.png"
                save_png01(depth_path, depth_used01)

            index_rows.append({
                "method": m,
                "class": item.cls,
                "stem": item.stem,
                "rgb_path": str(item.rgb_path),
                "depth_path": str(item.depth_path),
                "out_path": str(out_path),
            })

        if (i + 1) % 200 == 0:
            print(f"[INFO] Processed {i+1}/{len(pairs)} pairs...")

    # write an index CSV (nice for training scripts)
    index_csv = out_root / "index.csv"
    with index_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["method", "class", "stem", "rgb_path", "depth_path", "out_path"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in index_rows:
            w.writerow(r)

    print(f"[DONE] Wrote index: {index_csv}")
    print(f"[DONE] Output root: {out_root}")


# ----------------------------
# CLI
# ----------------------------

def parse_methods(s: str) -> List[str]:
    ms = [x.strip() for x in s.split(",") if x.strip()]
    # A curated menu of “attention-ish” fusions (plus rgbd_npz for true RGBD training)
    allowed = {
        "weighted_mul",      # rgb * gain(A)
        "weighted_blend",    # rgb composited with background using A
        "log_weighted",      # log-shaped A to emphasize near
        "binary_fg",         # hard foreground mask
        "soft_fg",           # sigmoid soft mask
        "foreground_keep",   # multiply by mask, background black
        "depth_overlay",     # grayscale depth overlay on RGB
        "edge_overlay",      # depth-edge overlay
        "depth_of_field",    # blur background with A
        "rgbd_npz",          # store RGB+depth as 4ch in npz
    }
    bad = [m for m in ms if m not in allowed]
    if bad:
        raise ValueError(f"Unknown method(s): {bad}\nAllowed: {sorted(list(allowed))}")
    return ms

import re
import hashlib
from datetime import datetime

def _slug(s: str, maxlen: int = 80) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9._+=-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:maxlen] if maxlen > 0 else s

def auto_out_root_from_args(args, methods: List[str]) -> Path:
    """
    Build an informative + unique output directory name from args.
    Example:
      cue-conflict__depth_anything_v2__m-weighted_mul+binary_fg__a0.6__b8__p80__bg-blur__sig12__bs3__inv1__h3f9a1c2d
    """
    rgb_name = Path(args.rgb_root).name
    depth_name = Path(args.depth_root).name

    # keep method list readable but not too long
    mtag = "+".join(methods)
    mtag = _slug(mtag, maxlen=80)

    # key knobs that change outputs
    parts = [
        # _slug(rgb_name, 40),
        # _slug(depth_name, 40),
        f"cue_conflict_attention_depth_guide_avg",
        # f"m_{mtag}",
        f"bg_{args.bg_mode}",
        f"a{args.alpha:g}",
        f"b{args.beta:g}",
        f"p{args.fg_percentile:g}",
        f"sig{args.sigmoid_k:g}",
        f"bs{args.blur_sigma:g}",
        f"bin_{args.binary_mode}",
        f"thr{args.fixed_thresh:g}" if args.binary_mode == "fixed" else None,
        f"inv{0 if args.no_invert_depth else 1}",
    ]
    parts = [p for p in parts if p is not None]

    # add a short hash to avoid collisions (also helps if method list gets truncated)
    # canonical = "|".join([
    #     str(args.rgb_root),
    #     str(args.depth_root),
    #     ",".join(methods),
    #     f"alpha={args.alpha}",
    #     f"beta={args.beta}",
    #     f"fgp={args.fg_percentile}",
    #     f"sigmoid_k={args.sigmoid_k}",
    #     f"binary_mode={args.binary_mode}",
    #     f"fixed_thresh={args.fixed_thresh}",
    #     f"bg_mode={args.bg_mode}",
    #     f"blur_sigma={args.blur_sigma}",
    #     f"invert={0 if args.no_invert_depth else 1}",
    # ])
    # h = hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:8]
    # parts.append(f"h{h}")

    # choose a parent folder (default: alongside rgb_root)
    parent = Path(args.rgb_root).resolve().parent
    out_dirname = "_".join(parts)
    return parent / out_dirname


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb_root", type=str, required=True,
                    help="Root of RGB cue-conflict dataset (class subdirs, .png files).")
    ap.add_argument("--depth_root", type=str, required=True,
                    help="Root of depth dataset (class subdirs, .tiff files).")
    ap.add_argument("--out_root", type=str, default=None,
                help="Optional. If not set, auto-generate based on args.")
    ap.add_argument("--methods", type=str, default="weighted_mul,weighted_blend,log_weighted,binary_fg,soft_fg,depth_of_field,edge_overlay,rgbd_npz",
                    help="Comma-separated methods. See script header for meanings.")

    # Common knobs
    ap.add_argument("--alpha", type=float, default=0.6,
                    help="Overall fusion strength (0..1) used by many methods.")
    ap.add_argument("--beta", type=float, default=8.0,
                    help="Shape parameter for log/gamma/bandpass attention (bigger => stronger).")
    ap.add_argument("--fg_percentile", type=float, default=80.0,
                    help="Foreground percentile for percentile threshold and soft_fg.")
    ap.add_argument("--sigmoid_k", type=float, default=12.0,
                    help="Steepness for soft_fg sigmoid.")
    ap.add_argument("--binary_mode", type=str, default="otsu", choices=["otsu", "percentile", "fixed"],
                    help="How to threshold for binary_fg.")
    ap.add_argument("--fixed_thresh", type=float, default=0.5,
                    help="Used if binary_mode=fixed.")
    ap.add_argument("--bg_mode", type=str, default="blur", choices=["black", "white", "mean", "blur"],
                    help="Background used for blending/masking methods.")
    ap.add_argument("--blur_sigma", type=float, default=3.0,
                    help="Blur strength when bg_mode=blur or depth_of_field.")
    ap.add_argument("--no_invert_depth", action="store_true",
                    help="If set, do NOT invert depth. Default is invert (black near -> near=1).")

    ap.add_argument("--save_mask", action="store_true",
                    help="Also save attention/mask images when available.")
    ap.add_argument("--save_depth", action="store_true",
                    help="Also save normalized (near=1) depth images.")

    args = ap.parse_args()

    rgb_root = Path(args.rgb_root)
    depth_root = Path(args.depth_root)
    # out_root = Path(args.out_root)

    methods = parse_methods(args.methods)
    if args.out_root is None:
        out_root = auto_out_root_from_args(args, methods)
        print(f"[INFO] --out_root not set. Auto out_root = {out_root}")
    else:
        out_root = Path(args.out_root)
    methods = parse_methods(args.methods)

    pairs = collect_pairs(rgb_root, depth_root)

    build_datasets(
        pairs,
        out_root=out_root,
        methods=methods,
        alpha=float(args.alpha),
        beta=float(args.beta),
        fg_percentile=float(args.fg_percentile),
        binary_mode=str(args.binary_mode),
        fixed_thresh=float(args.fixed_thresh),
        sigmoid_k=float(args.sigmoid_k),
        bg_mode=str(args.bg_mode),
        blur_sigma=float(args.blur_sigma),
        invert_depth=(not args.no_invert_depth),
        save_mask=bool(args.save_mask),
        save_depth=bool(args.save_depth),
    )


if __name__ == "__main__":
    main()

"""
python scripts/generate_depth_attention_cue_conflict.py     --rgb_root datasets/cue-conflict     --depth_root datasets/cue-conflict-depth_anything_v2_depth_only     --methods weighted_blend     --bg_mode white     --alpha 0.6 --beta 8.0 --fg_percentile 80 --sigmoid_k 12  --blur_sigma 4 
python scripts/generate_depth_attention_cue_conflict.py     --rgb_root datasets/cue-conflict_original     --depth_root datasets/cue-conflict-depth_anything_v2_depth_only     --methods weighted_blend     --bg_mode white     --alpha 1 --beta 8.0 --fg_percentile 80 --sigmoid_k 12  --blur_sigma 4  # worse
python scripts/generate_depth_attention_cue_conflict.py     --rgb_root datasets/cue-conflict_original     --depth_root datasets/cue-conflict-depth_anything_v2_depth_only     --methods weighted_blend     --bg_mode white     --alpha 0.2 --beta 8.0 --fg_percentile 80 --sigmoid_k 12  --blur_sigma 4  # slightly better --> 0.59 --> DVD-B 0.84 ish
python scripts/generate_depth_attention_cue_conflict.py     --rgb_root datasets/cue-conflict_original     --depth_root datasets/cue-conflict-depth_anything_v2_depth_only     --methods weighted_blend     --bg_mode black     --alpha 0.2 --beta 8.0 --fg_percentile 80 --sigmoid_k 12  --blur_sigma 4   # black wose  --> 0.54
python scripts/generate_depth_attention_cue_conflict.py     --rgb_root datasets/cue-conflict_original     --depth_root datasets/cue-conflict-depth_anything_v2_depth_only     --methods weighted_blend     --bg_mode blur     --alpha 0.2 --beta 8.0 --fg_percentile 80 --sigmoid_k 12  --blur_sigma 4  # much better --> 0.74 ! shape bais? --> DVD-B 0.86 # --> magno - parvo mixing? 
python scripts/generate_depth_attention_cue_conflict.py     --rgb_root datasets/cue-conflict_original     --depth_root datasets/cue-conflict-depth_anything_v2_depth_only     --methods weighted_blend     --bg_mode mean     --alpha 0.2 --beta 8.0 --fg_percentile 80 --sigmoid_k 12  --blur_sigma 4  # 0.65 --> not bad?  --> DVD-B 0.86
python scripts/generate_depth_attention_cue_conflict.py     --rgb_root datasets/cue-conflict_original     --depth_root datasets/cue-conflict-depth_anything_v2_depth_only     --methods weighted_blend     --bg_mode mean     --alpha 0.5 --beta 8.0 --fg_percentile 80 --sigmoid_k 12  --blur_sigma 4  # 0.58 worse
python scripts/generate_depth_attention_cue_conflict.py     --rgb_root datasets/cue-conflict_original     --depth_root datasets/cue-conflict-depth_anything_v2_depth_only     --methods weighted_blend     --bg_mode mean     --alpha 0.1 --beta 8.0 --fg_percentile 80 --sigmoid_k 12  --blur_sigma 4  # 0.71
python scripts/generate_depth_attention_cue_conflict.py     --rgb_root datasets/cue-conflict_original     --depth_root datasets/cue-conflict-depth_anything_v2_depth_only     --methods weighted_blend     --bg_mode blur     --alpha 0.1 --beta 8.0 --fg_percentile 80 --sigmoid_k 12  --blur_sigma 4  # 0.83 ?! --> amzing shape bias ! --> DVD-B 0.86 | Depth-of-field better, blur with increasing depth? 

/home/student/l/lzejin/codebase/neuro-vs-ai
python scripts/generate_depth_attention_cue_conflict.py     --rgb_root datasets/cue-conflict_original     --depth_root datasets/cue-conflict-depth_anything_v2_depth_only     --methods depth_of_field     --bg_mode blur     --alpha 0.1 --beta 8.0 --fg_percentile 80 --sigmoid_k 12  --blur_sigma 4  # 0.52
python scripts/generate_depth_attention_cue_conflict.py     --rgb_root datasets/cue-conflict_original     --depth_root datasets/cue-conflict-depth_anything_v2_depth_only     --methods weighted_blend     --bg_mode blur     --alpha 0 --beta 8.0 --fg_percentile 80 --sigmoid_k 12  --blur_sigma 4  # 0.85 ?! --> so basically making the whole cue-conlict sigma 4 blur --> give you 0.85 shape bias?

"""