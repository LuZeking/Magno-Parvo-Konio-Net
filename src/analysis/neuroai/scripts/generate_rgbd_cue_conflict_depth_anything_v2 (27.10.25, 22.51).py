#!/usr/bin/env python3
"""
Generate RGB-D (or depth-only) files for the cue-conflict dataset with the *same*
directory structure and basenames, writing to a new root directory.

Depth is predicted with Depth Anything V2 (relative), robustly normalized
per-image (1–99% quantiles) then inverted so 0=near, 1=far.

Each output is a *single file per image*.

Supported output formats (choose with --format):
  - npy (default):
      * RGB-D: one (H,W,4) uint16 array with channels [R,G,B,D].
          - R,G,B are original 8-bit values upscaled to 16-bit (x257)
          - D is relative depth in [0..65535]
      * depth-only: one (H,W) uint16 array of depth.
      * Extension: .npy
  - npz:
      * RGB-D: compressed NumPy archive with keys:
          - 'rgb' (uint8, H,W,3)
          - 'depth' (uint16, H,W)
      * depth-only: keys:
          - 'depth' (uint16, H,W)
      * Extension: .npz
  - tiff:
      * RGB-D: 16-bit 4-channel TIFF (requires tifffile). Stored as uint16 RGBD.
      * depth-only: single-channel 16-bit TIFF (uint16 depth).
      * Extension: .tiff
  - png4:
      * RGB-D: 16-bit 4-channel PNG (requires imageio>=2.31). RGBA where A=depth.
        Viewers may treat the fourth channel as alpha. Use for programmatic IO.
      * depth-only: single-channel 16-bit PNG (uint16 depth).
      * Extension: .png

Examples
--------
python scripts/generate_rgbd_cue_conflict_depth_anything_v2.py \
  --in-root datasets/cue-conflict-copy2 \
  --out-root datasets/cue-conflict-depth_anything_v2 \
  --format png4 --batch-size 512 --save-sanity 12 \
  --sanity-dir results/rgbd_cue-conflict_sanity

python scripts/generate_rgbd_cue_conflict_depth_anything_v2.py \
  --in-root datasets/cue-conflict-copy2 \
  --out-root datasets/cue-conflict-depth_anything_v2_depth_only \
  --format png4 --batch-size 512 --save-sanity 12 --depth-only \
  --sanity-dir results/rgbd_cue-conflict_sanity_depth_only

Notes
-----
- Keeps *exact* subfolder structure and basenames. Only the extension changes.
- Uses relative Depth Anything V2 Large. Switch model id via --model if needed.
- If you have varied image sizes, batching still works; resizing is per-sample.
"""

import os
import sys
import gc
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F

from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# =========================
# Fast console printing
# =========================
def fast_print(*args):
    sys.stdout.write(" ".join(map(str, args)) + "\n"); sys.stdout.flush()

print = fast_print  # type: ignore

# =========================
# I/O helpers
# =========================
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def list_images(root: Path) -> List[Path]:
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)
    files.sort()
    return files


def load_rgb_u8(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)
    return arr  # (H,W,3) uint8


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# =========================
# Saving helpers
# =========================

def to_uint16_rgb(rgb_u8: np.ndarray) -> np.ndarray:
    # Expand 8-bit to 16-bit by multiplying by 257 to fill full 16-bit range
    return (rgb_u8.astype(np.uint16) * 257).astype(np.uint16)


def depth01_to_u16(d01: np.ndarray) -> np.ndarray:
    return np.clip(np.round(d01 * 65535.0), 0, 65535).astype(np.uint16)


def save_depth_only(
    d01: np.ndarray,
    out_path: Path,
    fmt: str,
) -> None:
    """
    Save ONLY the depth channel.
    """
    fmt = fmt.lower()
    d16 = depth01_to_u16(d01)

    if fmt == "npy":
        # Just (H,W) uint16
        np.save(out_path.with_suffix(".npy"), d16)

    elif fmt == "npz":
        # Only 'depth'
        np.savez_compressed(out_path.with_suffix(".npz"), depth=d16)

    elif fmt == "tiff":
        try:
            import tifffile as tiff
        except Exception as e:
            raise RuntimeError("tiff format requires 'tifffile' package") from e
        # Single-channel uint16 TIFF
        tiff.imwrite(
            out_path.with_suffix(".tiff"),
            d16,
            dtype=np.uint16,
            photometric="minisblack",
        )

    elif fmt == "png4":
        try:
            import imageio.v3 as iio
        except Exception as e:
            raise RuntimeError("png4 format requires 'imageio' >= 2.31") from e
        # Save a single-channel 16-bit PNG. imageio will write mode 'I;16'
        iio.imwrite(out_path.with_suffix(".png"), d16)

    else:
        raise ValueError(f"Unknown format: {fmt}")


def save_rgbd(
    rgb_u8: np.ndarray,
    d01: np.ndarray,
    out_path: Path,
    fmt: str,
) -> None:
    """
    Save full RGB-D.
    """
    fmt = fmt.lower()
    if fmt == "npy":
        rgb16 = to_uint16_rgb(rgb_u8)
        d16 = depth01_to_u16(d01)
        rgbd = np.dstack([rgb16, d16[..., None]])  # (H,W,4) uint16
        np.save(out_path.with_suffix(".npy"), rgbd)

    elif fmt == "npz":
        d16 = depth01_to_u16(d01)
        np.savez_compressed(out_path.with_suffix(".npz"), rgb=rgb_u8, depth=d16)

    elif fmt == "tiff":
        try:
            import tifffile as tiff
        except Exception as e:
            raise RuntimeError("tiff format requires 'tifffile' package") from e
        rgb16 = to_uint16_rgb(rgb_u8)
        d16 = depth01_to_u16(d01)
        rgbd = np.dstack([rgb16, d16[..., None]])  # (H,W,4)
        tiff.imwrite(out_path.with_suffix(".tiff"), rgbd)

    elif fmt == "png4":
        try:
            import imageio.v3 as iio
        except Exception as e:
            raise RuntimeError("png4 format requires 'imageio' >= 2.31") from e
        rgb16 = to_uint16_rgb(rgb_u8)
        d16 = depth01_to_u16(d01)
        rgba16 = np.dstack([rgb16, d16[..., None]])  # RGBA16, A=depth
        iio.imwrite(out_path.with_suffix(".png"), rgba16)

    else:
        raise ValueError(f"Unknown format: {fmt}")


def save_sanity(rgb01: np.ndarray, d01: np.ndarray, out_path: Path, dpi: int = 150) -> None:
    """
    Make a side-by-side sanity check image where:
    - the RGB and depth panels have IDENTICAL draw area size
    - the depth colorbar is inset, so it doesn't shrink the axis
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    import numpy as np

    # 2 panels, equal width. We'll handle colorbar manually inside depth axis.
    fig = plt.figure(figsize=(6.5, 3.25), dpi=dpi)
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1, 1], wspace=0.05)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # --- RGB panel ---
    ax1.imshow(np.clip(rgb01, 0, 1))
    ax1.set_title("RGB")
    ax1.axis("off")

    # --- Depth panel ---
    im = ax2.imshow(d01, cmap="magma_r", vmin=0.0, vmax=1.0, origin="upper")
    ax2.set_title("Relative Depth (0=near,1=far)")
    ax2.axis("off")

    # Inset colorbar INSIDE ax2 so it doesn't change layout
    cax = inset_axes(
        ax2,
        width="3%",          # 3% of parent width
        height="60%",        # 60% of parent height
        loc="center right",
        borderpad=1.0,
    )
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label("relative", rotation=90)
    cbar.ax.tick_params(labelsize=6)
    cbar.ax.yaxis.set_offset_position("left")

    # tighter layout without letting colorbar steal space
    plt.tight_layout(pad=1.0, w_pad=0.5)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)


# =========================
# Depth Anything V2 (relative) inference
# =========================
@torch.inference_mode()
def infer_da2_relative_batch(
    batch_u8: List[np.ndarray],
    processor,
    model,
    device: torch.device,
    autocast_dtype: torch.dtype,
    orig_sizes: List[Tuple[int, int]],
) -> List[np.ndarray]:
    """
    Args:
        batch_u8: list of (H,W,3) uint8 arrays
        orig_sizes: list of (H,W) tuples, per image
    Returns:
        List of (H,W) float32 arrays in [0,1], 0=near,1=far
    """
    inputs = processor(images=batch_u8, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(memory_format=torch.channels_last)
    if device.type == "cuda":
        pixel_values = pixel_values.pin_memory().to(device, non_blocking=True)
    else:
        pixel_values = pixel_values.to(device)

    ctx = (
        torch.autocast(device_type=device.type, dtype=autocast_dtype)
        if device.type == "cuda"
        else torch.cuda.amp.autocast(enabled=False)
    )
    with ctx:
        out = model(pixel_values=pixel_values)
        d = out.predicted_depth.to(torch.float32)  # [B,h,w]

    # Resize individually to each original size; then robust normalize and invert
    B = d.shape[0]
    results: List[np.ndarray] = []
    for i in range(B):
        H, W = orig_sizes[i]
        di = d[i:i+1, None, ...]  # [1,1,h,w]
        di = F.interpolate(di, size=(H, W), mode="bicubic", align_corners=False)[0, 0]

        flat = di.flatten()
        q01 = torch.quantile(flat, 0.01)
        q99 = torch.quantile(flat, 0.99)
        q01 = torch.clamp(q01, min=1e-6)
        q99 = torch.maximum(q99, q01 + 1e-6)
        rel = torch.clamp((di - q01) / (q99 - q01), 0.0, 1.0)
        rel = 1.0 - rel  # 0=near, 1=far

        results.append(rel.detach().cpu().numpy().astype(np.float32))

    return results


# =========================
# Main processing
# =========================
def main():
    parser = argparse.ArgumentParser(
        description="Generate RGB-D or depth-only files for cue-conflict with DA2"
    )
    parser.add_argument(
        "--in-root",
        type=Path,
        required=True,
        help="Input dataset root (e.g., datasets/cue-conflict)",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        required=True,
        help="Output root (e.g., datasets/cue-conflict-depth_anything_v2)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="npy",
        choices=["npy", "npz", "tiff", "png4"],
        help="Output file format",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=96,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="depth-anything/Depth-Anything-V2-Large-hf",
        help="HF model id",
    )
    parser.add_argument(
        "--use-bf16",
        action="store_true",
        help="Use bfloat16 autocast when available",
    )
    parser.add_argument(
        "--save-sanity",
        type=int,
        default=10,
        help="Save N sanity images per (sub)folder; 0 to disable",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs",
    )
    parser.add_argument(
        "--sanity-dir",
        type=Path,
        default=Path("results/rgbd_cue-conflict_sanity"),
        help="Directory to save sanity images",
    )
    parser.add_argument(
        "--depth-only",
        action="store_true",
        help="If set, save only depth (no RGB).",
    )

    args = parser.parse_args()

    in_root: Path = args.in_root
    out_root: Path = args.out_root
    fmt: str = args.format.lower()
    BATCH_SIZE: int = args.batch_size
    MODEL_ID: str = args.model
    DEPTH_ONLY: bool = args.depth_only

    assert in_root.exists(), f"Input root not found: {in_root}"
    ensure_dir(out_root)

    # Device & CUDA settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    autocast_dtype = (
        torch.bfloat16
        if (args.use_bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        else (torch.float16 if device.type == "cuda" else torch.float32)
    )

    # Load model
    print(f"Loading Depth Anything V2: {MODEL_ID}")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID, use_fast=True)
    model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID).to(device)
    model.eval().to(memory_format=torch.channels_last)
    try:
        model = torch.compile(model, mode="max-autotune")
        print("torch.compile enabled.")
    except Exception as e:
        print(f"torch.compile not used: {e}")

    # Enumerate input images
    all_imgs = list_images(in_root)
    if not all_imgs:
        print("No images found under", in_root)
        return

    print(f"Found {len(all_imgs)} images.")

    # Group images by their immediate subdirectory to mirror sanity saves
    rel_to_class = {}
    for p in all_imgs:
        rel = p.relative_to(in_root)
        class_dir = rel.parts[0] if len(rel.parts) > 1 else ""
        rel_to_class.setdefault(class_dir, []).append(p)

    for class_dir, paths in rel_to_class.items():
        print(f"\n=== Processing folder: {class_dir or '(root)'} | {len(paths)} images ===")
        sanity_count = 0
        num_to_save = int(args.save_sanity)

        sanity_dir = args.sanity_dir / (class_dir if class_dir else "root")
        os.makedirs(sanity_dir, exist_ok=True)
        if num_to_save > 0:
            ensure_dir(sanity_dir)

        # Process in batches
        for i in tqdm(range(0, len(paths), BATCH_SIZE), desc=class_dir or "root"):
            batch_paths = paths[i : i + BATCH_SIZE]
            batch_imgs = [load_rgb_u8(p) for p in batch_paths]
            orig_sizes = [(im.shape[0], im.shape[1]) for im in batch_imgs]

            # Predict depths (relative 0..1, 0=near)
            d_rel_list = infer_da2_relative_batch(
                batch_imgs, processor, model, device, autocast_dtype, orig_sizes
            )

            # Save outputs mirroring structure and basenames
            for img_path, rgb_u8, d_rel in zip(batch_paths, batch_imgs, d_rel_list):
                rel = img_path.relative_to(in_root)
                out_dir = out_root / rel.parent
                ensure_dir(out_dir)
                out_base = out_dir / rel.stem  # keep basename

                # Skip if exists
                if not args.overwrite:
                    if fmt == "npy" and (out_base.with_suffix(".npy")).exists():
                        continue
                    if fmt == "npz" and (out_base.with_suffix(".npz")).exists():
                        continue
                    if fmt == "tiff" and (out_base.with_suffix(".tiff")).exists():
                        continue
                    if fmt == "png4" and (out_base.with_suffix(".png")).exists():
                        continue

                if DEPTH_ONLY:
                    save_depth_only(d_rel, out_base, fmt)
                else:
                    save_rgbd(rgb_u8, d_rel, out_base, fmt)

                # Sanity images
                # if num_to_save > 0 and sanity_count < num_to_save:
                # save all
                rgb01 = rgb_u8.astype(np.float32) / 255.0
                # default png-style dpi 150
                # save_sanity(rgb01, d_rel, sanity_dir / f"{out_base.name}.png", dpi=150)
                # high-res pdf dpi 300
                save_sanity(rgb01, d_rel, sanity_dir / f"{out_base.name}.pdf", dpi=300)
                sanity_count += 1

            # Housekeeping
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if (i // BATCH_SIZE) % 8 == 0:
                gc.collect()

    print("\nAll done. Output root:", out_root)


if __name__ == "__main__":
    main()

# #!/usr/bin/env python3
# """
# Generate RGB-D (or depth-only) files for the cue-conflict dataset with the *same*
# directory structure and basenames, writing to a new root directory.

# Depth is predicted with Depth Anything V2 (relative), robustly normalized
# per-image (1–99% quantiles) then inverted so 0=near, 1=far.

# Each output is a *single file per image*.

# Supported output formats (choose with --format):
#   - npy (default):
#       * RGB-D: one (H,W,4) uint16 array with channels [R,G,B,D].
#           - R,G,B are original 8-bit values upscaled to 16-bit (x257)
#           - D is relative depth in [0..65535]
#       * depth-only: one (H,W) uint16 array of depth.
#       * Extension: .npy
#   - npz:
#       * RGB-D: compressed NumPy archive with keys:
#           - 'rgb' (uint8, H,W,3)
#           - 'depth' (uint16, H,W)
#       * depth-only: keys:
#           - 'depth' (uint16, H,W)
#       * Extension: .npz
#   - tiff:
#       * RGB-D: 16-bit 4-channel TIFF (requires tifffile). Stored as uint16 RGBD.
#       * depth-only: single-channel 16-bit TIFF (uint16 depth).
#       * Extension: .tiff
#   - png4:
#       * RGB-D: 16-bit 4-channel PNG (requires imageio>=2.31). RGBA where A=depth.
#         Viewers may treat the fourth channel as alpha. Use for programmatic IO.
#       * depth-only: single-channel 16-bit PNG (uint16 depth).
#       * Extension: .png

# Examples
# --------
# python scripts/generate_rgbd_cue_conflict_depth_anything_v2.py \
#   --in-root datasets/cue-conflict-copy2 \
#   --out-root datasets/cue-conflict-depth_anything_v2 \
#   --format tiff --batch-size 512 --save-sanity 12 \
#   --sanity-dir .results/rgbd_cue-conflict_sanity

# python scripts/generate_rgbd_cue_conflict_depth_anything_v2.py \
# --in-root datasets/cue-conflict-copy2 \
# --out-root datasets/cue-conflict-depth_anything_v2_depth_only \
# --format tiff --batch-size 512 --save-sanity 12 --depth-only \
# --sanity-dir .results/rgbd_cue-conflict_sanity_depth_only \

# Notes
# -----
# - Keeps *exact* subfolder structure and basenames. Only the extension changes.
# - Uses relative Depth Anything V2 Large. Switch model id via --model if needed.
# - If you have varied image sizes, batching still works; resizing is per-sample.
# """

# import os
# import sys
# import gc
# import argparse
# from pathlib import Path
# from typing import List, Tuple

# import numpy as np
# from PIL import Image
# from tqdm import tqdm

# import torch
# import torch.nn.functional as F

# from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# # =========================
# # Fast console printing
# # =========================
# def fast_print(*args):
#     sys.stdout.write(" ".join(map(str, args)) + "\n"); sys.stdout.flush()

# print = fast_print  # type: ignore

# # =========================
# # I/O helpers
# # =========================
# IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


# def list_images(root: Path) -> List[Path]:
#     files = []
#     for p in root.rglob("*"):
#         if p.is_file() and p.suffix.lower() in IMG_EXTS:
#             files.append(p)
#     files.sort()
#     return files


# def load_rgb_u8(path: Path) -> np.ndarray:
#     img = Image.open(path).convert("RGB")
#     arr = np.asarray(img, dtype=np.uint8)
#     return arr  # (H,W,3) uint8


# def ensure_dir(p: Path) -> None:
#     p.mkdir(parents=True, exist_ok=True)


# # =========================
# # Saving helpers
# # =========================

# def to_uint16_rgb(rgb_u8: np.ndarray) -> np.ndarray:
#     # Expand 8-bit to 16-bit by multiplying by 257 to fill full 16-bit range
#     return (rgb_u8.astype(np.uint16) * 257).astype(np.uint16)


# def depth01_to_u16(d01: np.ndarray) -> np.ndarray:
#     return np.clip(np.round(d01 * 65535.0), 0, 65535).astype(np.uint16)


# def save_depth_only(
#     d01: np.ndarray,
#     out_path: Path,
#     fmt: str,
# ) -> None:
#     """
#     Save ONLY the depth channel.
#     """
#     fmt = fmt.lower()
#     d16 = depth01_to_u16(d01)

#     if fmt == "npy":
#         # Just (H,W) uint16
#         np.save(out_path.with_suffix(".npy"), d16)

#     elif fmt == "npz":
#         # Only 'depth'
#         np.savez_compressed(out_path.with_suffix(".npz"), depth=d16)

#     elif fmt == "tiff":
#         try:
#             import tifffile as tiff
#         except Exception as e:
#             raise RuntimeError("tiff format requires 'tifffile' package") from e
#         # Single-channel uint16 TIFF
#         # photometric="minisblack" marks it as grayscale (0=black)
#         tiff.imwrite(
#             out_path.with_suffix(".tiff"),
#             d16,
#             dtype=np.uint16,
#             photometric="minisblack",
#         )

#     elif fmt == "png4":
#         try:
#             import imageio.v3 as iio
#         except Exception as e:
#             raise RuntimeError("png4 format requires 'imageio' >= 2.31") from e
#         # Save a single-channel 16-bit PNG. imageio will write mode 'I;16'
#         iio.imwrite(out_path.with_suffix(".png"), d16)

#     else:
#         raise ValueError(f"Unknown format: {fmt}")


# def save_rgbd(
#     rgb_u8: np.ndarray,
#     d01: np.ndarray,
#     out_path: Path,
#     fmt: str,
# ) -> None:
#     """
#     Save full RGB-D.
#     """
#     fmt = fmt.lower()
#     if fmt == "npy":
#         rgb16 = to_uint16_rgb(rgb_u8)
#         d16 = depth01_to_u16(d01)
#         rgbd = np.dstack([rgb16, d16[..., None]])  # (H,W,4) uint16
#         np.save(out_path.with_suffix(".npy"), rgbd)

#     elif fmt == "npz":
#         d16 = depth01_to_u16(d01)
#         np.savez_compressed(out_path.with_suffix(".npz"), rgb=rgb_u8, depth=d16)

#     elif fmt == "tiff":
#         try:
#             import tifffile as tiff
#         except Exception as e:
#             raise RuntimeError("tiff format requires 'tifffile' package") from e
#         rgb16 = to_uint16_rgb(rgb_u8)
#         d16 = depth01_to_u16(d01)
#         rgbd = np.dstack([rgb16, d16[..., None]])  # (H,W,4)
#         # photometric="rgb" is valid for 3-channel RGB. We're giving 4 channels.
#         # Some TIFF readers still accept this for RGB+A style.
#         # We won't try to be fancy with extra tags; just write the array.
#         tiff.imwrite(out_path.with_suffix(".tiff"), rgbd)

#     elif fmt == "png4":
#         try:
#             import imageio.v3 as iio
#         except Exception as e:
#             raise RuntimeError("png4 format requires 'imageio' >= 2.31") from e
#         rgb16 = to_uint16_rgb(rgb_u8)
#         d16 = depth01_to_u16(d01)
#         rgba16 = np.dstack([rgb16, d16[..., None]])  # RGBA16, A=depth
#         iio.imwrite(out_path.with_suffix(".png"), rgba16)

#     else:
#         raise ValueError(f"Unknown format: {fmt}")


# def save_sanity(rgb01: np.ndarray, d01: np.ndarray, out_path: Path) -> None:
#     import matplotlib
#     matplotlib.use("Agg")
#     import matplotlib.pyplot as plt

#     fig = plt.figure(figsize=(6.5, 3.25), dpi=150)

#     ax1 = plt.subplot(1, 2, 1)
#     ax1.imshow(np.clip(rgb01, 0, 1))
#     ax1.set_title("RGB")
#     ax1.axis("off")

#     ax2 = plt.subplot(1, 2, 2)
#     im = ax2.imshow(d01, cmap="magma_r", vmin=0.0, vmax=1.0, origin="upper")
#     ax2.set_title("Relative Depth (0=near,1=far)")
#     ax2.axis("off")
#     cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
#     cbar.set_label("relative", rotation=90)

#     plt.tight_layout()
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     plt.savefig(out_path)
#     plt.close(fig)


# # =========================
# # Depth Anything V2 (relative) inference
# # =========================
# @torch.inference_mode()
# def infer_da2_relative_batch(
#     batch_u8: List[np.ndarray],
#     processor,
#     model,
#     device: torch.device,
#     autocast_dtype: torch.dtype,
#     orig_sizes: List[Tuple[int, int]],
# ) -> List[np.ndarray]:
#     """
#     Args:
#         batch_u8: list of (H,W,3) uint8 arrays
#         orig_sizes: list of (H,W) tuples, per image
#     Returns:
#         List of (H,W) float32 arrays in [0,1], 0=near,1=far
#     """
#     inputs = processor(images=batch_u8, return_tensors="pt")
#     pixel_values = inputs["pixel_values"].to(memory_format=torch.channels_last)
#     if device.type == "cuda":
#         pixel_values = pixel_values.pin_memory().to(device, non_blocking=True)
#     else:
#         pixel_values = pixel_values.to(device)

#     ctx = (
#         torch.autocast(device_type=device.type, dtype=autocast_dtype)
#         if device.type == "cuda"
#         else torch.cuda.amp.autocast(enabled=False)
#     )
#     with ctx:
#         out = model(pixel_values=pixel_values)
#         d = out.predicted_depth.to(torch.float32)  # [B,h,w]

#     # Resize individually to each original size; then robust normalize and invert
#     B = d.shape[0]
#     results: List[np.ndarray] = []
#     for i in range(B):
#         H, W = orig_sizes[i]
#         di = d[i:i+1, None, ...]  # [1,1,h,w]
#         di = F.interpolate(di, size=(H, W), mode="bicubic", align_corners=False)[0, 0]

#         flat = di.flatten()
#         q01 = torch.quantile(flat, 0.01)
#         q99 = torch.quantile(flat, 0.99)
#         q01 = torch.clamp(q01, min=1e-6)
#         q99 = torch.maximum(q99, q01 + 1e-6)
#         rel = torch.clamp((di - q01) / (q99 - q01), 0.0, 1.0)
#         rel = 1.0 - rel  # 0=near, 1=far

#         results.append(rel.detach().cpu().numpy().astype(np.float32))

#     return results


# # =========================
# # Main processing
# # =========================
# def main():
#     parser = argparse.ArgumentParser(
#         description="Generate RGB-D or depth-only files for cue-conflict with DA2"
#     )
#     parser.add_argument(
#         "--in-root",
#         type=Path,
#         required=True,
#         help="Input dataset root (e.g., datasets/cue-conflict)",
#     )
#     parser.add_argument(
#         "--out-root",
#         type=Path,
#         required=True,
#         help="Output root (e.g., datasets/cue-conflict-depth_anything_v2)",
#     )
#     parser.add_argument(
#         "--format",
#         type=str,
#         default="npy",
#         choices=["npy", "npz", "tiff", "png4"],
#         help="Output file format",
#     )
#     parser.add_argument(
#         "--batch-size",
#         type=int,
#         default=96,
#         help="Batch size for inference",
#     )
#     parser.add_argument(
#         "--model",
#         type=str,
#         default="depth-anything/Depth-Anything-V2-Large-hf",
#         help="HF model id",
#     )
#     parser.add_argument(
#         "--use-bf16",
#         action="store_true",
#         help="Use bfloat16 autocast when available",
#     )
#     parser.add_argument(
#         "--save-sanity",
#         type=int,
#         default=10,
#         help="Save N sanity images per (sub)folder; 0 to disable",
#     )
#     parser.add_argument(
#         "--overwrite",
#         action="store_true",
#         help="Overwrite existing outputs",
#     )
#     parser.add_argument(
#         "--sanity-dir",
#         type=Path,
#         default=Path("results/rgbd_cue-conflict_sanity"), # --sanity-dir .results/rgbd_cue-conflict_sanity
#         help="Directory to save sanity images",
#     )
#     parser.add_argument(
#         "--depth-only",
#         action="store_true",
#         help="If set, save only depth (no RGB).",
#     )

#     args = parser.parse_args()

#     in_root: Path = args.in_root
#     out_root: Path = args.out_root
#     fmt: str = args.format.lower()
#     BATCH_SIZE: int = args.batch_size
#     MODEL_ID: str = args.model
#     DEPTH_ONLY: bool = args.depth_only

#     assert in_root.exists(), f"Input root not found: {in_root}"
#     ensure_dir(out_root)

#     # Device & CUDA settings
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
#     torch.backends.cudnn.benchmark = True
#     torch.backends.cuda.matmul.allow_tf32 = True
#     torch.backends.cudnn.allow_tf32 = True
#     torch.set_float32_matmul_precision("high")

#     autocast_dtype = (
#         torch.bfloat16
#         if (args.use_bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported())
#         else (torch.float16 if device.type == "cuda" else torch.float32)
#     )

#     # Load model
#     print(f"Loading Depth Anything V2: {MODEL_ID}")
#     processor = AutoImageProcessor.from_pretrained(MODEL_ID, use_fast=True)
#     model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID).to(device)
#     model.eval().to(memory_format=torch.channels_last)
#     try:
#         model = torch.compile(model, mode="max-autotune")
#         print("torch.compile enabled.")
#     except Exception as e:
#         print(f"torch.compile not used: {e}")

#     # Enumerate input images
#     all_imgs = list_images(in_root)
#     if not all_imgs:
#         print("No images found under", in_root)
#         return

#     print(f"Found {len(all_imgs)} images.")

#     # Group images by their immediate subdirectory to mirror sanity saves
#     # and show progress per class folder.
#     # e.g., datasets/cue-conflict/airplane/*.png -> class_dir = airplane
#     rel_to_class = {}
#     for p in all_imgs:
#         rel = p.relative_to(in_root)
#         class_dir = rel.parts[0] if len(rel.parts) > 1 else ""
#         rel_to_class.setdefault(class_dir, []).append(p)

#     for class_dir, paths in rel_to_class.items():
#         print(f"\n=== Processing folder: {class_dir or '(root)'} | {len(paths)} images ===")
#         # Where to save sanity images
#         sanity_count = 0
#         num_to_save = int(args.save_sanity)
        
#         sanity_dir = args.sanity_dir / (class_dir if class_dir else "root")
#         os.makedirs(sanity_dir, exist_ok=True)
#         if num_to_save > 0:
#             ensure_dir(sanity_dir)

#         # Process in batches
#         for i in tqdm(range(0, len(paths), BATCH_SIZE), desc=class_dir or "root"):
#             batch_paths = paths[i : i + BATCH_SIZE]
#             batch_imgs = [load_rgb_u8(p) for p in batch_paths]
#             orig_sizes = [(im.shape[0], im.shape[1]) for im in batch_imgs]

#             # Predict depths (relative 0..1, 0=near)
#             d_rel_list = infer_da2_relative_batch(
#                 batch_imgs, processor, model, device, autocast_dtype, orig_sizes
#             )

#             # Save outputs mirroring structure and basenames
#             for img_path, rgb_u8, d_rel in zip(batch_paths, batch_imgs, d_rel_list):
#                 rel = img_path.relative_to(in_root)
#                 out_dir = out_root / rel.parent
#                 ensure_dir(out_dir)
#                 out_base = out_dir / rel.stem  # keep basename

#                 # Skip if exists
#                 if not args.overwrite:
#                     if fmt == "npy" and (out_base.with_suffix(".npy")).exists():
#                         continue
#                     if fmt == "npz" and (out_base.with_suffix(".npz")).exists():
#                         continue
#                     if fmt == "tiff" and (out_base.with_suffix(".tiff")).exists():
#                         continue
#                     if fmt == "png4" and (out_base.with_suffix(".png")).exists():
#                         continue

#                 if DEPTH_ONLY:
#                     save_depth_only(d_rel, out_base, fmt)
#                 else:
#                     save_rgbd(rgb_u8, d_rel, out_base, fmt)

#                 # Save all sanity images per class folder
#                 rgb01 = rgb_u8.astype(np.float32) / 255.0
#                 save_sanity(rgb01, d_rel, sanity_dir / f"{out_base.name}.png")
#                 save_sanity(rgb01, d_rel, sanity_dir / f"{out_base.name}.pdf",dpi=300)
#                 sanity_count += 1

#             # Housekeeping
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()
#             if (i // BATCH_SIZE) % 8 == 0:
#                 gc.collect()

#     print("\nAll done. Output root:", out_root)


# if __name__ == "__main__":
#     main()