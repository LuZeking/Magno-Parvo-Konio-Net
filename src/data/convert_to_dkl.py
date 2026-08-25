"""Convert RGB image splits in an HDF5 file to DKL color space, saved as uint8.

Two-pass streaming converter. The source file (e.g. texture2shape_miniecoset)
has split groups (train / val / test / testplus), each with a uint8 RGB
`data` dataset of shape (N, 256, 256, 3). We:

  Pass 1 — read each batch from source, convert RGB→DKL on the fly, and
           accumulate per-channel global min/max. Discard each converted
           batch after updating the stats.

  Pass 2 — read each batch again, convert RGB→DKL, normalize each DKL
           channel to [0, 1] using the global min/max from pass 1, scale
           to [0, 255], cast to uint8, and write into the destination
           HDF5 group's `data` (replacing the original RGB).

Constant memory wrt dataset size (only one batch held at a time). Non-image
datasets in the source (categories, fasttext_embeddings, labels,
image_size_hwc, train_img_mean_channels, etc.) are preserved verbatim via
an initial file copy. The destination also stores `dkl_channel_min` and
`dkl_channel_max` as float64 arrays for downstream reverse-mapping.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from src.utils.helpers import convert_rgb_to_dkl_numpy

SRC_PATH = Path("/share/klab/datasets/texture2shape_miniecoset_256px.h5")
DST_PATH = Path(
    "/share/klab/datasets/texture2shape_miniecoset_256px_dkl_kapoor.h5"
)
SPLITS = ("train", "val", "test", "testplus")
BATCH_SIZE = 100
N_CHANNELS = 3  # DKL axes: lum, rg, by


def channel_minmax(
    rgb_dataset: h5py.Dataset,
    batch_size: int,
    desc: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Stream RGB → DKL and return per-channel (min, max) without materializing.

    Returns two float64 arrays of shape (3,) — one row per DKL channel.
    """
    ch_min = np.full(N_CHANNELS, np.inf, dtype=np.float64)
    ch_max = np.full(N_CHANNELS, -np.inf, dtype=np.float64)
    n = rgb_dataset.shape[0]
    for start in tqdm(range(0, n, batch_size), desc=desc):
        end = min(start + batch_size, n)
        dkl_batch = convert_rgb_to_dkl_numpy(rgb_dataset[start:end])
        # axis order is (N, H, W, C); reduce over the spatial+sample axes
        ch_min = np.minimum(
            ch_min, dkl_batch.reshape(-1, N_CHANNELS).min(axis=0)
        )
        ch_max = np.maximum(
            ch_max, dkl_batch.reshape(-1, N_CHANNELS).max(axis=0)
        )
    return ch_min, ch_max


def write_dkl_uint8(
    dst_group: h5py.Group,
    src_dataset: h5py.Dataset,
    ch_min: np.ndarray,
    ch_max: np.ndarray,
    batch_size: int,
    desc: str,
) -> None:
    """Stream RGB → DKL → uint8 from `src_dataset` into `dst_group['data']`."""
    shape = src_dataset.shape
    span = (ch_max - ch_min) + 1e-12  # avoid div-by-zero on a flat channel

    if "data" in dst_group:
        del dst_group["data"]

    out = dst_group.create_dataset(
        "data",
        shape=shape,
        dtype=np.uint8,
        chunks=(1, *shape[1:]),  # one image per chunk — cheap random reads
        compression="gzip",
        compression_opts=4,
    )

    for start in tqdm(range(0, shape[0], batch_size), desc=desc):
        end = min(start + batch_size, shape[0])
        dkl = convert_rgb_to_dkl_numpy(
            src_dataset[start:end]
        )  # (b, H, W, 3) float64
        dkl -= ch_min[None, None, None, :]
        dkl /= span[None, None, None, :]
        np.clip(dkl, 0.0, 1.0, out=dkl)
        out[start:end] = (dkl * 255.0).round().astype(np.uint8)


def main() -> None:
    if not SRC_PATH.exists():
        raise FileNotFoundError(f"Source not found: {SRC_PATH}")

    if not DST_PATH.exists():
        print(f"Copying {SRC_PATH} → {DST_PATH}")
        DST_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SRC_PATH, DST_PATH)
    else:
        print(f"Destination already exists, modifying in place: {DST_PATH}")

    # ── Pass 1: compute per-channel min/max across all splits ──
    print("\n[Pass 1] Computing DKL per-channel min/max…")
    ch_min = np.full(N_CHANNELS, np.inf, dtype=np.float64)
    ch_max = np.full(N_CHANNELS, -np.inf, dtype=np.float64)
    with h5py.File(SRC_PATH, "r") as f_src:
        for split in SPLITS:
            if split not in f_src:
                print(f"  skip {split!r}: not in source")
                continue
            split_min, split_max = channel_minmax(
                f_src[split]["data"], BATCH_SIZE, desc=f"  stats[{split}]"
            )
            ch_min = np.minimum(ch_min, split_min)
            ch_max = np.maximum(ch_max, split_max)
    print(f"  channel min (lum, rg, by): {ch_min}")
    print(f"  channel max (lum, rg, by): {ch_max}")

    # ── Pass 2: write normalized DKL uint8 to dst ──
    print("\n[Pass 2] Writing DKL uint8…")
    with h5py.File(SRC_PATH, "r") as f_src, h5py.File(DST_PATH, "r+") as f_dst:
        for split in SPLITS:
            if split not in f_src:
                continue
            write_dkl_uint8(
                dst_group=f_dst[split],
                src_dataset=f_src[split]["data"],
                ch_min=ch_min,
                ch_max=ch_max,
                batch_size=BATCH_SIZE,
                desc=f"  write[{split}]",
            )

        # Persist normalization stats so downstream code can invert if needed.
        for k, v in (("dkl_channel_min", ch_min), ("dkl_channel_max", ch_max)):
            if k in f_dst:
                del f_dst[k]
            f_dst.create_dataset(k, data=v)

    print(f"\n✓ DKL uint8 written to {DST_PATH}")


if __name__ == "__main__":
    main()
