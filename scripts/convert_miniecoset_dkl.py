"""Convert MiniEcoset RGB -> DKL uint8, single pass.

Equivalent to src/data/convert_to_dkl.py but (a) reuses the already-computed
global per-channel min/max cache instead of re-running pass 1, (b) writes a
fresh file containing only what training needs (no 80 GB copy), and (c) stores
uncompressed chunks for dataloader throughput — the thesis identified
compressed/float64 reads as the training bottleneck.

Output layout mirrors the source: <split>/{data,labels}, data as
(N, 256, 256, 3) uint8 holding the normalized DKL axes (lum, rg, by).

Run:  python scripts/convert_miniecoset_dkl.py --out <path.h5>
"""

from __future__ import annotations

import argparse
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis.encoding.dkl_frontend import (  # noqa: E402
    MINIECOSET_H5,
    compute_dkl_channel_minmax,
)
from src.utils.helpers import convert_rgb_to_dkl_numpy  # noqa: E402

SPLITS = ("train", "val", "test", "testplus")
PASSTHROUGH = (
    "categories",
    "dataset_description",
    "id_of_categories_in_ecoset",
    "source_categories",
    "source_class_ids",
    "train_img_mean_channels",
    "train_img_std_channels",
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=MINIECOSET_H5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=200)
    args = ap.parse_args()

    ch_min, ch_max = compute_dkl_channel_minmax()
    span = (ch_max - ch_min) + 1e-12
    print(f"using cached DKL min={ch_min} max={ch_max}", flush=True)

    with h5py.File(args.src, "r") as fs, h5py.File(args.out, "w") as fd:
        for key in PASSTHROUGH:
            if key in fs:
                fs.copy(key, fd)
        fd.create_dataset("dkl_channel_min", data=ch_min)
        fd.create_dataset("dkl_channel_max", data=ch_max)

        for split in SPLITS:
            if split not in fs:
                print(f"skip {split!r}: absent", flush=True)
                continue
            src_data = fs[split]["data"]
            n = src_data.shape[0]
            grp = fd.create_group(split)
            for key in fs[split]:
                if key != "data":
                    fs[split].copy(key, grp)  # labels etc. verbatim
            dst = grp.create_dataset(
                "data",
                shape=src_data.shape,
                dtype=np.uint8,
                chunks=(1, *src_data.shape[1:]),
            )
            for start in range(0, n, args.batch_size):
                end = min(start + args.batch_size, n)
                dkl = convert_rgb_to_dkl_numpy(src_data[start:end])
                dkl -= ch_min[None, None, None, :]
                dkl /= span[None, None, None, :]
                np.clip(dkl, 0.0, 1.0, out=dkl)
                dst[start:end] = (dkl * 255.0).round().astype(np.uint8)
                if start % (args.batch_size * 50) == 0:
                    print(f"  {split}: {end}/{n}", flush=True)
            print(f"done {split}: {n} images", flush=True)

    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
