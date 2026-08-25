"""Model front end for NSD stimuli, replicating training preprocessing exactly.

Training pipeline being replicated (see src/data/convert_to_dkl.py and
src/models/lightning_two_stream_model.py):

  1. RGB uint8 -> DKL cone-contrast space via
     ``src.utils.helpers.convert_rgb_to_dkl_numpy`` (sRGB->XYZ->LMS(HPE),
     cone contrast vs. mid-gray, opponent axes lum / rg / by).
  2. Per-channel normalization to [0, 1] with the TRAINING DATASET's global
     min/max (stored in the converted h5; recomputed here from the local
     source h5 and cached), then uint8 quantization — both reproduced so
     encoding features see the same numeric range as training.
  3. Stream inputs: x1 = lum channel, x2 = rg channel (each 1 x H x W).
  4. Magno-stream transforms (matching ``_apply_gaussian_blur`` /
     ``_apply_scale_free_contrast``): kornia gaussian_blur2d with sigma=4,
     kernel 33, reflect border; DVD scale-free contrast-sensitivity cut with
     cs_logspan_start=5e-3, image_size=256, decomposition_mode="keep",
     evaluated at ``age_months`` (thesis default: 48).
"""

from __future__ import annotations

import json
import os

import h5py
import numpy as np
import torch

from src.utils.helpers import convert_rgb_to_dkl_numpy

MINIECOSET_H5 = os.environ.get(
    "MINIECOSET_H5",
    "/scratch/byrong/VST_data/VST_datasets/texture2shape_miniecoset_256px.h5",
)
MINMAX_CACHE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "dkl_channel_minmax.json"
)
NSD_STIMULI_H5 = os.environ.get(
    "NSD_STIMULI_H5",
    "/scratch/byrong/encoding/datasets/nsd/stimuli/nsd_stimuli.hdf5",
)
IMAGE_SIZE = 256
BLUR_SIGMA = 4.0
CS_LOGSPAN_START = 5e-3
DEFAULT_AGE_MONTHS = 48


def compute_dkl_channel_minmax(
    h5_path: str = MINIECOSET_H5,
    splits=("train", "val", "test", "testplus"),
    batch_size: int = 200,
    cache_path: str = MINMAX_CACHE,
) -> tuple[np.ndarray, np.ndarray]:
    """Global per-channel DKL min/max of the training dataset (cached).

    Must match convert_to_dkl.py pass 1 so that NSD images are normalized
    identically to the training images.
    """
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            d = json.load(f)
        return np.asarray(d["min"]), np.asarray(d["max"])

    ch_min = np.full(3, np.inf)
    ch_max = np.full(3, -np.inf)
    with h5py.File(h5_path, "r") as f:
        for split in splits:
            if split not in f:
                continue
            data = f[split]["data"]
            for start in range(0, data.shape[0], batch_size):
                dkl = convert_rgb_to_dkl_numpy(
                    data[start : start + batch_size]
                )
                flat = dkl.reshape(-1, 3)
                ch_min = np.minimum(ch_min, flat.min(axis=0))
                ch_max = np.maximum(ch_max, flat.max(axis=0))
    with open(cache_path, "w") as f:
        json.dump({"min": ch_min.tolist(), "max": ch_max.tolist()}, f)
    return ch_min, ch_max


def load_nsd_images(image_ids: np.ndarray) -> np.ndarray:
    """Load NSD stimuli (uint8 RGB, 425x425) for 0-based image ids."""
    order = np.argsort(image_ids)
    with h5py.File(NSD_STIMULI_H5, "r") as f:
        imgs_sorted = f["imgBrick"][np.sort(image_ids)]
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return imgs_sorted[inverse]


def rgb_to_stream_inputs(
    rgb_uint8: np.ndarray,
    ch_min: np.ndarray,
    ch_max: np.ndarray,
    image_size: int = IMAGE_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RGB uint8 (N, H, W, 3) -> (x_lum, x_rg), each (N, 1, S, S) in [0, 1].

    Resizing to the training resolution happens BEFORE the DKL conversion
    (bilinear + antialias), then normalization with the training min/max and
    uint8 quantization mirror the stored training dataset.
    """
    import torch.nn.functional as F

    t = torch.from_numpy(np.ascontiguousarray(rgb_uint8)).permute(0, 3, 1, 2)
    t = t.to(torch.float32)
    if t.shape[-1] != image_size:
        t = F.interpolate(
            t,
            size=(image_size, image_size),
            mode="bilinear",
            antialias=True,
            align_corners=False,
        )
    rgb_resized = (
        t.clamp(0, 255).round().permute(0, 2, 3, 1).numpy().astype(np.uint8)
    )

    dkl = convert_rgb_to_dkl_numpy(rgb_resized)  # (N, S, S, 3) float64
    span = (ch_max - ch_min) + 1e-12
    dkl = (dkl - ch_min[None, None, None, :]) / span[None, None, None, :]
    np.clip(dkl, 0.0, 1.0, out=dkl)
    dkl_u8 = (dkl * 255.0).round().astype(np.uint8)  # training quantization
    dkl01 = dkl_u8.astype(np.float32) / 255.0

    x = torch.from_numpy(dkl01).permute(0, 3, 1, 2)  # (N, 3, S, S)
    x_lum = x[:, 0:1]
    x_rg = x[:, 1:2]
    return x_lum, x_rg


def apply_gaussian_blur(x: torch.Tensor, sigma: float = BLUR_SIGMA):
    """Kornia gaussian blur exactly as in the training module."""
    import kornia.filters as KF

    k = int(8 * sigma)
    kernel_size = k + (1 if k % 2 == 0 else 0)
    return KF.gaussian_blur2d(
        x, (kernel_size, kernel_size), (sigma, sigma), border_type="reflect"
    )


def apply_scale_free_contrast(
    x: torch.Tensor, age_months: int = DEFAULT_AGE_MONTHS
) -> torch.Tensor:
    """DVD scale-free contrast-sensitivity cut, as in training."""
    from dvd_scale_free.dvd_scale_free.development import (  # noqa: WPS433
        DVDConfig,
        DVDTransformer,
    )

    dvdt = DVDTransformer(
        DVDConfig(
            apply_blur=0,
            apply_color=0,
            apply_contrast=1,
            cs_logspan_start=CS_LOGSPAN_START,
            image_size=IMAGE_SIZE,
            decomposition_mode="keep",
        )
    )
    return dvdt(x, age_months)


def magno_parvo_inputs(
    rgb_uint8: np.ndarray,
    ch_min: np.ndarray,
    ch_max: np.ndarray,
    lum_blur: bool = True,
    lum_cs: bool = True,
    age_months: int = DEFAULT_AGE_MONTHS,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Full front end: RGB uint8 -> (magno input, parvo input).

    Defaults correspond to the DKL B&CS family (blur + contrast on lum,
    rg untouched). Set both flags False for DKL Base.
    """
    x_lum, x_rg = rgb_to_stream_inputs(rgb_uint8, ch_min, ch_max)
    x_lum = x_lum.to(device)
    x_rg = x_rg.to(device)
    if lum_blur:
        x_lum = apply_gaussian_blur(x_lum)
    if lum_cs:
        x_lum = apply_scale_free_contrast(x_lum, age_months)
    return x_lum, x_rg
