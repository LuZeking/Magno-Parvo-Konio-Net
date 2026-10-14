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

# Which DKL opponent axis stream 2 receives. The dataset stores all three
# channels (lum, rg, by) -- see rgb_to_dkl3 -- so the koniocellular S-(L+M)
# input needs no new data conversion, only a different slice.
CHROMA_INDEX = {"rg": 1, "by": 2}
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


def rgb_to_dkl3(
    rgb_uint8: np.ndarray,
    ch_min: np.ndarray,
    ch_max: np.ndarray,
    image_size: int = IMAGE_SIZE,
) -> torch.Tensor:
    """RGB uint8 (N, H, W, 3) -> the full 3-channel DKL image (N, 3, S, S).

    This is the tensor the training dataset actually stores: lum, rg, by,
    each normalized to [0, 1] with the training-set min/max and quantized to
    uint8. The stream front ends below slice it; the ``rgb_tsn`` condition
    hands all three channels to both streams unsliced.

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

    return torch.from_numpy(dkl01).permute(0, 3, 1, 2)  # (N, 3, S, S)


def rgb_to_stream_inputs(
    rgb_uint8: np.ndarray,
    ch_min: np.ndarray,
    ch_max: np.ndarray,
    image_size: int = IMAGE_SIZE,
    chroma_channel: str = "rg",
) -> tuple[torch.Tensor, torch.Tensor]:
    """RGB uint8 (N, H, W, 3) -> (x_lum, x_chroma), each (N, 1, S, S) in [0, 1].

    ``chroma_channel`` selects the opponent axis for stream 2: "rg" (L-M,
    parvocellular, the trained default) or "by" (S-(L+M), koniocellular).
    Stream 1 is the luminance channel either way, so switching this changes
    exactly one thing about the model.
    """
    if chroma_channel not in CHROMA_INDEX:
        raise ValueError(
            f"chroma_channel={chroma_channel!r}; expected one of "
            f"{sorted(CHROMA_INDEX)}"
        )
    x = rgb_to_dkl3(rgb_uint8, ch_min, ch_max, image_size)
    j = CHROMA_INDEX[chroma_channel]
    return x[:, 0:1], x[:, j:j + 1]


def match_chroma_contrast(
    x_chroma: torch.Tensor, x_rg: torch.Tensor
) -> torch.Tensor:
    """Rescale ``x_chroma`` about its per-image mean to the per-image RMS
    contrast of ``x_rg``, image by image.

    WHY THIS EXISTS. Every DKL channel is normalised to [0, 1] with its OWN
    global min/max (see ``rgb_to_dkl3``), and those ranges differ a lot --
    rg spans 1.048 of cone contrast, by spans 8.189. The normalisation does
    not equalise what the network sees, it just re-centres it, so on NSD
    stimuli the by channel arrives with **2.05x** the per-image RMS contrast
    of rg (0.0941 vs 0.0460) and 3.5x the absolute power above 1/8 cyc/px.
    A konio stream that explains MORE cortical variance than the parvo
    stream would therefore be uninterpretable: the chromatic axis and the
    input contrast would be confounded, and the confound points the way that
    flatters the new arm.

    Matching is a pure contraction here (rg has the lower contrast), so no
    value leaves [0, 1] and no clipping is applied. It happens AFTER the
    uint8 quantisation inside ``rgb_to_dkl3``, deliberately: the quantisation
    is part of what training saw and is not re-applied, so the matched
    channel is a rescaled version of the exact tensor the unmatched arm uses.

    Both channels come from the same image, so the target is defined per
    image rather than as a dataset-wide constant -- a global factor would
    leave the per-image contrast ratio varying with image content.
    """
    if x_chroma.shape != x_rg.shape:
        raise ValueError(f"{x_chroma.shape} != {x_rg.shape}")
    n = x_chroma.shape[0]
    mu = x_chroma.reshape(n, -1).mean(dim=1).reshape(n, 1, 1, 1)
    sd_c = x_chroma.reshape(n, -1).std(dim=1).reshape(n, 1, 1, 1)
    sd_t = x_rg.reshape(n, -1).std(dim=1).reshape(n, 1, 1, 1)
    # A flat image has sd 0 and no contrast to match; leave it alone rather
    # than dividing by zero and producing NaNs that would poison the PCA.
    scale = torch.where(sd_c > 1e-8, sd_t / sd_c.clamp_min(1e-8),
                        torch.ones_like(sd_c))
    return mu + (x_chroma - mu) * scale


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


def apply_dvd_diet(x: torch.Tensor, age_months: float) -> torch.Tensor:
    """DVD acuity blur, then the same scale-free CS cut, both at ``age_months``.

    The progressive magno diet of the Stage-2e TSN arms: unlike the thesis
    diet (fixed ``BLUR_SIGMA`` 4, about a 3-month-old's DVD acuity, plus CS at
    48 months) both stages take one age. Blur sigma is DVD's acuity at that
    age scaled to the image width (0.50 px at 48 months for 256 px). The CS
    stage is identical to ``apply_scale_free_contrast``.
    """
    from dvd_scale_free.dvd_scale_free.development import (  # noqa: WPS433
        DVDConfig,
        DVDTransformer,
    )

    dvdt = DVDTransformer(
        DVDConfig(
            apply_blur=1,
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
    transform_target: str = "magno",
    parvo_highsf: bool = False,
    rgb_mode: bool = False,
    chroma_channel: str = "rg",
    chroma_rescale: str = "none",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Full front end: RGB uint8 -> (magno input, parvo input).

    Defaults correspond to the DKL B&CS family (blur + contrast on lum,
    rg untouched). Set both flags False for DKL Base.

    ``transform_target="parvo"`` is the channel-swap control: the blur and
    contrast-sensitivity cut are applied to the rg channel instead, which
    dissociates "which transformation" from "which chromatic channel" as
    the cause of any stream difference.

    ``rgb_mode=True`` is the ``rgb_tsn`` architecture-only control: BOTH
    streams receive the identical, untransformed 3-channel image, so the two
    streams differ by nothing except initialization. Note the name is a
    misnomer inherited from the training config -- ``slurm/train_array.sh``
    feeds every condition the DKL h5, so those three channels are lum/rg/by,
    not R/G/B. What the condition really controls for is "two streams with no
    input asymmetry at all". Requires ``build_model(in_channels=(3, 3))``, and
    blur/CS are ignored because the trained condition applies neither.

    ``parvo_highsf=True`` is the biologically corrected parvo input: stream 2
    receives UNTRANSFORMED high-acuity luminance stacked with rg (2 channels),
    because real parvocellular cells carry both. It mirrors
    ``standalone_trainer.StreamInputBuilder`` exactly -- note that the
    luminance it stacks is the raw one, captured before any transform is
    written back, not the blurred copy magno receives. Requires
    ``features.build_model(in_channels=(1, 2))``.

    ``chroma_channel="by"`` is the KONIO arm: stream 2 receives the S-(L+M)
    axis instead of L-M. Nothing else changes -- same luminance stream, same
    transforms, same architecture, same seed -- so this is a single-factor
    swap of the chromatic axis, and it is the honest test of whether the null
    colour result in VO/hV4 is a failure of the colour prediction or a
    failure of L-M as the model of colour cortex (PAPER_PLAN B3 / §65).

    ``chroma_rescale="match-rg"`` additionally matches that channel's
    per-image RMS contrast to the rg channel's, which is required for the
    comparison to be about chromaticity at all -- see
    ``match_chroma_contrast`` for the measured 2.05x contrast gap it removes.

    Combining ``chroma_channel="by"`` with ``transform_target="parvo"`` puts
    the developmental transform on the by channel; combining it with
    ``parvo_highsf`` stacks raw luminance with by. Both are expressible but
    neither is part of the pre-registered design.
    """
    if transform_target not in ("magno", "parvo"):
        raise ValueError(f"transform_target={transform_target!r}")
    if chroma_channel not in CHROMA_INDEX:
        raise ValueError(f"chroma_channel={chroma_channel!r}")
    if chroma_rescale not in ("none", "match-rg"):
        raise ValueError(f"chroma_rescale={chroma_rescale!r}")
    if chroma_rescale == "match-rg" and chroma_channel == "rg":
        # Matching rg to rg is the identity. Refuse rather than accept a
        # flag that makes a run LOOK like the contrast-matched control while
        # being bit-identical to the unmatched one.
        raise ValueError(
            "chroma_rescale='match-rg' with chroma_channel='rg' is a no-op"
        )
    if rgb_mode:
        if parvo_highsf:
            raise ValueError("rgb_mode and parvo_highsf are exclusive")
        # rgb_mode hands BOTH streams all three channels, so a non-default
        # chroma_channel would be silently ignored. Refuse rather than
        # produce a run whose name promises a konio input it never received.
        if chroma_channel != "rg" or chroma_rescale != "none":
            raise ValueError(
                "rgb_mode gives both streams all three DKL channels, so "
                "chroma_channel/chroma_rescale would have no effect"
            )
        x = rgb_to_dkl3(rgb_uint8, ch_min, ch_max).to(device)
        return x, x
    # One DKL conversion, then slice. The conversion is the expensive step,
    # and the contrast-matching control needs the rg channel as a reference
    # even when stream 2 receives a different axis, so taking all three
    # channels once is both cheaper and the only way to define the target.
    x3 = rgb_to_dkl3(rgb_uint8, ch_min, ch_max)
    j = CHROMA_INDEX[chroma_channel]
    x_lum = x3[:, 0:1].to(device)
    x_rg = x3[:, j:j + 1].to(device)
    if chroma_rescale == "match-rg":
        x_rg = match_chroma_contrast(x_rg, x3[:, 1:2].to(device))
    # Captured before the transform is written back below: parvo_highsf feeds
    # stream 2 the UNTRANSFORMED luminance, so taking it afterwards would hand
    # parvo the blurred copy and silently make this the wrong condition.
    raw_lum, raw_rg = x_lum, x_rg

    target = x_lum if transform_target == "magno" else x_rg
    if lum_blur:
        target = apply_gaussian_blur(target)
    if lum_cs:
        target = apply_scale_free_contrast(target, age_months)
    if transform_target == "magno":
        x_lum = target
    else:
        x_rg = target
    if parvo_highsf:
        return x_lum, torch.cat([raw_lum, raw_rg], dim=1)
    return x_lum, x_rg
