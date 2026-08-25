#!/usr/bin/env python3
"""
repr_shape_bias.py
==================

Representational shape-texture bias via *triple correlation*.

For every cue-conflict image ``{shape_cls}{i}-{tex_cls}{j}.png`` we:

    1. Extract a feature vector from a target layer of the model for
          (a) the cue-conflict image,
          (b) the original *shape* image ``original_shape_images/{shape_cls}/{shape_cls}{i}.png``,
          (c) the pre-processed *texture* image ``preprocessed_textures_224/{tex_cls}/{tex_cls}{j}.png``.
    2. Compute cosine similarities  ``s_shape = cos(a, b)``  and
       ``s_tex   = cos(a, c)``.
    3. Aggregate across samples in one of two modes:
         - ``"winner-take-all"`` (default): +1 each time ``s_shape > s_tex``,
           yielding a fraction in ``[0, 1]``.
         - ``"ratio"``: keep ``s_shape / (s_shape + s_tex)``, the per-sample
           soft shape bias; the reported score is its mean across samples.

This file is standalone and only depends on torch / PIL / numpy; it does
not touch the behavioral shape-bias pipeline.
"""

from __future__ import annotations

import logging
import os
import re
from collections import defaultdict
from os.path import join as pjoin
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.utils.helpers import (
    apply_gaussian_blur,
    apply_scale_free_contrast,
    convert_rgb_to_dkl,
    normalize_images_channels,
)

logger = logging.getLogger(__name__)

# Filename regex for cue-conflict images: "<shape_cls><i>-<tex_cls><j>.<ext>"
_CC_RE = re.compile(
    r"^([a-zA-Z]+)(\d+)-([a-zA-Z]+)(\d+)\.(png|jpg|jpeg|bmp|tif|tiff)$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------


def _default_repr_dirs(cue_conflict_paths: Sequence[str]) -> Tuple[str, str]:
    """Guess the reference directories from a cue-conflict image path.

    Assumes the user's layout mirrors the screenshot::

        <parent>/cue-conflict/<cls>/<cls>{i}-<cls>{j}.png
        <parent>/original_shape_images/<cls>/<cls>{i}.png
        <parent>/preprocessed_textures_224/<cls>/<cls>{j}.png
    """
    if not cue_conflict_paths:
        raise ValueError(
            "cue_conflict_paths is empty; cannot infer reference dirs."
        )

    p = Path(cue_conflict_paths[0]).resolve()
    cc_root: Optional[Path] = None
    for anc in [p] + list(p.parents):
        if anc.name == "cue-conflict":
            cc_root = anc
            break
    if cc_root is None:
        raise FileNotFoundError(
            "Could not locate a 'cue-conflict' component in the sample path: %s"
            % p
        )

    parent = cc_root.parent
    return str(parent / "original_shape_images"), str(
        parent / "preprocessed_textures_224"
    )


def _parse_cue_conflict_name(
    filename: str,
) -> Optional[Tuple[str, str, str, str]]:
    """Return ``(shape_cls, shape_id, tex_cls, tex_id)`` from a cue-conflict name.

    Returns ``None`` if the filename does not match the expected pattern
    (e.g. filler images where shape == texture).
    """
    m = _CC_RE.match(os.path.basename(filename))
    if m is None:
        return None
    shape_cls, shape_id, tex_cls, tex_id, _ = m.groups()
    return shape_cls, shape_id, tex_cls, tex_id


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def _guess_head_module(nn_model: torch.nn.Module) -> Optional[torch.nn.Module]:
    """Heuristically locate the classifier head to hook features just before it."""
    for name in ("fc", "classifier", "head", "heads"):
        if hasattr(nn_model, name):
            mod = getattr(nn_model, name)
            if isinstance(mod, torch.nn.Module):
                return mod
    return None


class _FeatureHook:
    """Context manager that records the *input* to the classifier head."""

    def __init__(self, head_module: torch.nn.Module):
        self._head = head_module
        self._captured: List[torch.Tensor] = []
        self._handle = None

    def __enter__(self):
        def _hook(_mod, inputs, _output):
            x = inputs[0].detach()
            if x.dim() > 2:  # e.g. (B, C, H, W) -> (B, C)
                x = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
            self._captured.append(x.cpu())

        self._handle = self._head.register_forward_hook(_hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._handle is not None:
            self._handle.remove()

    def consume(self) -> torch.Tensor:
        out = (
            torch.cat(self._captured, dim=0)
            if self._captured
            else torch.empty(0)
        )
        self._captured.clear()
        return out


def _build_transform(image_size: int, norm_type: str):
    """Minimal transform pipeline that matches the evaluator's preprocessing."""
    import torchvision.transforms as T

    steps = [
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor(),  # -> [0, 1]
    ]
    if norm_type == "mean-std":
        steps.append(
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
    # For "0-1" we leave ToTensor()'s output alone; _normalize_images in
    # simple_eval.py rescales per-batch, but a per-image [0,1] range is a
    # very close proxy for single-image feature extraction and avoids
    # coupling this module to simple_eval internals.
    return T.Compose(steps)


def _extract_features(
    nn_model: torch.nn.Module,
    image_paths: Sequence[str],
    image_size: int = 224,
    batch_size: int = 64,
    norm_type: str = "0-1",
    device: Optional[torch.device] = None,
    desc: str = "features",
    baseline: bool = None,
    stream: int = None,
    dvd_strategy: str = None,
    age_m: int = None,
    stream_concat: bool = None,
    lum: bool = None,
    rg: bool = None,
    by: bool = None,
    lum_blur: bool = None,
    lum_cs: bool = None,
    rg_blur: bool = None,
    rg_cs: bool = None,
    by_blur: bool = None,
    by_cs: bool = None,
    combined_color: bool = None,
) -> Dict[str, np.ndarray]:
    """Return ``{path: feature_vector}`` for every path in ``image_paths``.

    Uses a forward hook on the classifier head so it works for all
    standard torchvision backbones (resnet/vgg/alexnet/vit/...).
    If no head is found, falls back to the model's direct output.

    For two_stream_dvd models use the the last layer before concatination.
    """
    if len(image_paths) == 0:
        return {}
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform = _build_transform(image_size, norm_type)

    # Load images lazily in order; keep a list of (path, tensor).
    ordered = list(image_paths)
    nn_model.eval()

    if not stream:
        head = _guess_head_module(nn_model)
    else:
        head = None

    feats: Dict[str, np.ndarray] = {}

    def _run_batch(paths_chunk: List[str]):
        tensors = []
        for p in paths_chunk:
            with Image.open(p) as im:
                im = im.convert("RGB")
                tensors.append(transform(im))
        batch = torch.stack(tensors, dim=0).to(device)
        if head is not None:
            with _FeatureHook(head) as h, torch.no_grad():
                _ = nn_model(batch)
            arr = h.consume().numpy()
        else:
            with torch.no_grad():
                if not stream:
                    arr = nn_model(batch).detach().cpu().numpy()
                else:
                    if dvd_strategy in ["dkl", "dkl_dvd"]:
                        # checkt the scale of the images and shape in batch
                        batch = convert_rgb_to_dkl(batch, device)

                        batch = normalize_images_channels(batch, norm_type)

                        # magno/lum/dorsal
                        x_1 = batch[:, 0, :, :]
                        # parvo/rg/ventral
                        x_2 = batch[:, 1, :, :]
                        # konio/by
                        x_3 = batch[:, 2, :, :]

                        # unsqueeze
                        x_1 = x_1.unsqueeze(1)
                        x_2 = x_2.unsqueeze(1)
                        x_3 = x_3.unsqueeze(1)

                        if lum:

                            if lum_blur:
                                x_1 = apply_gaussian_blur(x_1)

                            if lum_cs:
                                x_1 = apply_scale_free_contrast(
                                    x_1, age_m, 224
                                )

                        if rg:

                            if rg_blur:
                                x_2 = apply_gaussian_blur(x_2)

                            if rg_cs:
                                x_2 = apply_scale_free_contrast(
                                    x_2, age_m, 224
                                )

                        if by:

                            if by_blur:
                                x_3 = apply_gaussian_blur(x_3)

                            if by_cs:
                                x_3 = apply_scale_free_contrast(
                                    x_3, age_m, 224
                                )

                        if stream == 1:

                            if baseline:
                                if lum:
                                    input = x_1
                                elif rg and not by:
                                    input = x_2
                                elif by and not rg:
                                    input = x_3
                                elif combined_color and by and rg:
                                    input = batch[:, 1:3, :, :]
                                    input[:, 0, :, :] = x_2.squeeze(1)
                                    input[:, 1, :, :] = x_3.squeeze(1)
                                
                                output = nn_model.stream1_block1(input)

                            else:
                                if lum and rg and not by:
                                    s1_input = x_1
                                    s2_input = x_2
                                elif lum and by and not rg:
                                    s1_input = x_1
                                    s2_input = x_3
                                elif rg and by and not lum:
                                    s1_input = x_2
                                    s2_input = x_3
                                elif combined_color and lum and rg and by:
                                    s1_input = x_1
                                    s2_input = batch[:, 1:3, :, :]
                                    s2_input[:, 0, :, :] = x_2.squeeze(1)
                                    s2_input[:, 1, :, :] = x_3.squeeze(1)

                                output = nn_model.stream1_block1(s1_input)

                                if stream_concat:
                                    output_s2 = nn_model.stream2_block1(s2_input)
                                    output = (3 * output + output_s2) / 4

                                    output = nn_model.stream1_block2(output)

                            arr = (
                                torch.nn.functional.adaptive_avg_pool2d(
                                    output, 1
                                )
                                .flatten(1)
                                .detach()
                                .cpu()
                                .numpy()
                            )

                        elif stream == 2:

                            if lum and rg and not by:
                                s1_input = x_1
                                s2_input = x_2
                            elif lum and by and not rg:
                                s1_input = x_1
                                s2_input = x_3
                            elif rg and by and not lum:
                                s1_input = x_2
                                s2_input = x_3
                            elif combined_color and lum and rg and by:
                                s1_input = x_1
                                s2_input = batch[:, 1:3, :, :]
                                s2_input[:, 0, :, :] = x_2.squeeze(1)
                                s2_input[:, 1, :, :] = x_3.squeeze(1)

                            output = nn_model.stream2_block1(s2_input)
                            if stream_concat:
                                output_s1 = nn_model.stream1_block1(s1_input)
                                output = (3 * output + output_s1) / 4

                                output = nn_model.stream2_block2(output)

                            arr = (
                                torch.nn.functional.adaptive_avg_pool2d(
                                    output, 1
                                )
                                .flatten(1)
                                .detach()
                                .cpu()
                                .numpy()
                            )
                    else:

                        if nn_model.shared_block is not None:
                            batch = nn_model.shared_block(batch)

                        if stream == 1:
                            batch_s1 = nn_model.stream1_block1(batch)

                            if stream_concat:
                                batch_s2 = nn_model.stream2_block1(batch)
                                batch_s1 = (3 + batch_s1 + batch_s2)

                                batch_s1 = nn_model.stream1_block2(batch_s1)

                            arr = (
                                torch.nn.functional.adaptive_avg_pool2d(
                                    batch_s1, 1
                                )
                                .flatten(1)
                                .detach()
                                .cpu()
                                .numpy()
                            )

                        if stream == 2:
                            batch_s2 = nn_model.stream2_block1(batch)

                            if stream_concat:
                                batch_s1 = nn_model.stream1_block1(batch)
                                batch_s2 = (3 + batch_s2 + batch_s1)

                                batch_s2 = nn_model.stream1_block2(batch_s2)

                            arr = (
                                torch.nn.functional.adaptive_avg_pool2d(
                                    batch_s2, 1
                                )
                                .flatten(1)
                                .detach()
                                .cpu()
                                .numpy()
                            )

        for p, row in zip(paths_chunk, arr):
            feats[p] = row

    chunk: List[str] = []
    for p in tqdm(ordered, desc=f"[repr-bias:{desc}]"):
        chunk.append(p)
        if len(chunk) == batch_size:
            _run_batch(chunk)
            chunk = []
    if chunk:
        _run_batch(chunk)
    return feats


# ---------------------------------------------------------------------------
# Triple correlation
# ---------------------------------------------------------------------------


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / (na * nb))


def compute_repr_shape_bias(
    model,
    cue_conflict_paths: Sequence[str],
    original_shape_dir: Optional[str] = None,
    texture_dir: Optional[str] = None,
    mode: str = "winner-take-all",
    image_size: int = 224,
    batch_size: int = 64,
    norm_type: str = "0-1",
    baseline: bool = None,
    stream: int = None,
    dvd_strategy: str = None,
    age_m: int = None,
    stream_concat: bool = None,
    lum: bool = None,
    rg: bool = None,
    by: bool = None,
    lum_blur: bool = None,
    lum_cs: bool = None,
    rg_blur: bool = None,
    rg_cs: bool = None,
    by_blur: bool = None,
    by_cs: bool = None,
    combined_color=None,
) -> Dict:
    """Compute representational shape-texture bias via triple correlation.

    Parameters
    ----------
    model
        Either a ``PytorchModel`` wrapper (in which case ``model.model`` is
        used) or a raw ``torch.nn.Module``.
    cue_conflict_paths
        Full paths to the cue-conflict images produced by the dataset loader.
        Images whose filename does not match the
        ``{shape_cls}{i}-{tex_cls}{j}`` pattern are silently skipped (these
        are filler images where shape == texture).
    original_shape_dir, texture_dir
        Folders containing ``<class>/<class><i>.png`` images.  If ``None``
        they are auto-discovered as siblings of the ``cue-conflict`` folder.
    mode
        ``"winner-take-all"`` (default): each sample contributes 1 iff
            ``sim_shape > sim_texture``, so the score is the *fraction of
            shape-favouring samples*.
        ``"ratio"``:  each sample contributes ``sim_shape / (sim_shape +
            sim_texture)``; the reported score is the mean of that quantity.
    image_size, batch_size, norm_type
        Preprocessing / batching knobs for feature extraction.
    stream
        Either stream 1 or 2 for the two_stream_dvd models.

    Returns
    -------
    dict
        ::

            {
                "mode":                  mode,
                "mean_repr_shape_bias":  float,
                "median_repr_shape_bias":float,
                "n_samples_used":        int,
                "n_samples_skipped":     int,
                "per_category":          {cls: score, ...},
                "per_sample":            {path: (sim_shape, sim_tex, score)},
            }
    """
    if mode not in ("winner-take-all", "ratio"):
        raise ValueError(
            f"repr mode must be 'winner-take-all' or 'ratio'; got {mode!r}"
        )

    # Unwrap PytorchModel.
    nn_model = getattr(model, "model", model)

    if original_shape_dir is None or texture_dir is None:
        guess_shape, guess_tex = _default_repr_dirs(cue_conflict_paths)
        original_shape_dir = original_shape_dir or guess_shape
        texture_dir = texture_dir or guess_tex
    if not os.path.isdir(original_shape_dir):
        raise FileNotFoundError(
            f"original_shape_dir not found: {original_shape_dir}. "
            "Pass it explicitly via `repr_shape_dir=...`."
        )
    if not os.path.isdir(texture_dir):
        raise FileNotFoundError(
            f"texture_dir not found: {texture_dir}. "
            "Pass it explicitly via `repr_texture_dir=...`."
        )

    # Build the list of triples (cc_path, shape_ref_path, tex_ref_path).
    triples: List[Tuple[str, str, str, str, str]] = (
        []
    )  # cc, shape, tex, shape_cls, tex_cls
    skipped = 0
    missing = 0
    for cc_path in cue_conflict_paths:
        parsed = _parse_cue_conflict_name(cc_path)
        if parsed is None:
            skipped += 1
            continue
        shape_cls, shape_id, tex_cls, tex_id = parsed
        # Try common extensions if .png isn't present.
        shape_ref = _resolve_ref(original_shape_dir, shape_cls, shape_id)
        tex_ref = _resolve_ref(texture_dir, tex_cls, tex_id)
        if shape_ref is None or tex_ref is None:
            missing += 1
            continue
        triples.append((cc_path, shape_ref, tex_ref, shape_cls, tex_cls))

    if not triples:
        raise RuntimeError(
            "Found no valid (cue-conflict, shape, texture) triples. "
            "Check that original_shape_dir / texture_dir contain the "
            "matching reference images."
        )

    # Collect the unique paths to extract features only once.
    unique_paths: List[str] = sorted({p for t in triples for p in t[:3]})
    feats = _extract_features(
        nn_model,
        unique_paths,
        image_size=image_size,
        batch_size=batch_size,
        norm_type=norm_type,
        desc=f"{len(unique_paths)} imgs",
        baseline=baseline,
        stream=stream,
        dvd_strategy=dvd_strategy,
        age_m=age_m,
        stream_concat=stream_concat,
        lum=lum,
        rg=rg,
        by=by,
        lum_blur=lum_blur,
        lum_cs=lum_cs,
        rg_blur=rg_blur,
        rg_cs=rg_cs,
        by_blur=by_blur,
        by_cs=by_cs,
        combined_color=combined_color,
    )

    # Accumulate metrics.
    per_sample: Dict[str, Tuple[float, float, float]] = {}
    per_cat_scores: Dict[str, List[float]] = defaultdict(list)
    scores: List[float] = []

    for cc, shape_ref, tex_ref, shape_cls, _tex_cls in triples:
        s_shape = _cosine(feats[cc], feats[shape_ref])
        s_tex = _cosine(feats[cc], feats[tex_ref])
        if mode == "winner-take-all":
            score = 1.0 if s_shape > s_tex else 0.0
        else:  # "ratio"
            denom = abs(s_shape) + abs(s_tex) + 1e-12
            score = s_shape / denom
        per_sample[cc] = (s_shape, s_tex, score)
        per_cat_scores[shape_cls].append(score)
        scores.append(score)

    scores_arr = np.asarray(scores, dtype=np.float64)
    per_category = {k: float(np.mean(v)) for k, v in per_cat_scores.items()}

    return {
        "mode": mode,
        "mean_repr_shape_bias": float(scores_arr.mean()),
        "median_repr_shape_bias": float(np.median(scores_arr)),
        "n_samples_used": int(len(scores)),
        "n_samples_skipped": int(skipped + missing),
        "per_category": per_category,
        "per_sample": per_sample,
    }


def _resolve_ref(root: str, cls: str, idx: str) -> Optional[str]:
    """Find ``<root>/<cls>/<cls><idx>.<ext>`` for a few common extensions."""
    base = pjoin(root, cls, f"{cls}{idx}")
    for ext in (
        "png",
        "jpg",
        "jpeg",
        "bmp",
        "tif",
        "tiff",
        "PNG",
        "JPG",
        "JPEG",
    ):
        cand = f"{base}.{ext}"
        if os.path.isfile(cand):
            return cand
    return None
