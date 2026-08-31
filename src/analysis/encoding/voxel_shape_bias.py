"""Per-voxel shape-bias index from encoding-model predictions (PAPER_PLAN §7.3b).

The behavioural leg of this project measures shape bias at the *readout*
(`src/analysis/behavior/cue_conflict.py`, B_shape over the Geirhos
cue-conflict set). The cortical leg measures stream preference per voxel
(SPI). Nothing connects them. This module builds the connecting quantity:
push the same cue-conflict images through a *fitted* voxelwise encoding model
and ask, per voxel, whether the predicted responses group by the image's
shape cue or by its texture cue.

Each cue-conflict image carries two crossed 16-level labels, so for a voxel's
predicted responses p:

    eta2_shape   = SS_between(shape)   / SS_total
    eta2_texture = SS_between(texture) / SS_total
    VSB          = (eta2_shape - eta2_texture) / (eta2_shape + eta2_texture)

VSB is bounded [-1, +1] and is the voxel-level analogue of B_shape. It needs
no original (non-conflict) images, and being a ratio it does not reward
voxels the model simply predicts well -- but it *is* unstable where the
denominator is small, which is why the eta2 terms are stored alongside it
rather than only the ratio (§21.4).

Note the Geirhos set is close to but not exactly balanced (same-cue images
are excluded, so each shape appears with 15 of the 16 textures and the counts
per cell are not identical). The eta2 terms are therefore descriptive
proportions of variance, not the terms of an orthogonal ANOVA decomposition,
and they need not sum to <= 1. Both factors are treated identically, so the
contrast VSB is unaffected by the imbalance.
"""

from __future__ import annotations

import numpy as np


def eta2_by_factor(pred: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Per-voxel between-group variance ratio for one categorical factor.

    pred   : (n_images, n_voxels) predicted responses
    labels : (n_images,) categorical labels
    returns: (n_voxels,) SS_between / SS_total, NaN where SS_total == 0
    """
    pred = np.asarray(pred, dtype=np.float64)
    labels = np.asarray(labels)
    grand = pred.mean(axis=0, keepdims=True)
    ss_total = ((pred - grand) ** 2).sum(axis=0)

    ss_between = np.zeros(pred.shape[1], dtype=np.float64)
    for lev in np.unique(labels):
        sel = labels == lev
        n = int(sel.sum())
        if n == 0:
            continue
        ss_between += n * (pred[sel].mean(axis=0) - grand[0]) ** 2

    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(ss_total > 0, ss_between / ss_total, np.nan)


def voxel_shape_bias(
    pred: np.ndarray, shapes: np.ndarray, textures: np.ndarray
) -> dict[str, np.ndarray]:
    """eta2 per factor and the VSB contrast, per voxel."""
    e_s = eta2_by_factor(pred, shapes)
    e_t = eta2_by_factor(pred, textures)
    denom = e_s + e_t
    with np.errstate(invalid="ignore", divide="ignore"):
        vsb = np.where(denom > 0, (e_s - e_t) / denom, np.nan)
    return {"eta2_shape": e_s, "eta2_texture": e_t, "vsb": vsb}


def cue_conflict_design() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(rgb_uint8, shape_labels, texture_labels) for the Geirhos set.

    Reuses the behavioural module's listing so the encoding panel and the
    behavioural report are computed over the identical image set -- including
    its exclusion of same-cue images.
    """
    from src.analysis.behavior import cue_conflict

    items = cue_conflict.list_cue_conflict_images()
    if not items:
        raise RuntimeError(
            f"no cue-conflict images under {cue_conflict.CUE_CONFLICT_DIR}"
        )
    paths = [i[0] for i in items]
    shapes = np.asarray([i[1] for i in items])
    textures = np.asarray([i[2] for i in items])
    return cue_conflict.load_images(paths), shapes, textures
