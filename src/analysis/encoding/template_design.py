"""Extra stimulus designs pushed through a fitted encoding model.

docs/PLAN_CORTEX_DECISION_2026-10-26.md §1. ``voxel_shape_bias.cue_conflict_design``
supplies the 1,200 cue-conflict images (the test set). Two further sets ride
along in the same extra design of ``banded_ridge.fit_banded`` -- same front end,
feature stage, PCA and band statistics -- and are split off by row afterwards:

  * ``miniecoset_val``: the first 50 MiniEcoset validation images of each of the
    16 Geirhos categories (800 images), natural photographs the networks were
    not trained on and that share no exemplar with the Geirhos set. Their
    predicted patterns are the category templates of the cortex-to-decision
    test.
  * ``same_cue``: the 80 images of the Geirhos release whose shape and texture
    category coincide, which the behavioural set excludes. They are the
    manipulation check (does the template space carry category at all?) and
    are never scored against the observers.

Images of different sizes (512-px PNGs, 256-px h5 rows) are returned as separate
arrays; the front end resizes each set to 256 px (``dkl_frontend.rgb_to_dkl3``).
"""

from __future__ import annotations

import glob
import os
import re

import numpy as np

TEMPLATE_SOURCE = "miniecoset_val"
SAME_CUE_SOURCE = "same_cue"
N_PER_CLASS = 50
_PAIR = re.compile(r"([a-z]+)\d+-([a-z]+)\d+$")


def cue_conflict_stems() -> list[str]:
    """Stems of the behavioural set, in the order ``cue_conflict_design`` uses.

    The same listing (``list_cue_conflict_images``) orders both, so a stored
    prediction row i and the observers' per-image profile for stem i refer to
    the same image; ``cortex_decision`` asserts the match.
    """
    from src.analysis.behavior import cue_conflict

    return [
        os.path.splitext(os.path.basename(p))[0]
        for p, _, _ in cue_conflict.list_cue_conflict_images()
    ]


def same_cue_images() -> list[tuple[str, str]]:
    """[(path, category)] of the same-cue images (shape == texture)."""
    from src.analysis.behavior import cue_conflict

    out = []
    paths = sorted(
        glob.glob(os.path.join(cue_conflict.CUE_CONFLICT_DIR, "**", "*.png"),
                  recursive=True)
    )
    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        m = _PAIR.match(stem)
        if not m or m.group(1) != m.group(2):
            continue
        if m.group(1) in cue_conflict.GEIRHOS_CLASSES:
            out.append((path, m.group(1)))
    return out


def miniecoset16_images(
    n_per_class: int = N_PER_CLASS, split: str = "val"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(rgb_uint8, labels, stems): the first ``n_per_class`` images of each
    Geirhos category in the h5 split, in h5 row order."""
    import h5py

    from src.analysis.behavior import cue_conflict

    index = cue_conflict.load_class_index()
    with h5py.File(cue_conflict.MINIECOSET_H5, "r") as f:
        labels_all = f[f"{split}/labels"][:]
        rows, labels = [], []
        for cat in cue_conflict.GEIRHOS_CLASSES:
            idx = np.flatnonzero(labels_all == index[cat])[:n_per_class]
            if len(idx) < n_per_class:
                raise RuntimeError(
                    f"{cat}: {len(idx)} {split} images, {n_per_class} needed"
                )
            rows.extend(int(i) for i in idx)
            labels.extend([cat] * len(idx))
        order = np.argsort(rows, kind="stable")  # h5py wants increasing rows
        rows = np.asarray(rows)[order]
        labels = np.asarray(labels)[order]
        rgb = f[f"{split}/data"][rows]
    stems = np.asarray([f"{split}/{r}" for r in rows])
    return np.asarray(rgb, dtype=np.uint8), labels, stems


def build(name: str) -> dict:
    """The template design: ``rgb`` is a list of uint8 arrays (one per image
    set), ``labels`` / ``source`` / ``stems`` are aligned over the concatenation
    of those sets in order."""
    if name != "miniecoset16":
        raise ValueError(f"unknown template design {name!r}")
    from src.analysis.behavior import cue_conflict

    rgb_t, lab_t, stem_t = miniecoset16_images()
    sc = same_cue_images()
    if not sc:
        raise RuntimeError(
            f"no same-cue images under {cue_conflict.CUE_CONFLICT_DIR}"
        )
    rgb_s = cue_conflict.load_images([p for p, _ in sc])
    lab_s = np.asarray([c for _, c in sc])
    stem_s = np.asarray(
        [os.path.splitext(os.path.basename(p))[0] for p, _ in sc]
    )
    return {
        "rgb": [rgb_t, rgb_s],
        "labels": np.concatenate([lab_t, lab_s]),
        "source": np.asarray(
            [TEMPLATE_SOURCE] * len(lab_t) + [SAME_CUE_SOURCE] * len(lab_s)
        ),
        "stems": np.concatenate([stem_t, stem_s]),
    }
