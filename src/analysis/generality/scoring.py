"""Cue-conflict scorers shared by the generality analyses.

  miniecoset112  the paper's primary estimator (Methods 4.3): logits restricted
                 to the 16 MiniEcoset columns, argmax, rate over images decided
                 as the shape or the texture category; plus the forced two-way
                 estimator (shape-column logit > texture-column logit, every
                 image counted).
  imagenet16     the Geirhos decision rule for ImageNet networks: softmax over
                 1,000 classes, mean probability within each of the 16
                 categories (imagenet16.json), argmax over 16. Every image is
                 decided, so primary and forced coincide up to the aggregation.
"""
from __future__ import annotations

import json
import os

import numpy as np

from src.analysis.behavior.cue_conflict import GEIRHOS_CLASSES, load_class_index

_HERE = os.path.dirname(__file__)


class Scorer:
    def __init__(self, kind: str):
        self.kind = kind
        if kind == "miniecoset112":
            self.class_index = load_class_index()
            self.cols = np.array([self.class_index[c] for c in GEIRHOS_CLASSES])
        elif kind == "imagenet16":
            cats = json.load(open(os.path.join(_HERE, "imagenet16.json")))["categories"]
            self.members = [np.array(cats[c]) for c in GEIRHOS_CLASSES]
        else:
            raise ValueError(kind)
        self.cat_of = {c: i for i, c in enumerate(GEIRHOS_CLASSES)}

    def _restricted(self, logits: np.ndarray) -> np.ndarray:
        """(N, 16) decision scores in GEIRHOS_CLASSES order."""
        if self.kind == "miniecoset112":
            return logits[:, self.cols]
        z = logits - logits.max(1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(1, keepdims=True)
        return np.stack([p[:, m].mean(1) for m in self.members], 1)

    def score(self, logits: np.ndarray, shapes, textures) -> dict:
        s_idx = np.array([self.cat_of[s] for s in shapes])
        t_idx = np.array([self.cat_of[t] for t in textures])
        r = self._restricted(np.asarray(logits, dtype=np.float64))
        dec = r.argmax(1)
        n_shape = int((dec == s_idx).sum())
        n_texture = int((dec == t_idx).sum())
        denom = n_shape + n_texture
        rows = np.arange(len(s_idx))
        forced = float(np.mean(r[rows, s_idx] > r[rows, t_idx]))
        return {
            "shape_bias": (n_shape / denom) if denom else float("nan"),
            "forced_two_way": forced,
            "n_shape": n_shape, "n_texture": n_texture,
            "n_other": int(len(s_idx) - denom), "n_images": int(len(s_idx)),
            "shape_accuracy": n_shape / len(s_idx), "texture_accuracy": n_texture / len(s_idx),
        }
