"""Group-level inference for the stream x cortical-axis dissociation.

Unit of inference is the SUBJECT (n=8). For each subject and ROI we
summarize noise-ceiling-normalized unique variance per stream, then test:
  * simple effects (magno vs parvo within each ROI): Wilcoxon across subjects
  * the interaction (difference of differences between two ROI classes):
    permutation / Wilcoxon across subjects
A double dissociation claim requires the interaction AND both simple
effects (Nieuwenhuis et al. 2011).
"""

from __future__ import annotations

import glob
import os

import numpy as np
from scipy import stats as sps

STREAMS_LABELS = {
    1: "early",
    2: "midventral",
    3: "midlateral",
    4: "midparietal",
    5: "ventral",
    6: "lateral",
    7: "parietal",
}
ECC_LABELS = {1: "ecc_0.5", 2: "ecc_1", 3: "ecc_2", 4: "ecc_4", 5: "ecc_4+"}


def load_results(out_dir: str, model_name: str) -> dict[int, dict]:
    """Load per-subject npz results for one model."""
    results = {}
    for path in sorted(
        glob.glob(os.path.join(out_dir, f"{model_name}_subj*.npz"))
    ):
        z = np.load(path, allow_pickle=True)
        results[int(z["subject"])] = dict(z)
    return results


def roi_summary(
    res: dict,
    roi_key: str,
    labels: dict[int, str],
    value_keys=("unique_magno", "unique_parvo"),
    nc_min: float = 0.1,
    statistic=np.nanmedian,
) -> dict[str, dict[str, float]]:
    """Per-ROI summary of voxelwise values for one subject.

    Voxels included: ROI label > 0, noise ceiling above nc_min, and finite.
    """
    roi = res[roi_key]
    nc = res["noise_ceiling"]
    out: dict[str, dict[str, float]] = {}
    for lab, name in labels.items():
        vox = (roi == lab) & (nc > nc_min)
        out[name] = {}
        for k in value_keys:
            v = res[k][vox]
            out[name][k] = float(statistic(v[np.isfinite(v)]))
        out[name]["n_voxels"] = int(vox.sum())
    return out


def collect_group_table(
    results: dict[int, dict], roi_key: str, labels: dict[int, str], **kw
):
    """(n_subjects, n_rois) arrays for magno- and parvo-unique variance."""
    subjects = sorted(results)
    names = list(labels.values())
    um = np.full((len(subjects), len(names)), np.nan)
    up = np.full((len(subjects), len(names)), np.nan)
    for i, s in enumerate(subjects):
        summ = roi_summary(results[s], roi_key, labels, **kw)
        for j, name in enumerate(names):
            um[i, j] = summ[name]["unique_magno"]
            up[i, j] = summ[name]["unique_parvo"]
    return subjects, names, um, up


def simple_effect(um_roi: np.ndarray, up_roi: np.ndarray):
    """Magno vs parvo within one ROI, across subjects (paired Wilcoxon)."""
    d = um_roi - up_roi
    d = d[np.isfinite(d)]
    stat, p = sps.wilcoxon(d)
    return {"median_diff": float(np.median(d)), "W": float(stat), "p": float(p)}


def interaction(
    um: np.ndarray, up: np.ndarray, rois_a: list[int], rois_b: list[int],
    n_perm: int = 10000, seed: int = 0,
):
    """Difference-of-differences across subjects with sign-flip permutation.

    d_s = (U_m - U_p)|A  -  (U_m - U_p)|B  per subject; H0: median d = 0.
    """
    diff = (um - up)
    d = np.nanmean(diff[:, rois_a], axis=1) - np.nanmean(
        diff[:, rois_b], axis=1
    )
    d = d[np.isfinite(d)]
    obs = np.mean(d)
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    null = (signs * d[None, :]).mean(axis=1)
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)
    return {"mean_dod": float(obs), "p_perm": float(p), "per_subject": d}


def fdr(pvals: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg: boolean array of rejected hypotheses."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    thresh = q * (np.arange(1, m + 1) / m)
    passed = p[order] <= thresh
    k = np.max(np.nonzero(passed)[0]) + 1 if passed.any() else 0
    out = np.zeros(m, dtype=bool)
    out[order[:k]] = True
    return out
