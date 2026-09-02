"""Extract everything Figure 1 needs into one small cache.

Loading eight subject npz files per model is slow, and the figure needs
several passes over the same arrays, so the extraction is separated from the
drawing. Run this once; iterate on fig1.py freely.

Writes results/figures/fig1_data.npz
"""

from __future__ import annotations

import json
import os

import numpy as np
from scipy import stats as sps

from src.analysis.encoding.stats import load_results

# (key, model name, cell) -- cell drives colour/grouping in the figure
MODELS = [
    ("untrained_diet",   "untrained_seed0",         "untrained"),
    ("untrained_nodiet", "untrained_seed0_noblur",  "untrained"),
    ("nodiet_s0",        "dkl_base_seed0_trained",  "trained_nodiet"),
    ("nodiet_s1",        "dkl_base_seed1_trained",  "trained_nodiet"),
    ("diet_thesis",      "thesis_dvd_trans",        "trained_diet"),
    ("diet_ours_s0",     "dkl_bcs_seed0_trained",   "trained_diet"),
    ("swap_s0",          "dkl_bcs_swap_seed0_trained", "trained_swap"),
]
VIS = [1, 2, 3, 4, 5, 6, 7]          # V1v V1d V2v V2d V3v V3d hV4
HIER = np.array([1, 1, 2, 2, 3, 3, 4], float)
NC_MIN, MIN_VOX = 0.1, 20


def norm_stat(z, sel):
    """Scale-free stream preference: median(Um-Up) / median(Um+Up)."""
    um, up = z["unique_magno"], z["unique_parvo"]
    v = sel & (z["noise_ceiling"] > NC_MIN) & np.isfinite(um) & np.isfinite(up)
    if v.sum() < MIN_VOX:
        return np.nan
    a, b = um[v], up[v]
    den = np.median(a + b)
    return np.median(a - b) / den if den > 1e-4 else np.nan


out = {}
for key, model, cell in MODELS:
    R = load_results("results/encoding", model)
    subs = sorted(R)
    prof = np.array([[norm_stat(R[s], R[s]["roi_prf-visualrois"] == k)
                      for k in VIS] for s in subs])
    centred = prof - np.nanmean(prof, axis=1, keepdims=True)
    rho = np.array([sps.spearmanr(HIER[np.isfinite(r)],
                                  r[np.isfinite(r)]).statistic
                    for r in prof])
    out[f"{key}__profile"] = prof
    out[f"{key}__centred"] = centred
    out[f"{key}__rho"] = rho
    out[f"{key}__subs"] = np.array(subs)
    out[f"{key}__cell"] = np.array([cell])
    print(f"{model:<26} n={len(subs)}  hV4-V1v={np.nanmean(prof[:, 6] - prof[:, 0]):+.4f}  "
          f"rho={rho.mean():+.3f}", flush=True)

# Leg (a): stream shape/texture bias from the probe
probe = {
    "untrained_nodiet": ["untrained_noblur"],
    "untrained_diet":   ["untrained_blur_cs"],
    # Same architecture, transform moved to rg: the untrained end of the
    # swap slope. magno here should reproduce untrained_noblur's magno
    # exactly, since both hand stream 1 untransformed luminance.
    "untrained_swap":   ["untrained_swap"],
    "trained_nodiet":   [f"dkl_base_seed{s}_trained" for s in (0, 1, 2)],
    "trained_diet":     [f"dkl_bcs_seed{s}_trained" for s in (0, 1, 2)]
                        + ["thesis_dvd_trans"],
    # The channel-swap control: the identical developmental transform, moved
    # onto the red-green channel. It is the arm that separates "the diet does
    # this" from "the diet does this to an ACHROMATIC channel", so panel C is
    # not interpretable without it.
    "trained_swap":     [f"dkl_bcs_swap_seed{s}_trained" for s in (0, 1, 2)],
}
for cell, names in probe.items():
    m, p = [], []
    for n in names:
        fp = f"results/shape_texture/{n}.json"
        if not os.path.exists(fp):
            print(f"  missing {fp}")
            continue
        d = json.load(open(fp))
        m.append(d["streams"]["magno"]["bias"])
        p.append(d["streams"]["parvo"]["bias"])
    out[f"probe__{cell}__magno"] = np.array(m)
    out[f"probe__{cell}__parvo"] = np.array(p)
    print(f"probe {cell:<18} magno {np.mean(m):+.3f}  parvo {np.mean(p):+.3f}  "
          f"(n={len(m)})")

os.makedirs("results/figures", exist_ok=True)
np.savez("results/figures/fig1_data.npz", **out)
print("\nwrote results/figures/fig1_data.npz")
