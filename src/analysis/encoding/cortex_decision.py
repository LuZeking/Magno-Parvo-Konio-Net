"""The cortex-to-decision test (docs/PLAN_CORTEX_DECISION_2026-10-26.md).

Every quantity here is defined in the plan's §2–§4 and nothing is chosen after
seeing data. Inputs are the refit encoding files written by ``run_encoding``
with ``--save-predictions --template-design miniecoset16`` (the per-image
predicted response of every voxel to the 1,200 cue-conflict images, to the 80
same-cue images and to the 800 MiniEcoset templates), the ROI labels stored in
the same files, and the observers' per-image profiles and ceilings of
``tmp_analysis/human_perimage.py``.

Per run x subject x ROI x band:
    E_ic  Fisher-z Pearson correlation over voxels between image i's predicted
          pattern and category c's template (voxels normalised by the
          templates' mean and s.d.)
    m_i   E_i,shape(i) - E_i,texture(i)                  the shape margin
    c_i   max(E_i,shape, E_i,texture) - max over the other 14   the cue margin
    B     mean_i [m_i > 0]                               cortical shape-bias rate
    r_h   Pearson(resid(m), resid(h)),  r_v = Pearson(resid(c), resid(v)),
          each as % of the observers' split-half ceiling (Fig. 5c's unit)

Seeds are averaged per subject at the level of m_i and c_i; seeds are never
pooled with subjects. Exact tests are enumerated here (Wilcoxon signed-rank at
n = 8, Mann-Whitney 3 v 3) so that no library version decides a floor.

Usage:
    PYTHONPATH=. python -m src.analysis.encoding.cortex_decision \\
        --fits-dir /scratch/byrong/mpk/encoding/results/2026-10-26_ctd \\
        --human-dir /scratch/byrong/mpk/perimage_behavior/results/2026-09-07_geirhos_perimage \\
        --out-dir <dir>
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re

import numpy as np

ROIS = {
    "V1": ("prf-visualrois", (1, 2)),
    "V2": ("prf-visualrois", (3, 4)),
    "V3": ("prf-visualrois", (5, 6)),
    "hV4": ("prf-visualrois", (7,)),
    "ventral": ("streams", (5,)),
    "lateral": ("streams", (6,)),
    "parietal": ("streams", (7,)),
}
EXTRASTRIATE = ("hV4", "ventral", "lateral")
BANDS = ("joint", "magno", "parvo")
ARMS = {
    "model": ["fs2_dvd300_avg_w050_ls_seed{}".format(s) for s in (0, 1, 2)],
    "control": ["fs2_dvdfix300_avg_w050_ls_seed{}".format(s) for s in (0, 1, 2)],
    "swap": ["fs2_dvd300swap_avg_w050_ls_seed{}".format(s) for s in (0, 1, 2)],
    "untrained": ["untrained_dvd300_seed0"],
    "gabor": ["gabor_dvd300"],
}
SUBJECTS = tuple(range(1, 9))
MIN_VOX = 20
BEHAV_RATE_MODEL = 0.682      # Table 1, the model's integrated readout
TEMPLATE_SOURCE = "miniecoset_val"
SAME_CUE_SOURCE = "same_cue"
N_PERM = 2000
N_BOOT_IMAGES = 2000
N_BOOT_EXEMPLAR = 1000
GATE_P = 0.01
_PAIR = re.compile(r"([a-z]+)\d+-([a-z]+)\d+$")


# ----------------------------------------------------------------- arithmetic
def residualise(y: np.ndarray, shapes, textures) -> np.ndarray:
    """Remove shape- and texture-category main effects (additive design with
    one indicator per category, intercept absorbed). Mirrors
    ``tmp_analysis/human_perimage.residualise``, which the observers' profiles
    were residualised with; ``test_cortex_decision`` asserts the two agree."""
    y = np.asarray(y, float)
    finite = np.isfinite(y)
    sh = sorted(set(shapes))
    tx = sorted(set(textures))
    X = np.zeros((len(y), 1 + len(sh) + len(tx)))
    X[:, 0] = 1.0
    for j, c in enumerate(sh):
        X[np.asarray(shapes) == c, 1 + j] = 1.0
    for j, c in enumerate(tx):
        X[np.asarray(textures) == c, 1 + len(sh) + j] = 1.0
    beta, *_ = np.linalg.lstsq(X[finite], y[finite], rcond=None)
    out = np.full_like(y, np.nan)
    out[finite] = y[finite] - X[finite] @ beta
    return out


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan


def pearson_rows(M: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pearson r of every row of M with y."""
    A = M - M.mean(axis=1, keepdims=True)
    b = y - y.mean()
    den = np.sqrt((A * A).sum(axis=1) * (b * b).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, A @ b / den, np.nan)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import rankdata
    m = np.isfinite(a) & np.isfinite(b)
    return pearson(rankdata(a[m]), rankdata(b[m]))


def fisher_z(r: np.ndarray) -> np.ndarray:
    return np.arctanh(np.clip(r, -0.999999, 0.999999))


def template_evidence(
    P: np.ndarray, Q: np.ndarray, q_labels: np.ndarray, classes: list[str],
    method: str = "pearson",
) -> np.ndarray:
    """E (n_images, n_classes): Fisher-z correlation over voxels of each test
    pattern with each category template (plan §2.1-2.3).

    Voxels are centred and scaled by the templates' mean and s.d. (zero-s.d.
    voxels dropped); templates are the class means of the normalised template
    patterns. ``method`` 'spearman' ranks each pattern over voxels first (the
    plan's sensitivity analysis 1).
    """
    P = np.asarray(P, np.float64)
    Q = np.asarray(Q, np.float64)
    mu = Q.mean(axis=0)
    sd = Q.std(axis=0)
    keep = sd > 0
    if keep.sum() < 2:
        raise ValueError("fewer than two voxels with template variance")
    Pn = (P[:, keep] - mu[keep]) / sd[keep]
    Qn = (Q[:, keep] - mu[keep]) / sd[keep]
    T = np.stack([Qn[np.asarray(q_labels) == c].mean(axis=0) for c in classes])
    if method == "spearman":
        from scipy.stats import rankdata
        Pn = rankdata(Pn, axis=1)
        T = rankdata(T, axis=1)
    elif method != "pearson":
        raise ValueError(method)
    Pc = Pn - Pn.mean(axis=1, keepdims=True)
    Tc = T - T.mean(axis=1, keepdims=True)
    Pc /= np.maximum(np.linalg.norm(Pc, axis=1, keepdims=True), 1e-12)
    Tc /= np.maximum(np.linalg.norm(Tc, axis=1, keepdims=True), 1e-12)
    return fisher_z(Pc @ Tc.T)


def margins(
    E: np.ndarray, shapes: np.ndarray, textures: np.ndarray, classes: list[str]
) -> dict[str, np.ndarray]:
    """Shape margin m, cue margin c, and the two evidences (plan §2.3-2.4)."""
    idx = {c: j for j, c in enumerate(classes)}
    js = np.asarray([idx[s] for s in shapes])
    jt = np.asarray([idx[t] for t in textures])
    rows = np.arange(len(E))
    e_s = E[rows, js]
    e_t = E[rows, jt]
    other = E.copy()
    other[rows, js] = -np.inf
    other[rows, jt] = -np.inf
    c = np.maximum(e_s, e_t) - other.max(axis=1)
    return {"e_shape": e_s, "e_texture": e_t, "m": e_s - e_t, "c": c}


def same_cue_gate(E_sc: np.ndarray, labels_sc: np.ndarray, classes: list[str]):
    """Nearest-template accuracy on the same-cue images and its exact binomial
    p against chance 1/16 (plan §2.7)."""
    from scipy.stats import binomtest
    pred = np.asarray(classes)[E_sc.argmax(axis=1)]
    k = int((pred == np.asarray(labels_sc)).sum())
    n = len(labels_sc)
    p = binomtest(k, n, 1.0 / len(classes), alternative="greater").pvalue
    return k / n, float(p)


# ----------------------------------------------------------------- exact tests
def exact_wilcoxon(x: np.ndarray) -> dict[str, float]:
    """Exact signed-rank test of median(x) = 0 by enumerating all sign
    assignments (n <= 14; zeros dropped, ties mid-ranked). Returns one-sided
    'greater'/'less' and two-sided p."""
    from scipy.stats import rankdata
    x = np.asarray(x, float)
    x = x[np.isfinite(x) & (x != 0)]
    n = len(x)
    if n == 0:
        return {"n": 0, "p_greater": np.nan, "p_less": np.nan, "p_two": np.nan, "w_plus": np.nan}
    if n > 14:
        raise ValueError("enumeration is for small n")
    ranks = rankdata(np.abs(x))
    w_obs = float(ranks[x > 0].sum())
    signs = np.array(list(itertools.product((0, 1), repeat=n)), dtype=float)
    w_null = signs @ ranks
    p_greater = float((w_null >= w_obs - 1e-9).mean())
    p_less = float((w_null <= w_obs + 1e-9).mean())
    return {"n": n, "w_plus": w_obs, "p_greater": p_greater, "p_less": p_less,
            "p_two": float(min(1.0, 2 * min(p_greater, p_less)))}


def exact_mannwhitney(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Exact Mann-Whitney by enumerating every assignment of the pooled
    values to the two groups (n_a + n_b <= 12). U counts pairs a > b (ties
    0.5). 'p_greater' is P(U >= U_obs) under the null."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    pooled = np.concatenate([a, b])
    na = len(a)

    def u_of(ia):
        ia = np.asarray(ia)
        mask = np.zeros(len(pooled), bool)
        mask[ia] = True
        x, y = pooled[mask], pooled[~mask]
        d = x[:, None] - y[None, :]
        return float((d > 0).sum() + 0.5 * (d == 0).sum())

    u_obs = u_of(np.arange(na))
    us = np.array([u_of(c) for c in itertools.combinations(range(len(pooled)), na)])
    p_greater = float((us >= u_obs - 1e-9).mean())
    p_less = float((us <= u_obs + 1e-9).mean())
    return {"u": u_obs, "n_assignments": len(us), "p_greater": p_greater,
            "p_less": p_less, "p_two": float(min(1.0, 2 * min(p_greater, p_less)))}


# ----------------------------------------------------------------- resampling
def cell_ids(shapes, textures) -> np.ndarray:
    keys = [f"{s}|{t}" for s, t in zip(shapes, textures)]
    _, inv = np.unique(keys, return_inverse=True)
    return inv


def within_cell_permutations(cells: np.ndarray, n_perm: int, seed: int = 0) -> np.ndarray:
    """(n_perm, n_images) index arrays permuting images within each cell, one
    shared sequence (plan §4)."""
    rng = np.random.default_rng(seed)
    n = len(cells)
    members = [np.flatnonzero(cells == c) for c in np.unique(cells)]
    out = np.empty((n_perm, n), dtype=np.int64)
    for k in range(n_perm):
        idx = np.arange(n)
        for mem in members:
            idx[mem] = rng.permutation(mem)
        out[k] = idx
    return out


def perm_p(r_obs: float, r_null: np.ndarray) -> float:
    return float((1 + np.sum(np.abs(r_null) >= abs(r_obs) - 1e-12)) / (1 + len(r_null)))


def object_exemplar(stems) -> np.ndarray:
    """Object-exemplar id of each image ('airplane1' of 'airplane1-bicycle2')."""
    out = []
    for s in stems:
        m = _PAIR.search(str(s))
        if not m:
            raise ValueError(f"bad stem {s!r}")
        out.append(m.group(0).split("-")[0])
    return np.asarray(out)


def cluster_bootstrap_counts(clusters, n_boot: int, seed: int = 1):
    """(counts, inverse): how often each cluster is drawn in each of n_boot
    resamples of the clusters with replacement, and each image's cluster."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(np.asarray(clusters), return_inverse=True)
    counts = rng.multinomial(len(uniq), np.full(len(uniq), 1.0 / len(uniq)), size=n_boot)
    return counts, inv


def cluster_bootstrap_rs(x, y, clusters, n_boot: int, seed: int = 1) -> np.ndarray:
    """Pearson r of every cluster resample (object exemplars drawn with
    replacement; images inherit their cluster's draw). A resample is a count
    per cluster, so its correlation is the count-weighted correlation of the
    original images, computed for all draws at once (``test_cortex_decision``
    checks this against explicit resampling)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    counts, inv = cluster_bootstrap_counts(clusters, n_boot, seed)
    W = counts[:, inv].astype(float)                     # (n_boot, n_images)
    n = W.sum(axis=1)
    mx = W @ x / n
    my = W @ y / n
    sxx = W @ (x * x) / n - mx ** 2
    syy = W @ (y * y) / n - my ** 2
    sxy = W @ (x * y) / n - mx * my
    with np.errstate(invalid="ignore", divide="ignore"):
        return sxy / np.sqrt(sxx * syy)


def cluster_bootstrap_r(x, y, clusters, n_boot: int, seed: int = 1):
    """Percentile CI of Pearson r under the cluster bootstrap."""
    rs = cluster_bootstrap_rs(x, y, clusters, n_boot, seed)
    return float(np.nanpercentile(rs, 2.5)), float(np.nanpercentile(rs, 97.5))


def image_bootstrap_rate(m: np.ndarray, n_boot: int, seed: int = 2):
    rng = np.random.default_rng(seed)
    pos = (m > 0).astype(float)
    counts = rng.multinomial(len(m), np.full(len(m), 1.0 / len(m)), size=n_boot)
    rates = counts @ pos / len(m)
    return float(np.percentile(rates, 2.5)), float(np.percentile(rates, 97.5))


# ----------------------------------------------------------------- loading
def load_human(human_dir: str) -> dict:
    H = np.load(os.path.join(human_dir, "human_perimage.npz"), allow_pickle=False)
    with open(os.path.join(human_dir, "human_ceiling.json")) as fh:
        ceil = json.load(fh)["ceilings"]
    stems = np.asarray([str(s) for s in H["stems"]])
    return {
        "stems": stems,
        "shapes": np.asarray([str(s) for s in H["shapes"]]),
        "textures": np.asarray([str(s) for s in H["textures"]]),
        "h": np.asarray(H["h"], float), "v": np.asarray(H["v"], float),
        "h_resid": np.asarray(H["h_resid"], float),
        "v_resid": np.asarray(H["v_resid"], float),
        "ceiling_h": float(ceil["shape_rate_resid"]["pearson_mean"]),
        "ceiling_v": float(ceil["valid_rate_resid"]["pearson_mean"]),
        "ceiling_h_raw": float(ceil["shape_rate"]["pearson_mean"]),
        "ceiling_v_raw": float(ceil["valid_rate"]["pearson_mean"]),
    }


def roi_voxels(z, roi: str, reliable: bool = True) -> np.ndarray:
    atlas, labels = ROIS[roi]
    lab = z[f"roi_{atlas}"]
    sel = np.isin(lab, labels)
    if reliable:
        sel &= z["reliable_mask"]
    return np.flatnonzero(sel)


def load_fit(path: str, repro_tol: float) -> dict | None:
    """The arrays one file contributes, or None (with a reason) if the file
    fails its reproduction gate."""
    z = np.load(path, allow_pickle=False)
    out = {"path": path, "subject": int(z["subject"]), "model_name": str(z["model_name"])}
    if "repro_max_abs_diff" in z.files:
        d = float(z["repro_max_abs_diff"])
        out["repro_max_abs_diff"] = d
        if d > repro_tol:
            out["excluded"] = f"repro_max_abs_diff {d:.3e} > {repro_tol:g}"
            return out
    if "cc_pred_joint" not in z.files or "tpl_pred_joint" not in z.files:
        out["excluded"] = "no stored predictions / templates"
        return out
    out["stems"] = np.asarray([str(s) for s in z["cc_stems"]])
    out["shapes"] = np.asarray([str(s) for s in z["cc_shapes"]])
    out["textures"] = np.asarray([str(s) for s in z["cc_textures"]])
    out["tpl_labels"] = np.asarray([str(s) for s in z["tpl_labels"]])
    out["tpl_source"] = np.asarray([str(s) for s in z["tpl_source"]])
    for band in BANDS:
        out[f"cc_{band}"] = np.asarray(z[f"cc_pred_{band}"], np.float32)
        out[f"tpl_{band}"] = np.asarray(z[f"tpl_pred_{band}"], np.float32)
    out["r2_joint"] = np.asarray(z["r2_joint"], float)
    out["z"] = z
    return out


# ----------------------------------------------------------------- the cells
def cell_margins(fit: dict, vox: np.ndarray, band: str, classes: list[str],
                 method: str = "pearson") -> dict:
    is_t = fit["tpl_source"] == TEMPLATE_SOURCE
    is_s = fit["tpl_source"] == SAME_CUE_SOURCE
    Q = fit[f"tpl_{band}"][is_t][:, vox]
    E = template_evidence(fit[f"cc_{band}"][:, vox], Q, fit["tpl_labels"][is_t], classes, method)
    E_sc = template_evidence(fit[f"tpl_{band}"][is_s][:, vox], Q, fit["tpl_labels"][is_t], classes, method)
    out = margins(E, fit["shapes"], fit["textures"], classes)
    out["E_same_cue"] = E_sc
    out["labels_same_cue"] = fit["tpl_labels"][is_s]
    return out


def summarise_margins(m, c, human, classes_same_cue=None, E_sc=None, labels_sc=None,
                      clusters=None, full=True) -> dict:
    """B, mean margin, correspondence (raw and residual, % of ceiling), the
    gate, and CIs for one (arm, subject, ROI, band)."""
    sh, tx = human["shapes"], human["textures"]
    m_res = residualise(m, sh, tx)
    c_res = residualise(c, sh, tx)
    r_h = pearson(m_res, human["h_resid"])
    r_v = pearson(c_res, human["v_resid"])
    out = {
        "B": float(np.mean(m > 0)),
        "mean_m": float(np.mean(m)),
        "r_h_resid": r_h, "pct_h": 100 * r_h / human["ceiling_h"],
        "r_v_resid": r_v, "pct_v": 100 * r_v / human["ceiling_v"],
        "r_h_raw": pearson(m, human["h"]), "r_v_raw": pearson(c, human["v"]),
        "rho_h_resid": spearman(m_res, human["h_resid"]),
        "rho_v_resid": spearman(c_res, human["v_resid"]),
    }
    if E_sc is not None:
        acc, p = same_cue_gate(E_sc, labels_sc, classes_same_cue)
        out["gate_acc"] = acc
        out["gate_p"] = p
        out["gate_pass"] = bool(p < GATE_P)
    if full:
        out["B_ci"] = image_bootstrap_rate(m, N_BOOT_IMAGES)
        if clusters is not None:
            out["r_h_resid_ci_exemplar"] = cluster_bootstrap_r(m_res, human["h_resid"], clusters, N_BOOT_EXEMPLAR)
            out["r_v_resid_ci_exemplar"] = cluster_bootstrap_r(c_res, human["v_resid"], clusters, N_BOOT_EXEMPLAR)
    return out


# ----------------------------------------------------------------- driver
def run(fits_dir: str, human_dir: str, out_dir: str, repro_tol: float = 1e-3,
        method: str = "pearson", reliable: bool = True, n_perm: int = N_PERM,
        arms: dict | None = None, v1_streams: bool = False) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    if v1_streams:                      # plan §4, sensitivity analysis 2
        ROIS["V1"] = ("streams", (1,))
    human = load_human(human_dir)
    arms = arms or ARMS
    classes = sorted(set(human["shapes"]) | set(human["textures"]))
    cells = cell_ids(human["shapes"], human["textures"])
    clusters = object_exemplar(human["stems"])
    perms = within_cell_permutations(cells, n_perm)
    h_perm = human["h_resid"][perms]            # (n_perm, n_images)
    v_perm = human["v_resid"][perms]

    # margins[arm][subject][seed_run][roi][band] = {"m":, "c":, "E_same_cue":, ...}
    store: dict = {}
    files: list[dict] = []
    for arm, runs in arms.items():
        for run in runs:
            for s in SUBJECTS:
                path = os.path.join(fits_dir, f"{run}_subj{s:02d}.npz")
                if not os.path.exists(path):
                    files.append({"run": run, "subject": s, "excluded": "missing"})
                    continue
                fit = load_fit(path, repro_tol)
                rec = {"run": run, "subject": s, "repro_max_abs_diff": fit.get("repro_max_abs_diff")}
                if fit.get("excluded"):
                    rec["excluded"] = fit["excluded"]
                    files.append(rec)
                    continue
                if not np.array_equal(fit["stems"], human["stems"]):
                    raise RuntimeError(f"{path}: cue-conflict stems differ from the observers' listing")
                rec["n_vox_roi"] = {}
                for roi in ROIS:
                    vox = roi_voxels(fit["z"], roi, reliable)
                    rec["n_vox_roi"][roi] = int(len(vox))
                    if len(vox) < MIN_VOX:
                        continue
                    for band in BANDS:
                        cm = cell_margins(fit, vox, band, classes, method)
                        cm["n_vox"] = int(len(vox))
                        cm["r2_median"] = float(np.nanmedian(fit["r2_joint"][vox]))
                        store.setdefault(arm, {}).setdefault(s, {}).setdefault(run, {}).setdefault(roi, {})[band] = cm
                files.append(rec)
                print(f"[ctd] {run} subj{s:02d}: {rec['n_vox_roi']}", flush=True)

    # seed-averaged per subject, plus per-seed summaries
    results: dict = {"per_subject": {}, "per_seed": {}}
    for arm in store:
        for s in store[arm]:
            for roi in ROIS:
                for band in BANDS:
                    seeds = [store[arm][s][run][roi][band] for run in store[arm][s]
                             if roi in store[arm][s][run] and band in store[arm][s][run][roi]]
                    if not seeds:
                        continue
                    for run in store[arm][s]:
                        if roi in store[arm][s][run] and band in store[arm][s][run][roi]:
                            cm = store[arm][s][run][roi][band]
                            results["per_seed"].setdefault(arm, {}).setdefault(run, {}).setdefault(str(s), {}).setdefault(roi, {})[band] = \
                                summarise_margins(cm["m"], cm["c"], human, classes, cm["E_same_cue"], cm["labels_same_cue"], full=False) | \
                                {"n_vox": cm["n_vox"], "r2_median": cm["r2_median"]}
                    m = np.mean([cm["m"] for cm in seeds], axis=0)
                    c = np.mean([cm["c"] for cm in seeds], axis=0)
                    E_sc = np.mean([cm["E_same_cue"] for cm in seeds], axis=0)
                    summ = summarise_margins(m, c, human, classes, E_sc, seeds[0]["labels_same_cue"], clusters, full=True)
                    summ["n_seeds"] = len(seeds)
                    summ["n_vox"] = int(np.mean([cm["n_vox"] for cm in seeds]))
                    summ["r2_median"] = float(np.mean([cm["r2_median"] for cm in seeds]))
                    summ["_m_resid"] = residualise(m, human["shapes"], human["textures"])
                    summ["_c_resid"] = residualise(c, human["shapes"], human["textures"])
                    results["per_subject"].setdefault(arm, {}).setdefault(str(s), {}).setdefault(roi, {})[band] = summ

    # permutation null on the subject-mean correspondence (joint band), per arm x ROI
    results["permutation"] = {}
    for arm in results["per_subject"]:
        for roi in ROIS:
            subs = [results["per_subject"][arm][s][roi]["joint"]
                    for s in results["per_subject"][arm] if roi in results["per_subject"][arm][s]]
            if not subs:
                continue
            rh = np.mean([pearson_rows(h_perm, d["_m_resid"]) for d in subs], axis=0)
            rv = np.mean([pearson_rows(v_perm, d["_c_resid"]) for d in subs], axis=0)
            obs_h = float(np.mean([d["r_h_resid"] for d in subs]))
            obs_v = float(np.mean([d["r_v_resid"] for d in subs]))
            results["permutation"].setdefault(arm, {})[roi] = {
                "r_h_mean": obs_h, "p_h": perm_p(obs_h, rh), "null_h_97.5": float(np.percentile(np.abs(rh), 97.5)),
                "r_v_mean": obs_v, "p_v": perm_p(obs_v, rv), "null_v_97.5": float(np.percentile(np.abs(rv), 97.5)),
                "n_subjects": len(subs), "n_perm": int(n_perm),
            }
    for arm in results["per_subject"]:
        for s in results["per_subject"][arm]:
            for roi in results["per_subject"][arm][s]:
                for band in results["per_subject"][arm][s][roi]:
                    d = results["per_subject"][arm][s][roi][band]
                    d.pop("_m_resid", None)
                    d.pop("_c_resid", None)

    results["predictions"] = score_predictions(results, human)
    results["reading_of_record"] = reading(results["predictions"])
    results["files"] = files
    results["settings"] = {
        "fits_dir": fits_dir, "human_dir": human_dir, "repro_tol": repro_tol,
        "method": method, "reliable_voxels_only": reliable, "n_perm": n_perm,
        "v1_streams": v1_streams,
        "ceilings": {k: human[k] for k in ("ceiling_h", "ceiling_v", "ceiling_h_raw", "ceiling_v_raw")},
        "behavioural_rate_model": BEHAV_RATE_MODEL, "min_vox": MIN_VOX, "gate_p": GATE_P,
        "arms": arms, "rois": {k: list(v[1]) for k, v in ROIS.items()},
    }
    with open(os.path.join(out_dir, "ctd_analysis.json"), "w") as fh:
        json.dump(results, fh, indent=1, default=_json_default)
    with open(os.path.join(out_dir, "ctd_summary.md"), "w") as fh:
        fh.write(summary_md(results))
    print(f"[ctd] reading of record: {results['reading_of_record']}", flush=True)
    print(f"[ctd] saved -> {out_dir}", flush=True)
    return results


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


def _subject_values(results, arm, roi, band, key):
    """(subjects, values) for one arm x ROI x band over the subjects that have the cell."""
    per = results["per_subject"].get(arm, {})
    subs = sorted(per, key=int)
    vals = [per[s][roi][band][key] for s in subs if roi in per[s] and band in per[s][roi]]
    subs = [s for s in subs if roi in per[s] and band in per[s][roi]]
    return subs, np.asarray(vals, float)


def score_predictions(results: dict, human: dict) -> dict:
    """P-C1 .. P-C5 of the plan, scored by its rules; nothing else."""
    P: dict = {}
    per = results["per_subject"]
    if "model" not in per:
        return {"error": "no model arm"}

    # P-C1: B < 0.5 in hV4 / ventral / lateral, 8/8 subjects, every seed; B < 0.682 everywhere
    c1 = {"rois": {}}
    met = True
    for roi in EXTRASTRIATE:
        subs, B = _subject_values(results, "model", roi, "joint", "B")
        seed_ok = all(
            d[roi]["joint"]["B"] < 0.5
            for run in results["per_seed"].get("model", {})
            for d in results["per_seed"]["model"][run].values() if roi in d
        )
        c1["rois"][roi] = {"n_subjects": len(subs), "n_below_half": int((B < 0.5).sum()),
                           "all_seeds_below_half": bool(seed_ok), "B": B.tolist()}
        met &= len(subs) == 8 and int((B < 0.5).sum()) == 8 and seed_ok
    below_behav = {roi: _subject_values(results, "model", roi, "joint", "B")[1].tolist() for roi in ROIS}
    all_below = all(all(b < BEHAV_RATE_MODEL for b in v) for v in below_behav.values() if v)
    c1["B_below_behavioural_rate_everywhere"] = bool(all_below)
    c1["met"] = bool(met and all_below)
    P["P-C1"] = c1

    # P-C2: r_h > 0 in 8/8 (exact Wilcoxon), > gabor floor in 8/8, perm p < .01 (shape target decides);
    # P-C2b: the same three criteria on the resolution target, reported
    c2 = {"rois": {}}
    met = {"h": True, "v": True}
    gate_fail = 0
    for roi in EXTRASTRIATE:
        d = {}
        for tgt, key in (("h", "r_h_resid"), ("v", "r_v_resid")):
            subs, r = _subject_values(results, "model", roi, "joint", key)
            gsubs, g = _subject_values(results, "gabor", roi, "joint", key)
            common = [s for s in subs if s in gsubs]
            rr = np.asarray([r[subs.index(s)] for s in common])
            gg = np.asarray([g[gsubs.index(s)] for s in common])
            w = exact_wilcoxon(r)
            wg = exact_wilcoxon(rr - gg) if len(common) else {"p_greater": np.nan}
            pp = results["permutation"].get("model", {}).get(roi, {}).get(f"p_{tgt}", np.nan)
            d[tgt] = {"n_subjects": len(subs), "n_positive": int((r > 0).sum()), "wilcoxon_p_greater": w["p_greater"],
                      "n_above_gabor": int((rr > gg).sum()), "n_common_with_gabor": len(common),
                      "wilcoxon_vs_gabor_p_greater": wg["p_greater"], "perm_p": pp,
                      "pct_ceiling_mean": float(np.mean(_subject_values(results, "model", roi, "joint", f"pct_{tgt}")[1]))}
            met[tgt] &= (len(subs) == 8 and int((r > 0).sum()) == 8 and len(common) == 8
                         and int((rr > gg).sum()) == 8 and (pp < 0.01))
        gates = _subject_values(results, "model", roi, "joint", "gate_pass")[1]
        d["gate_pass_n"] = int(np.sum(gates))
        d["noest"] = bool(len(gates) < 8 or np.sum(gates) < 8)
        gate_fail += int(d["noest"])
        c2["rois"][roi] = d
    c2["n_noest_rois"] = gate_fail
    c2["met"] = bool(met["h"] and gate_fail == 0)
    c2["met_b_resolution"] = bool(met["v"] and gate_fail == 0)
    P["P-C2"] = c2

    # P-C3: B(magno) > B(parvo) in every ROI, 8/8 subjects, 3/3 seeds; joint between
    c3 = {"rois": {}}
    met = True
    for roi in ROIS:
        subs, Bm = _subject_values(results, "model", roi, "magno", "B")
        _, Bp = _subject_values(results, "model", roi, "parvo", "B")
        _, Bj = _subject_values(results, "model", roi, "joint", "B")
        seeds_ok = sum(
            d[roi]["magno"]["B"] > d[roi]["parvo"]["B"]
            for run in results["per_seed"].get("model", {})
            for d in results["per_seed"]["model"][run].values() if roi in d
        )
        seeds_n = sum(1 for run in results["per_seed"].get("model", {})
                      for d in results["per_seed"]["model"][run].values() if roi in d)
        between = int(np.sum((Bj <= np.maximum(Bm, Bp)) & (Bj >= np.minimum(Bm, Bp)))) if len(subs) else 0
        c3["rois"][roi] = {"n_subjects": len(subs), "n_magno_gt_parvo": int((Bm > Bp).sum()),
                           "seed_cells_magno_gt_parvo": f"{seeds_ok}/{seeds_n}", "n_joint_between": between}
        met &= len(subs) == 8 and int((Bm > Bp).sum()) == 8 and seeds_ok == seeds_n and between == 8
    c3["met"] = bool(met)
    P["P-C3"] = c3

    # P-C4: extrastriate mean pct_v, seed means: model > control and > swap, complete separation
    c4 = {}
    def seed_means(arm, key):
        out = {}
        for run, subj in results["per_seed"].get(arm, {}).items():
            vals = [np.mean([subj[s][roi]["joint"][key] for roi in EXTRASTRIATE if roi in subj[s]]) for s in subj]
            out[run] = float(np.mean(vals))
        return out
    for tgt in ("v", "h"):
        mm = seed_means("model", f"pct_{tgt}")
        cc = seed_means("control", f"pct_{tgt}")
        ss = seed_means("swap", f"pct_{tgt}")
        d = {"model_seed_means": mm, "control_seed_means": cc, "swap_seed_means": ss}
        if len(mm) == 3 and len(cc) == 3:
            d["mw_vs_control"] = exact_mannwhitney(list(mm.values()), list(cc.values()))
        if len(mm) == 3 and len(ss) == 3:
            d["mw_vs_swap"] = exact_mannwhitney(list(mm.values()), list(ss.values()))
        d["complete_separation"] = bool(
            len(mm) == 3 and len(cc) == 3 and len(ss) == 3
            and min(mm.values()) > max(cc.values()) and min(mm.values()) > max(ss.values()))
        c4[tgt] = d
    c4["met"] = bool(c4["v"]["complete_separation"])        # directional only on v
    P["P-C4"] = c4

    # P-C5: profile, no prediction
    c5 = {}
    for arm in per:
        c5[arm] = {roi: {
            "pct_h_mean": float(np.mean(_subject_values(results, arm, roi, "joint", "pct_h")[1])) if _subject_values(results, arm, roi, "joint", "pct_h")[1].size else None,
            "pct_v_mean": float(np.mean(_subject_values(results, arm, roi, "joint", "pct_v")[1])) if _subject_values(results, arm, roi, "joint", "pct_v")[1].size else None,
            "B_mean": float(np.mean(_subject_values(results, arm, roi, "joint", "B")[1])) if _subject_values(results, arm, roi, "joint", "B")[1].size else None,
            "n_vox_mean": float(np.mean(_subject_values(results, arm, roi, "joint", "n_vox")[1])) if _subject_values(results, arm, roi, "joint", "n_vox")[1].size else None,
            "r2_median_mean": float(np.mean(_subject_values(results, arm, roi, "joint", "r2_median")[1])) if _subject_values(results, arm, roi, "joint", "r2_median")[1].size else None,
        } for roi in ROIS}
    P["P-C5"] = c5
    return P


def reading(P: dict) -> str:
    if "error" in P:
        return "NOEST"
    if P["P-C2"]["n_noest_rois"] >= 2:
        return "NOEST"
    if not P["P-C1"]["met"]:
        return "CORTEX_NOT_TEXTURE_BIASED_ON_THESE_IMAGES"
    if P["P-C2"]["met"]:
        return "PARADOX_REPRODUCED_IN_PREDICTED_CORTEX"
    return "TEXTURE_BIASED_NO_EVIDENCE"


def summary_md(results: dict) -> str:
    lines = [f"# Cortex-to-decision test — reading of record: `{results['reading_of_record']}`", ""]
    P = results["predictions"]
    for k in ("P-C1", "P-C2", "P-C3", "P-C4"):
        if k in P:
            lines.append(f"- {k}: **{'MET' if P[k].get('met') else 'NOT MET'}**")
    if "P-C2" in P:
        lines.append(f"- P-C2b (resolution target): **{'MET' if P['P-C2'].get('met_b_resolution') else 'NOT MET'}**")
    lines += ["", "## Joint band, seeds averaged per subject (mean over subjects)", "",
              "| arm | ROI | n subj | voxels | median r² | B | % ceiling (shape) | % ceiling (resolution) | perm p (shape / resolution) | gate pass |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for arm in results["per_subject"]:
        for roi in ROIS:
            subs, B = _subject_values(results, arm, roi, "joint", "B")
            if not len(subs):
                continue
            ph = _subject_values(results, arm, roi, "joint", "pct_h")[1]
            pv = _subject_values(results, arm, roi, "joint", "pct_v")[1]
            nv = _subject_values(results, arm, roi, "joint", "n_vox")[1]
            r2 = _subject_values(results, arm, roi, "joint", "r2_median")[1]
            g = _subject_values(results, arm, roi, "joint", "gate_pass")[1]
            pp = results["permutation"].get(arm, {}).get(roi, {})
            lines.append(f"| {arm} | {roi} | {len(subs)} | {nv.mean():.0f} | {r2.mean():.3f} | {B.mean():.3f} "
                         f"| {ph.mean():+.1f} | {pv.mean():+.1f} | {pp.get('p_h', float('nan')):.4f} / {pp.get('p_v', float('nan')):.4f} | {int(g.sum())}/{len(g)} |")
    excl = [f for f in results["files"] if f.get("excluded")]
    lines += ["", f"Excluded files: {len(excl)}"] + [f"- {f['run']} subj{f['subject']:02d}: {f['excluded']}" for f in excl]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fits-dir", required=True)
    p.add_argument("--human-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--repro-tol", type=float, default=1e-3)
    p.add_argument("--method", choices=("pearson", "spearman"), default="pearson",
                   help="'spearman' is the plan's sensitivity analysis 1")
    p.add_argument("--all-voxels", action="store_true",
                   help="sensitivity analysis 3: no reliability mask")
    p.add_argument("--v1-streams", action="store_true",
                   help="sensitivity analysis 2: V1 = streams early (label 1)")
    p.add_argument("--n-perm", type=int, default=N_PERM)
    args = p.parse_args()
    run(args.fits_dir, args.human_dir, args.out_dir, args.repro_tol, args.method,
        not args.all_voxels, args.n_perm, v1_streams=args.v1_streams)


if __name__ == "__main__":
    main()
