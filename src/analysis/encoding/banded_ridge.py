"""Banded ridge regression + stream-wise variance partitioning (himalaya).

Fits, per subject:
  * joint 2-band banded ridge (magno features, parvo features)
  * single-band ridge per stream (identical CV splits)
and computes on the held-out shared-515 (rep-averaged) responses:
  unique_magno = R2_joint - R2_parvo
  unique_parvo = R2_joint - R2_magno
  shared       = R2_magno + R2_parvo - R2_joint
  SPI          = (U_m - U_p) / (U_m + U_p) for reliably predicted voxels
All R2 values can be normalized by the ncsnr-derived noise ceiling.
"""

from __future__ import annotations

import numpy as np


def make_grouped_splits(
    image_ids: np.ndarray, n_splits: int = 5, seed: int = 0
) -> list[tuple[np.ndarray, np.ndarray]]:
    """K-fold splits over TRIALS, grouped by image id (no image straddles
    train/val)."""
    rng = np.random.default_rng(seed)
    unique_ids = np.unique(image_ids)
    perm = rng.permutation(len(unique_ids))
    fold_of_image = np.empty(len(unique_ids), dtype=np.int64)
    fold_of_image[perm] = np.arange(len(unique_ids)) % n_splits
    id_to_fold = dict(zip(unique_ids.tolist(), fold_of_image.tolist()))
    trial_folds = np.array([id_to_fold[i] for i in image_ids])
    return [
        (np.flatnonzero(trial_folds != k), np.flatnonzero(trial_folds == k))
        for k in range(n_splits)
    ]


def _as_numpy(x, backend_mod) -> np.ndarray:
    """Predictions come back as numpy when Y_in_cpu=True and as backend
    tensors otherwise; normalize both."""
    if isinstance(x, np.ndarray):
        return x
    try:
        return np.asarray(backend_mod.to_numpy(x))
    except (AttributeError, TypeError):
        return np.asarray(x)


def r2_per_target(Y_true: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
    """Per-voxel R^2, computed in numpy.

    himalaya's own scoring assumes backend tensors and raises on the numpy
    arrays that Y_in_cpu=True yields, so the metric is computed here.
    Matches himalaya's definition: 1 - SS_res / SS_tot with SS_tot about the
    test-set mean of each target.
    """
    Y_true = np.asarray(Y_true, dtype=np.float64)
    Y_pred = np.asarray(Y_pred, dtype=np.float64)
    ss_res = np.sum((Y_true - Y_pred) ** 2, axis=0)
    ss_tot = np.sum((Y_true - Y_true.mean(axis=0, keepdims=True)) ** 2, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, np.nan)


def prepare_band(
    Xb_train: np.ndarray,
    Xb_test: np.ndarray,
    Xb_extra: np.ndarray | None = None,
    var_tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Center, drop degenerate dimensions, and scale one feature band.

    Do NOT z-score PCA output: components whose variance is ~0 (the feature
    matrix is rank-deficient well below the requested component count) get
    rescaled to unit variance, which amplifies numerical noise and leaves the
    design ill-conditioned -- torch's SVD then fails to converge outright.
    Instead drop those dimensions and give the band unit root-mean-square
    scale, which also puts the two bands on comparable footing for the
    banded-ridge group weights.

    All statistics come from the training split only. ``Xb_extra`` is an
    optional further design (e.g. the cue-conflict stimuli of PAPER_PLAN
    §7.3b) that must receive the *same* centring, dimension drop and scale as
    the test set -- refitting them on 1280 out-of-distribution images would
    make its predictions incomparable with the NSD ones.
    """
    mu = Xb_train.mean(axis=0, keepdims=True)
    tr = (Xb_train - mu).astype(np.float64)
    te = (Xb_test - mu).astype(np.float64)
    ex = None if Xb_extra is None else (Xb_extra - mu).astype(np.float64)

    sd = tr.std(axis=0)
    keep = sd > (sd.max() * var_tol if sd.size and sd.max() > 0 else 0.0)
    if not keep.any():
        raise ValueError("feature band has no non-degenerate dimensions")
    tr, te = tr[:, keep], te[:, keep]
    if ex is not None:
        ex = ex[:, keep]
    if int(keep.size - keep.sum()):
        print(
            f"[banded_ridge] dropped {int(keep.size - keep.sum())} of "
            f"{keep.size} degenerate band dimensions",
            flush=True,
        )

    scale = np.sqrt((tr**2).sum() / tr.shape[0])
    if scale <= 0:
        raise ValueError("feature band has zero scale")
    return tr / scale, te / scale, (None if ex is None else ex / scale)


def fit_banded(
    X_bands_train: list[np.ndarray],
    Y_train: np.ndarray,
    X_bands_test: list[np.ndarray],
    Y_test: np.ndarray,
    splits,
    backend: str = "torch_cuda",
    n_iter: int = 20,
    alphas=np.logspace(-2, 8, 21),
    seed: int = 0,
    X_bands_extra: list[np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Joint banded ridge + per-band single ridge fits; test R2 per voxel.

    ``X_bands_extra`` is an optional design of further stimuli, in the same
    band order. When given, the *fitted* magno-only, parvo-only and joint
    estimators are applied to it and the predictions returned under
    ``pred_extra_{magno,parvo,joint}``, each ``(n_extra, n_voxels)``. This is
    what PAPER_PLAN §7.3b needs: the estimators are otherwise discarded here,
    so the cue-conflict images cannot be pushed through an existing fit and
    the encoding run has to produce the predictions itself.

    Cost note: the random search re-decomposes the band-weighted design for
    every sampled gamma, so wall time scales as n_iter * n_folds * SVD cost.
    With only two bands the search space is effectively the 1-D ratio between
    them, so n_iter=20 is ample; n_iter=100 (with 5 folds and voxel chunking)
    ran past a 12-hour limit for a single subject.
    """
    from himalaya.backend import set_backend
    from himalaya.ridge import BandedRidgeCV, RidgeCV

    backend_mod = set_backend(backend, on_error="warn")

    extra_in = (
        [None] * len(X_bands_train) if X_bands_extra is None else X_bands_extra
    )
    prepared = [
        prepare_band(tr, te, ex)
        for tr, te, ex in zip(X_bands_train, X_bands_test, extra_in)
    ]
    Xtr = [p[0] for p in prepared]
    Xte = [p[1] for p in prepared]
    Xex = [p[2] for p in prepared]

    results: dict[str, np.ndarray] = {}

    # ---- single-band ridge fits ----
    for name, Xb_tr, Xb_te, Xb_ex in zip(("magno", "parvo"), Xtr, Xte, Xex):
        ridge = RidgeCV(alphas=alphas, cv=splits)
        ridge.fit(Xb_tr.astype(np.float64), Y_train)
        pred = ridge.predict(Xb_te.astype(np.float64))
        results[f"r2_{name}"] = r2_per_target(
            Y_test, _as_numpy(pred, backend_mod)
        )
        if Xb_ex is not None:
            results[f"pred_extra_{name}"] = _as_numpy(
                ridge.predict(Xb_ex.astype(np.float64)), backend_mod
            ).astype(np.float32)

    # ---- joint banded ridge, PRIMAL space ----
    # With ~1k features per band and tens of thousands of trials, the primal
    # solver is far cheaper than a kernel one (which would build an
    # n_samples x n_samples kernel per band).
    groups = np.concatenate(
        [np.full(Xb.shape[1], i) for i, Xb in enumerate(Xtr)]
    )
    Xtr_cat = np.concatenate(Xtr, axis=1).astype(np.float64)
    Xte_cat = np.concatenate(Xte, axis=1).astype(np.float64)

    # NB: Y_in_cpu=True expects a backend (torch) tensor and raises on the
    # numpy arrays used here, so targets stay in the backend and memory is
    # bounded by n_targets_batch instead.
    banded = BandedRidgeCV(
        groups=groups,
        solver="random_search",
        solver_params=dict(
            n_iter=n_iter,
            alphas=alphas,
            n_targets_batch=2000,
            n_targets_batch_refit=500,
        ),
        random_state=seed,
        cv=splits,
    )
    banded.fit(Xtr_cat, Y_train)
    pred = banded.predict(Xte_cat)
    results["r2_joint"] = r2_per_target(Y_test, _as_numpy(pred, backend_mod))

    if X_bands_extra is not None:
        Xex_cat = np.concatenate(Xex, axis=1).astype(np.float64)
        results["pred_extra_joint"] = _as_numpy(
            banded.predict(Xex_cat), backend_mod
        ).astype(np.float32)

    return results


def variance_partition(
    r2_joint: np.ndarray,
    r2_magno: np.ndarray,
    r2_parvo: np.ndarray,
    noise_ceiling: np.ndarray | None = None,
    min_joint_r2: float = 0.0,
) -> dict[str, np.ndarray]:
    """Unique/shared variance per voxel, optionally noise-ceiling-normalized.

    Caveat (report in Methods): with correlated bands, unique components
    understate a band's total contribution; negative shared variance can
    signal suppression — do not clip silently, keep the raw values too.
    """
    unique_m = r2_joint - r2_parvo
    unique_p = r2_joint - r2_magno
    shared = r2_magno + r2_parvo - r2_joint

    # NB: these were named *_raw until 2026-09-07. That collided head-on with
    # the manuscript's "raw", which means the DIFFERENCE statistic (u_m - u_p)
    # and is itself divided by the noise ceiling -- the opposite sense. A
    # reader following `_raw` from code to paper landed on the wrong quantity
    # (PAPER_PLAN §86.3). `_pre_nc` says what these actually are.
    # Files written before the rename carry the old keys; readers of stored
    # .npz must fall back (see tmp_analysis/group_check.py).
    out = {
        "unique_magno_pre_nc": unique_m,
        "unique_parvo_pre_nc": unique_p,
        "shared_pre_nc": shared,
    }
    if noise_ceiling is not None:
        nc = np.where(noise_ceiling > 0, noise_ceiling, np.nan)
        out.update(
            unique_magno=unique_m / nc,
            unique_parvo=unique_p / nc,
            shared=shared / nc,
            r2_joint_nc=r2_joint / nc,
        )

    reliable = r2_joint > min_joint_r2
    denom = unique_m + unique_p
    spi = np.full(r2_joint.shape, np.nan)
    ok = reliable & (np.abs(denom) > 1e-12)
    spi[ok] = (unique_m[ok] - unique_p[ok]) / denom[ok]
    out["spi"] = spi
    out["reliable_mask"] = reliable
    return out
