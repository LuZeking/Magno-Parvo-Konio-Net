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


def fit_banded(
    X_bands_train: list[np.ndarray],
    Y_train: np.ndarray,
    X_bands_test: list[np.ndarray],
    Y_test: np.ndarray,
    splits,
    backend: str = "torch_cuda",
    n_iter: int = 100,
    alphas=np.logspace(-2, 8, 21),
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Joint banded ridge + per-band single ridge fits; test R2 per voxel."""
    from himalaya.backend import set_backend
    from himalaya.ridge import BandedRidgeCV, RidgeCV
    from sklearn.preprocessing import StandardScaler

    backend_mod = set_backend(backend, on_error="warn")

    # per-band standardization, fit on train only
    scalers = [StandardScaler().fit(Xb) for Xb in X_bands_train]
    Xtr = [s.transform(Xb) for s, Xb in zip(scalers, X_bands_train)]
    Xte = [s.transform(Xb) for s, Xb in zip(scalers, X_bands_test)]

    results: dict[str, np.ndarray] = {}

    # ---- single-band ridge fits ----
    for name, Xb_tr, Xb_te in zip(("magno", "parvo"), Xtr, Xte):
        ridge = RidgeCV(alphas=alphas, cv=splits)
        ridge.fit(Xb_tr.astype(np.float32), Y_train)
        pred = ridge.predict(Xb_te.astype(np.float32))
        results[f"r2_{name}"] = r2_per_target(
            Y_test, _as_numpy(pred, backend_mod)
        )

    # ---- joint banded ridge, PRIMAL space ----
    # With ~1k features per band and tens of thousands of trials, the primal
    # solver is far cheaper than a kernel one (which would build an
    # n_samples x n_samples kernel per band). Y_in_cpu keeps the voxel matrix
    # off the GPU so the fit is not limited by card memory.
    groups = np.concatenate(
        [np.full(Xb.shape[1], i) for i, Xb in enumerate(Xtr)]
    )
    Xtr_cat = np.concatenate(Xtr, axis=1).astype(np.float32)
    Xte_cat = np.concatenate(Xte, axis=1).astype(np.float32)

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

    out = {
        "unique_magno_raw": unique_m,
        "unique_parvo_raw": unique_p,
        "shared_raw": shared,
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
