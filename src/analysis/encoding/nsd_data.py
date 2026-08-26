"""NSD data access for voxelwise encoding analyses.

Local NSD layout (verified 2026-08-25):
  betas:   {NSD_ROOT}/betas/subjXX/betas_fithrf_GLMdenoise_RR/betas_sessionNN.hdf5
           (dataset "betas", int16, scale 1/300 -> percent signal change,
            axis order (trial, z, y, x); the paired .nii.gz files are (x, y, z, trial))
  ncsnr:   same dir, ncsnr.nii.gz (+ ncsnr_split1/2)
  ROIs:    {NSD_ROOT}/rois/subjXX/func1pt8mm/roi/{name}.nii.gz  (x, y, z)
  stimuli: {NSD_ROOT}/stimuli/nsd_stimuli.hdf5 ("imgBrick", 73000 x 425 x 425 x 3 uint8)
           {NSD_ROOT}/stimuli/nsd_stim_info_merged.pkl

Conventions used throughout:
  * "image id" = 0-based row index into nsd_stim_info_merged / imgBrick.
    (The stim-info column ``nsdId`` is that same 0-based index.)
  * Trial indices in subjectX_repN columns are 1-based positions in the
    subject's full 30,000-trial design; only trials within the sessions
    present on disk are usable.
"""

from __future__ import annotations

import glob
import os
import pickle
import re

import h5py
import nibabel as nib
import numpy as np
import pandas as pd

NSD_ROOT = os.environ.get(
    "NSD_ROOT", "/scratch/byrong/encoding/datasets/nsd"
)
BETA_VERSION = "betas_fithrf_GLMdenoise_RR"
TRIALS_PER_SESSION = 750
N_REPS = 3
N_SUBJECTS = 8


def _betas_dir(subject: int) -> str:
    return os.path.join(NSD_ROOT, "betas", f"subj{subject:02d}", BETA_VERSION)


def _roi_dir(subject: int) -> str:
    return os.path.join(
        NSD_ROOT, "rois", f"subj{subject:02d}", "func1pt8mm", "roi"
    )


def load_stim_info() -> pd.DataFrame:
    """Load the NSD design table.

    `nsd_stim_info_merged.pkl` was written by an older Python, so newer
    pandas/pickle versions raise UnicodeDecodeError unless the legacy
    latin1 encoding is requested explicitly.
    """
    path = os.path.join(NSD_ROOT, "stimuli", "nsd_stim_info_merged.pkl")
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except UnicodeDecodeError:
            f.seek(0)
            return pickle.load(f, encoding="latin1")


def available_sessions(subject: int) -> list[int]:
    files = glob.glob(os.path.join(_betas_dir(subject), "betas_session*.hdf5"))
    sessions = sorted(
        int(re.search(r"session(\d+)", os.path.basename(f)).group(1))
        for f in files
    )
    return sessions


def n_completed_trials(subject: int) -> int:
    return len(available_sessions(subject)) * TRIALS_PER_SESSION


def subject_trial_image_ids(
    stim_info: pd.DataFrame, subject: int
) -> np.ndarray:
    """Return the 0-based image id shown on each of the subject's trials.

    Output has length ``n_completed_trials(subject)``; entry t is the image
    id of (1-based) trial t+1.
    """
    n_trials = n_completed_trials(subject)
    trial_ids = np.full(30000, -1, dtype=np.int64)
    seen = stim_info[stim_info[f"subject{subject}"] == 1]
    for rep in range(N_REPS):
        col = f"subject{subject}_rep{rep}"
        trial_pos = seen[col].to_numpy(dtype=np.int64)  # 1-based
        trial_ids[trial_pos - 1] = seen.index.to_numpy(dtype=np.int64)
    trial_ids = trial_ids[:n_trials]
    if (trial_ids < 0).any():
        raise RuntimeError(
            f"subject {subject}: {np.sum(trial_ids < 0)} trials without an "
            "image id — design table / session count mismatch"
        )
    return trial_ids


def shared515_image_ids(stim_info: pd.DataFrame) -> np.ndarray:
    """Image ids of shared-1000 stimuli seen 3x by ALL subjects.

    An image counts for a subject when all three rep trials fall within the
    sessions available on disk. With full 40-session data this yields the
    canonical "special515" set.
    """
    # .to_numpy() can hand back a read-only view of the frame's block,
    # which breaks the in-place &= below.
    mask = np.array(stim_info["shared1000"].to_numpy(dtype=bool), copy=True)
    for subject in range(1, N_SUBJECTS + 1):
        n_trials = n_completed_trials(subject)
        for rep in range(N_REPS):
            col = f"subject{subject}_rep{rep}"
            trial_pos = stim_info[col].to_numpy()
            mask &= (trial_pos >= 1) & (trial_pos <= n_trials)
    ids = np.flatnonzero(mask)
    return ids


def _load_volume_zyx(path: str) -> np.ndarray:
    """Load a (x, y, z) NIfTI and return it as (z, y, x) to match the
    HDF5 beta axis order."""
    vol = np.asarray(nib.load(path).dataobj)
    return np.transpose(vol, (2, 1, 0))


def load_mask(subject: int, name: str = "nsdgeneral") -> np.ndarray:
    """Boolean (z, y, x) mask. For label volumes, mask = label > 0."""
    vol = _load_volume_zyx(os.path.join(_roi_dir(subject), f"{name}.nii.gz"))
    return vol > 0


def load_roi_labels(subject: int, name: str, mask: np.ndarray) -> np.ndarray:
    """Integer ROI labels within ``mask`` (flat, len = mask.sum())."""
    vol = _load_volume_zyx(os.path.join(_roi_dir(subject), f"{name}.nii.gz"))
    if vol.shape != mask.shape:
        raise ValueError(
            f"ROI {name} shape {vol.shape} != mask shape {mask.shape}"
        )
    return vol[mask].astype(np.int32)


def load_noise_ceiling(
    subject: int, mask: np.ndarray, n_reps_averaged: int = N_REPS
) -> np.ndarray:
    """Fraction of explainable variance for responses averaged over n reps:
    NC = ncsnr^2 / (ncsnr^2 + 1/n)."""
    ncsnr = _load_volume_zyx(
        os.path.join(_betas_dir(subject), "ncsnr.nii.gz")
    )[mask].astype(np.float64)
    return ncsnr**2 / (ncsnr**2 + 1.0 / n_reps_averaged)


def load_betas(
    subject: int,
    mask: np.ndarray,
    sessions: list[int] | None = None,
    dtype=np.float32,
) -> np.ndarray:
    """Load, scale to PSC, and z-score betas per voxel within session.

    Returns (n_trials, n_voxels) with trials in chronological order.
    """
    if sessions is None:
        sessions = available_sessions(subject)
    flat_mask = mask.reshape(-1)
    out = []
    for ses in sessions:
        path = os.path.join(
            _betas_dir(subject), f"betas_session{ses:02d}.hdf5"
        )
        with h5py.File(path, "r") as f:
            raw = f["betas"][:]  # (750, z, y, x) int16
        if raw.shape[1:] != mask.shape:
            raise ValueError(
                f"betas {raw.shape[1:]} vs mask {mask.shape} mismatch "
                f"(subject {subject}, session {ses})"
            )
        b = raw.reshape(raw.shape[0], -1)[:, flat_mask].astype(np.float64)
        b /= 300.0  # int16 -> percent signal change
        mu = b.mean(axis=0, keepdims=True)
        sd = b.std(axis=0, keepdims=True)
        sd[sd == 0] = 1.0
        out.append(((b - mu) / sd).astype(dtype))
    return np.concatenate(out, axis=0)


def build_train_test_split(
    subject: int, stim_info: pd.DataFrame
) -> dict[str, np.ndarray]:
    """Train/test split by image id.

    test:  shared-515 images; responses = mean over the subject's 3 reps.
    train: every trial of every other image the subject saw (single trial).
    """
    trial_ids = subject_trial_image_ids(stim_info, subject)
    test_ids = shared515_image_ids(stim_info)
    test_set = set(test_ids.tolist())

    is_test_trial = np.isin(trial_ids, test_ids)
    train_trials = np.flatnonzero(~is_test_trial)

    test_trials_per_image = np.stack(
        [np.flatnonzero(trial_ids == i) for i in test_ids]
    )  # (515, 3)
    if test_trials_per_image.shape[1] != N_REPS:
        raise RuntimeError("expected exactly 3 reps per shared-515 image")

    return {
        "trial_image_ids": trial_ids,
        "train_trials": train_trials,
        "train_image_ids": trial_ids[train_trials],
        "test_image_ids": test_ids,
        "test_trials_per_image": test_trials_per_image,
    }
