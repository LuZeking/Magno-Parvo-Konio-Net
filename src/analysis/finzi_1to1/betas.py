"""Cache image-averaged betas for the NSD `streams` high-level ROIs.

Voxels: `lh.streams` / `rh.streams` labels 5 (ventral), 6 (lateral), 7 (parietal
= dorsal), noise ceiling of the 3-repetition mean >= 10 %, no `nsdgeneral`
restriction (plan §2). Responses: `nsd_data.load_betas` (PSC, z-scored per voxel
within session), averaged over each image's repetitions. Train = every image the
subject saw except the shared-515; test = the shared-515.

    python -m src.analysis.finzi_1to1.betas --subject 1
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from src.analysis.encoding import nsd_data
from src.analysis.finzi_1to1 import NC_MIN_PERCENT, STREAM_LABELS, betas_cache_path


def stream_voxels(subject: int) -> dict[str, np.ndarray]:
    """Boolean (z, y, x) mask plus per-voxel hemisphere, label and NC (%)."""
    roi = nsd_data._roi_dir(subject)
    hemi = None
    label = None
    for h, name in enumerate(("lh.streams", "rh.streams")):
        vol = nsd_data._load_volume_zyx(os.path.join(roi, f"{name}.nii.gz"))
        vol = np.rint(vol).astype(np.int32)
        if hemi is None:
            hemi = np.full(vol.shape, -1, dtype=np.int8)
            label = np.zeros(vol.shape, dtype=np.int32)
        sel = np.isin(vol, list(STREAM_LABELS))
        if (sel & (hemi >= 0)).any():
            raise RuntimeError(f"subject {subject}: lh/rh streams voxels overlap")
        hemi[sel] = h
        label[sel] = vol[sel]
    ncsnr = nsd_data._load_volume_zyx(
        os.path.join(nsd_data._betas_dir(subject), "ncsnr.nii.gz")
    ).astype(np.float64)
    nc = 100.0 * ncsnr**2 / (ncsnr**2 + 1.0 / nsd_data.N_REPS)
    mask = (hemi >= 0) & (nc >= NC_MIN_PERCENT)
    return {
        "mask": mask,
        "hemi": hemi[mask],
        "label": label[mask],
        "nc": nc[mask],
        "zyx": np.argwhere(mask).astype(np.int16),  # C order == load_betas order
    }


def image_means(betas: np.ndarray, trial_rows: np.ndarray,
                trial_image_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean response per image over the given trials -> (ids, means, n_reps)."""
    ids = trial_image_ids[trial_rows]
    uniq, inv = np.unique(ids, return_inverse=True)
    order = trial_rows[np.argsort(inv, kind="stable")]
    counts = np.bincount(inv)
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    sums = np.add.reduceat(betas[order].astype(np.float64), starts, axis=0)
    return uniq, (sums / counts[:, None]).astype(np.float32), counts


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--subject", type=int, required=True)
    args = p.parse_args()
    s = args.subject
    t0 = time.time()

    vox = stream_voxels(s)
    for h in (0, 1):
        for lab, name in STREAM_LABELS.items():
            n = int(((vox["hemi"] == h) & (vox["label"] == lab)).sum())
            print(f"[betas] subj{s:02d} {'lr'[h]}h {name}: {n} voxels", flush=True)

    stim_info = nsd_data.load_stim_info()
    split = nsd_data.build_train_test_split(s, stim_info)
    betas = nsd_data.load_betas(s, vox["mask"])
    print(f"[betas] loaded {betas.shape} in {time.time() - t0:.0f}s", flush=True)

    trial_ids = split["trial_image_ids"]
    train_ids, train_b, train_reps = image_means(betas, split["train_trials"], trial_ids)
    test_rows = split["test_trials_per_image"]  # (515, 3)
    test_b = betas[test_rows].astype(np.float64).mean(axis=1).astype(np.float32)
    del betas

    out = betas_cache_path(s)
    tmp = out + ".tmp.npz"
    np.savez(
        tmp,
        train_image_ids=train_ids, train_betas=train_b, train_n_reps=train_reps,
        test_image_ids=split["test_image_ids"], test_betas=test_b,
        hemi=vox["hemi"], label=vox["label"], nc=vox["nc"], zyx=vox["zyx"],
    )
    os.replace(tmp, out)
    meta = {
        "subject": s, "n_voxels": int(vox["mask"].sum()),
        "n_train_images": int(len(train_ids)), "n_test_images": int(len(test_b)),
        "train_reps_hist": np.bincount(train_reps).tolist(),
        "nc_min_percent": NC_MIN_PERCENT, "seconds": round(time.time() - t0, 1),
    }
    with open(out.replace(".npz", ".json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[betas] wrote {out} {json.dumps(meta)}", flush=True)


if __name__ == "__main__":
    main()
