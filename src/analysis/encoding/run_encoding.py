"""Per-subject voxelwise encoding analysis (Slurm-array entry point).

Pipeline: NSD design -> DKL front end -> stream features (+PCA on train
images) -> banded ridge + single-band fits -> variance partitioning ->
save npz.

Examples
--------
Untrained-control network (no checkpoint needed):
    python -m src.analysis.encoding.run_encoding --subject 1 \
        --model-name untrained_seed0 --out-dir results/encoding

Trained checkpoint (e.g. DKL B&CS):
    python -m src.analysis.encoding.run_encoding --subject 1 \
        --checkpoint /path/to/ckpt --model-name dkl_bcs \
        --out-dir results/encoding
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from src.analysis.encoding import banded_ridge, dkl_frontend, features
from src.analysis.encoding import nsd_data


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--subject", type=int, required=True, help="1..8")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--model-name", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--stream-concat", action="store_true")
    p.add_argument("--stream1-weight", type=float, default=0.5)
    p.add_argument("--no-lum-blur", action="store_true")
    p.add_argument("--no-lum-cs", action="store_true")
    p.add_argument("--age-months", type=int, default=48)
    p.add_argument("--pca-dim", type=int, default=1000)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument(
        "--backend",
        type=str,
        default="torch_cuda" if torch.cuda.is_available() else "torch",
    )
    p.add_argument(
        "--feature-cache",
        type=str,
        default=None,
        help="npz path to cache per-image stream features",
    )
    return p.parse_args()


def get_features_for_images(args, image_ids: np.ndarray, device: str):
    """Stream features for a set of images, with optional on-disk cache."""
    if args.feature_cache and os.path.exists(args.feature_cache):
        z = np.load(args.feature_cache)
        cached_ids = z["image_ids"]
        pos = {int(i): k for k, i in enumerate(cached_ids)}
        idx = np.array([pos[int(i)] for i in image_ids])
        return z["magno"][idx], z["parvo"][idx]

    ch_min, ch_max = dkl_frontend.compute_dkl_channel_minmax()
    model = features.build_model(
        checkpoint=args.checkpoint,
        stream_concat=args.stream_concat,
        stream1_weight=args.stream1_weight,
        seed=args.seed,
        device=device,
    )

    magno_all, parvo_all = [], []
    chunk = 512
    for start in range(0, len(image_ids), chunk):
        ids = image_ids[start : start + chunk]
        rgb = dkl_frontend.load_nsd_images(ids)
        x_m, x_p = dkl_frontend.magno_parvo_inputs(
            rgb,
            ch_min,
            ch_max,
            lum_blur=not args.no_lum_blur,
            lum_cs=not args.no_lum_cs,
            age_months=args.age_months,
            device=device,
        )
        f = features.extract_stream_features(
            model, x_m, x_p, batch_size=args.batch_size, device=device
        )
        magno_all.append(f["magno"])
        parvo_all.append(f["parvo"])
        print(f"  features {start + len(ids)}/{len(image_ids)}", flush=True)

    magno = np.concatenate(magno_all)
    parvo = np.concatenate(parvo_all)
    if args.feature_cache:
        os.makedirs(os.path.dirname(args.feature_cache), exist_ok=True)
        np.savez(
            args.feature_cache, image_ids=image_ids, magno=magno, parvo=parvo
        )
    return magno, parvo


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    s = args.subject
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(
        args.out_dir, f"{args.model_name}_subj{s:02d}.npz"
    )

    print(f"[subj{s:02d}] design + betas", flush=True)
    stim_info = nsd_data.load_stim_info()
    split = nsd_data.build_train_test_split(s, stim_info)
    mask = nsd_data.load_mask(s, "nsdgeneral")
    betas = nsd_data.load_betas(s, mask)
    nc = nsd_data.load_noise_ceiling(s, mask)

    Y_train = betas[split["train_trials"]]
    Y_test = betas[split["test_trials_per_image"]].mean(axis=1)
    del betas

    print(f"[subj{s:02d}] features", flush=True)
    all_ids = np.unique(
        np.concatenate([split["train_image_ids"], split["test_image_ids"]])
    )
    magno, parvo = get_features_for_images(args, all_ids, device)
    row_of = {int(i): k for k, i in enumerate(all_ids)}
    tr_rows = np.array([row_of[int(i)] for i in split["train_image_ids"]])
    te_rows = np.array([row_of[int(i)] for i in split["test_image_ids"]])

    print(f"[subj{s:02d}] PCA to {args.pca_dim} per stream", flush=True)
    pcam = features.fit_pca(magno[np.unique(tr_rows)], args.pca_dim, args.seed)
    pcap = features.fit_pca(parvo[np.unique(tr_rows)], args.pca_dim, args.seed)
    Xm_tr, Xm_te = pcam.transform(magno[tr_rows]), pcam.transform(magno[te_rows])
    Xp_tr, Xp_te = pcap.transform(parvo[tr_rows]), pcap.transform(parvo[te_rows])
    del magno, parvo

    print(f"[subj{s:02d}] banded ridge", flush=True)
    splits = banded_ridge.make_grouped_splits(
        split["train_image_ids"], n_splits=args.n_splits, seed=args.seed
    )
    res = banded_ridge.fit_banded(
        [Xm_tr, Xp_tr], Y_train, [Xm_te, Xp_te], Y_test,
        splits, backend=args.backend, seed=args.seed,
    )
    part = banded_ridge.variance_partition(
        res["r2_joint"], res["r2_magno"], res["r2_parvo"], noise_ceiling=nc
    )

    rois = {}
    for roi_name in ("streams", "prf-eccrois", "prf-visualrois"):
        try:
            rois[f"roi_{roi_name}"] = nsd_data.load_roi_labels(
                s, roi_name, mask
            )
        except (FileNotFoundError, ValueError) as e:
            print(f"  skip ROI {roi_name}: {e}")

    np.savez(
        out_path,
        subject=s,
        model_name=args.model_name,
        noise_ceiling=nc,
        **res,
        **{k: v for k, v in part.items()},
        **rois,
    )
    print(f"[subj{s:02d}] saved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
