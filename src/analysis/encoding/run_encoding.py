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
import time

import numpy as np
import torch

from src.analysis.encoding import (banded_ridge, dkl_frontend, features,
                                    input_bands, voxel_shape_bias)
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
    p.add_argument(
        "--features", choices=("network", "gabor"), default="network",
        help="'gabor' replaces the stream network features with oriented "
             "energy on each stream's own transformed input -- the "
             "input-level null model of PAPER_PLAN 7.2 item 1",
    )
    p.add_argument("--n-splits", type=int, default=3,
                   help="CV folds for alpha/gamma selection; cost is linear "
                        "in this")
    p.add_argument("--voxel-chunk", type=int, default=0,
                   help="fit this many voxels at a time; 0 = all at once. "
                        "himalaya already bounds memory via n_targets_batch, "
                        "and each chunk repeats the per-gamma decomposition, "
                        "so chunking multiplies wall time by the chunk count.")
    p.add_argument(
        "--cue-conflict", action="store_true",
        help="also push the Geirhos cue-conflict set through the fitted "
             "encoding model and store a per-voxel shape-bias index -- the "
             "in-silico cue-conflict panel of PAPER_PLAN 7.3b/21",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--allow-partial", action="store_true",
                   help="encode a still-training checkpoint on purpose")
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


def build_extractor(args, device: str):
    """The front-end constants and (for network features) the model."""
    ch_min, ch_max = dkl_frontend.compute_dkl_channel_minmax()
    # A 12-hour fit on a mid-training checkpoint is expensive to discover
    # late, and the output npz records nothing about it (PAPER_PLAN §10.1).
    if args.checkpoint:
        features.assert_converged(args.checkpoint, args.allow_partial)

    model = None
    if args.features == "network":
        model = features.build_model(
            checkpoint=args.checkpoint,
            stream_concat=args.stream_concat,
            stream1_weight=args.stream1_weight,
            seed=args.seed,
            device=device,
        )
    return ch_min, ch_max, model


def features_from_rgb(args, rgb_uint8, ch_min, ch_max, model, device: str):
    """RGB uint8 (N, H, W, 3) -> (magno, parvo) feature matrices.

    Shared by the NSD stimuli and the cue-conflict set so both go through the
    identical front end and the identical feature stage; a cue-conflict panel
    computed with any other transform would not be comparable with the fit it
    is projected through.
    """
    magno_all, parvo_all = [], []
    chunk = 512
    n = len(rgb_uint8)
    for start in range(0, n, chunk):
        rgb = rgb_uint8[start : start + chunk]
        x_m, x_p = dkl_frontend.magno_parvo_inputs(
            rgb,
            ch_min,
            ch_max,
            lum_blur=not args.no_lum_blur,
            lum_cs=not args.no_lum_cs,
            age_months=args.age_months,
            device=device,
        )
        if args.features == "gabor":
            f = input_bands.extract_input_features(
                x_m, x_p, batch_size=args.batch_size, device=device
            )
        else:
            f = features.extract_stream_features(
                model, x_m, x_p, batch_size=args.batch_size, device=device
            )
        magno_all.append(f["magno"])
        parvo_all.append(f["parvo"])
        print(f"  features {min(start + chunk, n)}/{n}", flush=True)

    return np.concatenate(magno_all), np.concatenate(parvo_all)


def cache_provenance(args) -> dict[str, np.ndarray]:
    """The settings a cached feature matrix is only valid for.

    The cache path is keyed on model name and subject alone. Nothing in it
    records the *front end* the features were computed through, so a run whose
    `--no-lum-blur/--no-lum-cs` flags disagree with the cached file silently
    reuses features from the wrong front end. That is not hypothetical: it is
    what happened to `dkl_base_seed0_trained` subject 1 (PAPER_PLAN §22).
    """
    return {
        "prov_lum_blur": np.asarray(not args.no_lum_blur),
        "prov_lum_cs": np.asarray(not args.no_lum_cs),
        "prov_age_months": np.asarray(args.age_months),
        "prov_features": np.asarray(args.features),
        "prov_checkpoint": np.asarray(args.checkpoint or ""),
        "prov_stream1_weight": np.asarray(args.stream1_weight),
        "prov_seed": np.asarray(args.seed),
    }


def cache_is_valid(z, want: dict[str, np.ndarray]) -> bool:
    """True only if the cache records provenance and it matches."""
    missing = [k for k in want if k not in z.files]
    if missing:
        print(
            "[cache] no provenance recorded in this cache file "
            f"(missing {len(missing)} keys) -- recomputing features rather "
            "than trusting it",
            flush=True,
        )
        return False
    for k, v in want.items():
        if z[k].item() != v.item():
            print(
                f"[cache] MISMATCH {k}: cached={z[k].item()!r} "
                f"requested={v.item()!r} -- recomputing features",
                flush=True,
            )
            return False
    return True


def get_features_for_images(args, image_ids: np.ndarray, device: str):
    """Stream features for a set of NSD images, with optional on-disk cache."""
    want = cache_provenance(args)
    if args.feature_cache and os.path.exists(args.feature_cache):
        z = np.load(args.feature_cache, allow_pickle=False)
        if cache_is_valid(z, want):
            cached_ids = z["image_ids"]
            pos = {int(i): k for k, i in enumerate(cached_ids)}
            idx = np.array([pos[int(i)] for i in image_ids])
            print("[cache] hit, provenance verified", flush=True)
            return z["magno"][idx], z["parvo"][idx]

    ch_min, ch_max, model = build_extractor(args, device)

    magno_all, parvo_all = [], []
    chunk = 512
    for start in range(0, len(image_ids), chunk):
        ids = image_ids[start : start + chunk]
        rgb = dkl_frontend.load_nsd_images(ids)
        m, pv = features_from_rgb(args, rgb, ch_min, ch_max, model, device)
        magno_all.append(m)
        parvo_all.append(pv)

    magno = np.concatenate(magno_all)
    parvo = np.concatenate(parvo_all)
    if args.feature_cache:
        os.makedirs(os.path.dirname(args.feature_cache), exist_ok=True)
        np.savez(
            args.feature_cache,
            image_ids=image_ids,
            magno=magno,
            parvo=parvo,
            **want,
        )
    return magno, parvo


def main():
    args = parse_args()
    _t0 = time.time()

    def stage(msg: str) -> None:
        """Timestamped stage marker, so a run that hits its wall clock still
        shows which phase consumed the time."""
        print(f"[stage +{time.time() - _t0:7.1f}s] {msg}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    s = args.subject
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(
        args.out_dir, f"{args.model_name}_subj{s:02d}.npz"
    )

    stage("design + betas")
    stim_info = nsd_data.load_stim_info()
    split = nsd_data.build_train_test_split(s, stim_info)
    mask = nsd_data.load_mask(s, "nsdgeneral")
    betas = nsd_data.load_betas(s, mask)
    nc = nsd_data.load_noise_ceiling(s, mask)

    Y_train = betas[split["train_trials"]]
    Y_test = betas[split["test_trials_per_image"]].mean(axis=1)
    del betas

    stage("features")
    all_ids = np.unique(
        np.concatenate([split["train_image_ids"], split["test_image_ids"]])
    )
    magno, parvo = get_features_for_images(args, all_ids, device)
    row_of = {int(i): k for k, i in enumerate(all_ids)}
    tr_rows = np.array([row_of[int(i)] for i in split["train_image_ids"]])
    te_rows = np.array([row_of[int(i)] for i in split["test_image_ids"]])

    stage(f"PCA to {args.pca_dim} per stream")
    # Project at IMAGE resolution and only then expand to trials. Indexing the
    # 15360-dim features by trial first copies ~1.8 GB per stream (repeats
    # included) and was what pushed this job out of memory.
    unique_tr_rows = np.unique(tr_rows)
    pcam = features.fit_pca(magno[unique_tr_rows], args.pca_dim, args.seed)
    Xm_img = pcam.transform(magno).astype(np.float32)
    del magno
    pcap = features.fit_pca(parvo[unique_tr_rows], args.pca_dim, args.seed)
    Xp_img = pcap.transform(parvo).astype(np.float32)
    del parvo

    Xm_tr, Xm_te = Xm_img[tr_rows], Xm_img[te_rows]
    Xp_tr, Xp_te = Xp_img[tr_rows], Xp_img[te_rows]
    del Xm_img, Xp_img

    # ---- cue-conflict design (PAPER_PLAN §7.3b / §21) ----
    # Projected with the PCA fit on the NSD *training* images, never refit on
    # the cue-conflict set: the panel asks what this fitted model predicts for
    # those images, so every stage upstream of the ridge must be the one the
    # ridge was fit through.
    X_extra = None
    cc_shapes = cc_textures = None
    if args.cue_conflict:
        stage("cue-conflict features")
        rgb_cc, cc_shapes, cc_textures = voxel_shape_bias.cue_conflict_design()
        ch_min, ch_max, cc_model = build_extractor(args, device)
        cc_m, cc_p = features_from_rgb(
            args, rgb_cc, ch_min, ch_max, cc_model, device
        )
        del rgb_cc, cc_model
        X_extra = [
            pcam.transform(cc_m).astype(np.float32),
            pcap.transform(cc_p).astype(np.float32),
        ]
        del cc_m, cc_p
        stage(f"  cue-conflict design {X_extra[0].shape[0]} images")

    stage(f"banded ridge over {Y_train.shape[1]} voxels "
          f"(chunk={args.voxel_chunk or 'all'}, folds={args.n_splits})")
    splits = banded_ridge.make_grouped_splits(
        split["train_image_ids"], n_splits=args.n_splits, seed=args.seed
    )
    # Voxels are fit independently (alphas and group weights are per target),
    # so chunking is statistically identical and bounds peak memory.
    res_chunks: dict[str, list[np.ndarray]] = {}
    n_vox = Y_train.shape[1]
    chunk = args.voxel_chunk if args.voxel_chunk > 0 else n_vox
    for start in range(0, n_vox, chunk):
        stop = min(start + chunk, n_vox)
        part_res = banded_ridge.fit_banded(
            [Xm_tr, Xp_tr], Y_train[:, start:stop],
            [Xm_te, Xp_te], Y_test[:, start:stop],
            splits, backend=args.backend, seed=args.seed,
            X_bands_extra=X_extra,
        )
        for k, v in part_res.items():
            res_chunks.setdefault(k, []).append(v)
        stage(f"  voxels {stop}/{n_vox} done")
    # Per-voxel statistics are (n_voxels,) and stack along axis 0; the
    # cue-conflict predictions are (n_images, n_voxels) and stack along the
    # voxel axis instead.
    res = {
        k: np.concatenate(v, axis=1 if k.startswith("pred_extra_") else 0)
        for k, v in res_chunks.items()
    }

    extras: dict[str, np.ndarray] = {}
    if args.cue_conflict:
        for band in ("magno", "parvo", "joint"):
            pred = res.pop(f"pred_extra_{band}")
            for k, v in voxel_shape_bias.voxel_shape_bias(
                pred, cc_shapes, cc_textures
            ).items():
                extras[f"cc_{band}_{k}"] = v
        extras["cc_n_images"] = np.asarray(len(cc_shapes))
        stage("  cue-conflict shape-bias index computed")

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
        **extras,
        **rois,
    )
    stage(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
