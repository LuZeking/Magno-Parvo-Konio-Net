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
import sys
import time

import numpy as np
import torch

from src.analysis.encoding import (banded_ridge, dkl_frontend, features,
                                    input_bands, template_design,
                                    voxel_shape_bias)
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
    p.add_argument("--dvd-age-months", type=float, default=None,
                   help="progressive (DVD) magno diet of the fs2_dvd* runs, "
                        "evaluated at this age (the run's dvd_end_months): "
                        "dkl_frontend.apply_dvd_diet replaces the fixed blur "
                        "and CS cut, so --no-lum-blur/--no-lum-cs are implied")
    p.add_argument("--transform-target", default="magno",
                   choices=("magno", "parvo"),
                   help="channel-swap control: put the developmental blur and "
                        "CS cut on rg instead of lum. Leg (c) had no way to "
                        "express this, so the swap condition existed for legs "
                        "(a) and (b) only.")
    p.add_argument("--chroma-channel", default="rg", choices=("rg", "by"),
                   help="which DKL opponent axis stream 2 receives: 'rg' "
                        "(L-M, parvocellular; the trained default) or 'by' "
                        "(S-(L+M), koniocellular). The konio arm of "
                        "PAPER_PLAN B3/§65 -- a single-factor swap of the "
                        "chromatic axis with the luminance stream, the "
                        "architecture and the seed all held fixed.")
    p.add_argument("--chroma-rescale", default="none",
                   choices=("none", "match-rg"),
                   help="'match-rg' rescales stream 2's channel to the "
                        "per-image RMS contrast of rg. REQUIRED for the "
                        "konio comparison to be about chromaticity: by "
                        "arrives with 2.05x rg's contrast on NSD stimuli, a "
                        "confound that points the way that flatters konio.")
    p.add_argument("--parvo-highsf", action="store_true",
                   help="biologically corrected parvo input: stream 2 gets "
                        "untransformed luminance stacked with rg (2 channels), "
                        "in_channels=(1,2).")
    p.add_argument("--cache-only", action="store_true",
                   help="write the feature cache and stop, skipping betas and "
                        "the ridge fit. The participation-ratio analysis reads "
                        "only the cache, so a front-end probe does not need "
                        "(and should not pay for) an encoding fit.")
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
    p.add_argument("--ridge-batched", action="store_true",
                   help="batch the single-band RidgeCV fits (n_targets_batch=500, "
                        "n_alphas_batch=7) and the banded fit at 1000 targets; "
                        "identical estimates, peak memory ~1/3 of the unbatched "
                        "call (see banded_ridge.fit_banded)")
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
    # docs/PLAN_CORTEX_DECISION_2026-10-26.md -- all four off by default, so a
    # run without them is byte-for-byte the run this file always produced.
    p.add_argument("--save-predictions", action="store_true",
                   help="store the cue-conflict (and template) predictions "
                        "themselves, (n_images, n_voxels) float16 per band, "
                        "not only the per-voxel index; requires --cue-conflict")
    p.add_argument("--template-design", default="none",
                   choices=("none", "miniecoset16"),
                   help="also push a template design through the fit "
                        "(template_design.build): 'miniecoset16' = 50 "
                        "MiniEcoset validation images of each of the 16 "
                        "Geirhos classes + the 80 same-cue Geirhos images; "
                        "requires --cue-conflict")
    p.add_argument("--save-encoder", action="store_true",
                   help="store the fitted estimators (ridge coefficients, band "
                        "weights, band statistics, PCA bases) so further "
                        "stimuli can be pushed through this fit later without "
                        "refitting it")
    p.add_argument("--reproduce-check", type=str, default=None,
                   help="a stored result npz of the same run x subject; after "
                        "the fit, max |diff| of r2_joint / unique_magno / "
                        "unique_parvo over its reliable_mask is recorded as "
                        "repro_max_abs_diff, and the process exits 3 if it "
                        "exceeds --reproduce-tol (the result is saved first)")
    p.add_argument("--reproduce-tol", type=float, default=1e-3)
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
    args = p.parse_args()
    if (args.save_predictions or args.template_design != "none") \
            and not args.cue_conflict:
        p.error("--save-predictions / --template-design require --cue-conflict")
    if args.dvd_age_months is not None:
        # Recorded as such in the provenance, so a DVD-diet cache can never
        # pass for a blur+CS one.
        args.no_lum_blur = args.no_lum_cs = True
    return args


def check_frontend_matches_checkpoint(args) -> None:
    """Refuse a front end that is not the one the checkpoint was trained with.

    Stage-1/2 fusion checkpoints carry their condition config. Their DVD-diet
    arms are evaluated through ``apply_dvd_diet`` at ``dvd_end_months``, which
    no combination of the blur/CS flags expresses, so encoding such a
    checkpoint with the default flags would silently feed it the thesis diet.
    Older checkpoints carry no config and are left to the caller's flags.
    """
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = state.get("config") if isinstance(state, dict) else None
    if not isinstance(cfg, dict) or "input_mode" not in cfg:
        return
    dvd = cfg.get("dvd_end_months")
    want = {
        "dvd_age_months": None if dvd is None else float(dvd),
        "lum_blur": False if dvd is not None else bool(cfg["blur"]),
        "lum_cs": False if dvd is not None else bool(cfg["cs"]),
        "transform_target": cfg.get("transform_target", "magno"),
        "parvo_highsf": bool(cfg.get("parvo_highsf", False)),
    }
    got = {
        "dvd_age_months": args.dvd_age_months,
        "lum_blur": not args.no_lum_blur,
        "lum_cs": not args.no_lum_cs,
        # readout_w_cortex has neither flag; it only builds the default arm
        "transform_target": getattr(args, "transform_target", "magno"),
        "parvo_highsf": getattr(args, "parvo_highsf", False),
    }
    bad = {k: (got[k], want[k]) for k in want if got[k] != want[k]}
    if cfg.get("blur_sigma") is not None:
        bad["blur_sigma"] = (None, cfg["blur_sigma"])
    if bad:
        raise ValueError(
            f"front end does not match {args.checkpoint} (requested, trained): "
            f"{bad}"
        )
    print(f"[frontend] matches checkpoint config: {want}", flush=True)


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
            in_channels=(1, 2) if args.parvo_highsf else (1, 1),
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
            transform_target=args.transform_target,
            parvo_highsf=args.parvo_highsf,
            chroma_channel=args.chroma_channel,
            chroma_rescale=args.chroma_rescale,
            dvd_age=args.dvd_age_months,
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


def assemble_chunks(
    res_chunks: dict[str, list[np.ndarray]]
) -> dict[str, np.ndarray]:
    """Join the per-voxel-chunk results of ``fit_banded``.

    Per-voxel statistics are (n_voxels,) and stack along axis 0; predictions
    (n_images, n_voxels) and coefficients (n_features, n_voxels) carry voxels
    on axis 1; the band statistics ``enc_band_*`` do not depend on the chunk
    and are taken once.
    """
    out: dict[str, np.ndarray] = {}
    for k, v in res_chunks.items():
        if k.startswith("enc_band_"):
            out[k] = v[0]
        elif k.startswith(("pred_extra_", "enc_coef_", "enc_deltas_")):
            out[k] = np.concatenate(v, axis=1)
        else:
            out[k] = np.concatenate(v, axis=0)
    return out


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
        # -1 = no DVD diet (the fixed blur/CS front end above applies)
        "prov_dvd_age_months": np.asarray(
            -1.0 if args.dvd_age_months is None else float(args.dvd_age_months)),
        "prov_features": np.asarray(args.features),
        # Recorded because a .npz that does not name its front end cannot be
        # told apart from another condition's afterwards -- exactly how the
        # contaminated no-diet subject in §22 went unnoticed.
        "prov_transform_target": np.asarray(args.transform_target),
        "prov_parvo_highsf": np.asarray(args.parvo_highsf),
        "prov_chroma_channel": np.asarray(args.chroma_channel),
        "prov_chroma_rescale": np.asarray(args.chroma_rescale),
        "prov_checkpoint": np.asarray(args.checkpoint or ""),
        "prov_stream1_weight": np.asarray(args.stream1_weight),
        "prov_seed": np.asarray(args.seed),
    }


# Provenance keys added after caches were already on disk, with the value any
# pre-existing cache must have had. `--transform-target` and `--parvo-highsf`
# did not exist in this entry point until 2026-09-01, so a cache written
# before then was necessarily produced with the front-end defaults -- no other
# value was expressible. Treating an absent key as its default therefore
# preserves the safety property (asking for a non-default front end against an
# old cache still MISMATCHES and recomputes) without invalidating every cache
# on disk, which would have forced feature recomputation on the 17 remaining
# tasks of the in-flight §22 repair job.
BACKFILL_DEFAULTS = {
    "prov_transform_target": "magno",
    "prov_parvo_highsf": False,
    # `--chroma-channel` and `--chroma-rescale` did not exist until
    # 2026-09-03, so every cache written before then necessarily used the rg
    # axis unrescaled -- it was the only slice the code took. Backfilling
    # preserves the safety property (asking for `by` against an rg cache
    # still MISMATCHES and recomputes) without invalidating the 256 caches
    # already on disk.
    "prov_chroma_channel": "rg",
    "prov_chroma_rescale": "none",
    # `--dvd-age-months` did not exist until 2026-10-16; every earlier cache
    # used the fixed blur/CS front end.
    "prov_dvd_age_months": -1.0,
}


def cache_is_valid(z, want: dict[str, np.ndarray]) -> bool:
    """True only if the cache records provenance and it matches."""
    missing = [k for k in want
               if k not in z.files and k not in BACKFILL_DEFAULTS]
    if missing:
        print(
            "[cache] no provenance recorded in this cache file "
            f"(missing {len(missing)} keys) -- recomputing features rather "
            "than trusting it",
            flush=True,
        )
        return False
    for k, v in want.items():
        if k in z.files:
            cached = z[k].item()
        else:
            cached = BACKFILL_DEFAULTS[k]
            print(f"[cache] {k} absent, assuming pre-flag default "
                  f"{cached!r}", flush=True)
        if cached != v.item():
            print(
                f"[cache] MISMATCH {k}: cached={cached!r} "
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
    if args.checkpoint:
        check_frontend_matches_checkpoint(args)
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

    if args.cache_only:
        # Deliberately before load_betas: the betas are the expensive part of
        # this stage and nothing downstream of here is read by the PR analysis.
        stage("features (cache-only)")
        get_features_for_images(
            args,
            np.unique(np.concatenate(
                [split["train_image_ids"], split["test_image_ids"]]
            )),
            device,
        )
        stage("cache written; no fit requested")
        return

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
    n_cc = 0
    tpl = None
    if args.cue_conflict:
        stage("cue-conflict features")
        rgb_cc, cc_shapes, cc_textures = voxel_shape_bias.cue_conflict_design()
        n_cc = len(cc_shapes)
        ch_min, ch_max, cc_model = build_extractor(args, device)
        cc_m, cc_p = features_from_rgb(
            args, rgb_cc, ch_min, ch_max, cc_model, device
        )
        del rgb_cc
        if args.template_design != "none":
            # PLAN_CORTEX_DECISION_2026-10-26: the templates and the same-cue
            # images ride along as further rows of the same extra design --
            # same front end, feature stage, PCA and band statistics -- and
            # are split off by row after the fit. Image sets of different
            # sizes go through the extractor separately (the front end
            # resizes each set to 256 px).
            tpl = template_design.build(args.template_design)
            tm, tp = [cc_m], [cc_p]
            for rgb_t in tpl.pop("rgb"):
                m_t, p_t = features_from_rgb(
                    args, rgb_t, ch_min, ch_max, cc_model, device
                )
                tm.append(m_t)
                tp.append(p_t)
            cc_m, cc_p = np.concatenate(tm), np.concatenate(tp)
            del tm, tp
            stage(f"  template design {args.template_design}: "
                  f"{len(tpl['labels'])} images")
        del cc_model
        X_extra = [
            pcam.transform(cc_m).astype(np.float32),
            pcap.transform(cc_p).astype(np.float32),
        ]
        del cc_m, cc_p
        stage(f"  cue-conflict design {n_cc} images"
              + (f" + {X_extra[0].shape[0] - n_cc} template rows"
                 if tpl is not None else ""))

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
            save_encoder=args.save_encoder,
            single_solver_params=(dict(n_targets_batch=500, n_alphas_batch=7, n_targets_batch_refit=500)
                                  if args.ridge_batched else None),
            banded_targets_batch=1000 if args.ridge_batched else 2000,
            # the (n_alphas, n_test, n_train) intermediate of the banded random
            # search is the real memory peak (30 GB unbatched); see fit_banded
            banded_alphas_batch=7 if args.ridge_batched else None,
        )
        for k, v in part_res.items():
            res_chunks.setdefault(k, []).append(v)
        stage(f"  voxels {stop}/{n_vox} done")
    # Per-voxel statistics are (n_voxels,) and stack along axis 0; the
    # cue-conflict predictions are (n_images, n_voxels) and stack along the
    # voxel axis instead.
    res = assemble_chunks(res_chunks)

    extras: dict[str, np.ndarray] = {}
    if args.cue_conflict:
        for band in ("magno", "parvo", "joint"):
            pred = res.pop(f"pred_extra_{band}")
            pred_cc, pred_tpl = pred[:n_cc], pred[n_cc:]
            for k, v in voxel_shape_bias.voxel_shape_bias(
                pred_cc, cc_shapes, cc_textures
            ).items():
                extras[f"cc_{band}_{k}"] = v
            if args.save_predictions:
                extras[f"cc_pred_{band}"] = pred_cc.astype(np.float16)
                if tpl is not None:
                    extras[f"tpl_pred_{band}"] = pred_tpl.astype(np.float16)
        extras["cc_n_images"] = np.asarray(n_cc)
        if args.save_predictions:
            stems = np.asarray(template_design.cue_conflict_stems())
            if len(stems) != n_cc:
                raise RuntimeError(
                    f"cue-conflict listing changed under us: {len(stems)} "
                    f"stems for {n_cc} design rows"
                )
            extras["cc_shapes"] = np.asarray(cc_shapes)
            extras["cc_textures"] = np.asarray(cc_textures)
            extras["cc_stems"] = stems
            if tpl is not None:
                extras["tpl_labels"] = np.asarray(tpl["labels"])
                extras["tpl_source"] = np.asarray(tpl["source"])
                extras["tpl_stems"] = np.asarray(tpl["stems"])
        stage("  cue-conflict shape-bias index computed")
    if args.save_encoder:
        # The stored estimators act on PCA-projected, band-scaled features:
        #   z = ((x - enc_pca_mean) @ enc_pca_components.T - enc_band_mu)
        #   pred = (z[:, enc_band_keep] / enc_band_scale) @ enc_coef
        # with the joint design the concatenation of the two scaled bands.
        for name, pca in (("magno", pcam), ("parvo", pcap)):
            extras[f"enc_pca_mean_{name}"] = pca.mean_.astype(np.float32)
            extras[f"enc_pca_components_{name}"] = (
                pca.components_.astype(np.float16)
            )
        stage("  encoder stored")

    part = banded_ridge.variance_partition(
        res["r2_joint"], res["r2_magno"], res["r2_parvo"], noise_ceiling=nc
    )

    repro: dict[str, np.ndarray] = {}
    if args.reproduce_check:
        ref = np.load(args.reproduce_check, allow_pickle=False)
        rm = ref["reliable_mask"]
        if rm.shape != res["r2_joint"].shape:
            raise ValueError(
                f"reproduce-check: {args.reproduce_check} has "
                f"{rm.shape[0]} voxels, this fit {res['r2_joint'].shape[0]}"
            )
        diffs = {
            "r2_joint": float(np.nanmax(np.abs(res["r2_joint"][rm] - ref["r2_joint"][rm]))),
            "unique_magno": float(np.nanmax(np.abs(part["unique_magno"][rm] - ref["unique_magno"][rm]))),
            "unique_parvo": float(np.nanmax(np.abs(part["unique_parvo"][rm] - ref["unique_parvo"][rm]))),
        }
        repro = {
            "repro_ref": np.asarray(args.reproduce_check),
            "repro_max_abs_diff": np.asarray(max(diffs.values())),
            "repro_tol": np.asarray(args.reproduce_tol),
        }
        stage(f"  reproduce-check vs {args.reproduce_check}: {diffs}")

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
        # The front end this result was computed through. cache_provenance()
        # records the same settings, but only inside the *feature cache*; the
        # result file itself carried nothing, so a finished .npz could not be
        # told apart from another condition's after the fact. That is exactly
        # the position §22 left us in -- the contaminated no-diet subject had
        # to be identified by re-deriving features and comparing them to
        # bit-equality, because the file did not say what produced it.
        **cache_provenance(args),
        **res,
        **{k: v for k, v in part.items()},
        **extras,
        **rois,
        **repro,
    )
    stage(f"saved -> {out_path}")
    if repro and float(repro["repro_max_abs_diff"]) > args.reproduce_tol:
        print(
            f"FATAL: refit does not reproduce {args.reproduce_check}: max "
            f"|diff| {float(repro['repro_max_abs_diff']):.3e} > "
            f"{args.reproduce_tol:g} (result saved, exit 3)",
            file=sys.stderr, flush=True,
        )
        sys.exit(3)


if __name__ == "__main__":
    main()
