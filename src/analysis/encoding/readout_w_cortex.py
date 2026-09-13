"""Which readout weight does cortex prefer?

The manuscript's readout claim (Results 2.4) sweeps the stream-mixing weight w
of the classifier input  z(w) = w*y1 + (1-w)*y2  (y1: magno stream output,
y2: parvo stream output, both 512-d pooled vectors) and reads the cue-conflict
shape-bias rate at each w; the weight that best reproduces human trial-by-trial
choices is fitted in results/fit_w_observers.

This script asks the same question of the brain: for each w on a grid, the
identical mixture z(w) is used as the design of a ridge encoding model of every
nsdgeneral voxel of one NSD subject (same train/test split, same grouped folds
and alpha grid as run_encoding.py), and the test R^2 per voxel is stored. The
per-ROI w that maximises median R^2 is the cortex-optimal readout weight, to
be set against the behaviour-optimal one. Because z(w) sums the two streams
dimension by dimension, the ridge cannot undo the mixture: w is a genuine
constraint, not a re-parameterisation of the two-band model.

The concatenation [y1, y2] is fitted as well, as the ceiling any single mixture
can reach.

Outputs one .npz per (model, subject) with r2 (n_w, n_vox), the grid, the noise
ceiling and the ROI labels (prf-visualrois, streams, Kastner2015).

Usage (one array element):
  python -m src.analysis.encoding.readout_w_cortex --subject 1 \
      --checkpoint /scratch/.../dkl_bcs_seed0/checkpoint_last.pt \
      --model-name dkl_bcs_seed0_trained --out-dir /scratch/byrong/mpk/readout_w/results/<run> \
      [--no-lum-blur --no-lum-cs]
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from src.analysis.encoding import banded_ridge, dkl_frontend, features, nsd_data


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--subject", type=int, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--model-name", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--feature-dir", type=str, default="/scratch/byrong/mpk/readout_w/features")
    p.add_argument("--beta-cache", type=str, default="/scratch/byrong/mpk/readout_w/betas",
                   help="directory for per-subject cached betas ('' disables)")
    p.add_argument("--no-lum-blur", action="store_true")
    p.add_argument("--no-lum-cs", action="store_true")
    p.add_argument("--age-months", type=int, default=48)
    p.add_argument("--n-w", type=int, default=21)
    p.add_argument("--n-splits", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--quick", action="store_true",
                   help="smoke test: 2 sessions, 3 grid points, 500 voxels")
    return p.parse_args()


@torch.no_grad()
def stream_outputs(model, rgb_uint8, ch_min, ch_max, args, device):
    """RGB uint8 (N,H,W,3) -> (y1, y2), each (N, 512) float32: the stream
    outputs the classifier mixes. Mirrors cue_conflict.stream_logits."""
    y1s, y2s = [], []
    for start in range(0, len(rgb_uint8), 512):
        x_m, x_p = dkl_frontend.magno_parvo_inputs(
            rgb_uint8[start:start + 512], ch_min, ch_max,
            lum_blur=not args.no_lum_blur, lum_cs=not args.no_lum_cs,
            age_months=args.age_months, device=device,
        )
        for b in range(0, x_m.shape[0], args.batch_size):
            y1 = model.stream1_block1(x_m[b:b + args.batch_size].to(device))
            y2 = model.stream2_block1(x_p[b:b + args.batch_size].to(device))
            if model.stream_concat:
                y1, y2 = model._mix_streams(y1, y2)
                y1 = model.stream1_block2(y1)
                y2 = model.stream2_block2(y2)
            # For these checkpoints the stream blocks end at layer4 (512 x 8 x 8)
            # and the classifier is [avgpool, flatten, linear]. Pooling is
            # linear, so avgpool(w*y1 + (1-w)*y2) = w*avgpool(y1) + (1-w)*avgpool(y2):
            # the pooled 512-d vectors are exactly what the classifier mixes.
            y1s.append(F.adaptive_avg_pool2d(y1, 1).flatten(1).float().cpu().numpy())
            y2s.append(F.adaptive_avg_pool2d(y2, 1).flatten(1).float().cpu().numpy())
    return np.concatenate(y1s), np.concatenate(y2s)


def get_outputs(args, image_ids, device):
    os.makedirs(args.feature_dir, exist_ok=True)
    cache = os.path.join(args.feature_dir, f"{args.model_name}_subj{args.subject:02d}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        if np.array_equal(z["image_ids"], image_ids):
            print("[cache] stream outputs hit", flush=True)
            return z["y1"], z["y2"]
    ch_min, ch_max = dkl_frontend.compute_dkl_channel_minmax()
    features.assert_converged(args.checkpoint)
    model = features.build_model(checkpoint=args.checkpoint, device=device)
    y1s, y2s = [], []
    for start in range(0, len(image_ids), 2048):
        rgb = dkl_frontend.load_nsd_images(image_ids[start:start + 2048])
        a, b = stream_outputs(model, rgb, ch_min, ch_max, args, device)
        y1s.append(a); y2s.append(b)
        print(f"  outputs {min(start + 2048, len(image_ids))}/{len(image_ids)}", flush=True)
    y1, y2 = np.concatenate(y1s), np.concatenate(y2s)
    if y1.shape[1] != 512 or y2.shape[1] != 512:
        raise RuntimeError(f"expected pooled 512-d stream outputs, got {y1.shape} {y2.shape}")
    np.savez(cache, image_ids=image_ids, y1=y1, y2=y2)
    return y1, y2


def fit_r2(X_img, tr_rows, te_rows, Y_train, Y_test, splits, alphas):
    from himalaya.backend import set_backend
    from himalaya.ridge import RidgeCV
    backend = set_backend("torch", on_error="warn")
    Xtr = X_img[tr_rows].astype(np.float64)
    Xte = X_img[te_rows].astype(np.float64)
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True)
    sd[sd == 0] = 1.0
    # float32 throughout and small target batches: with 21 alphas the
    # validation predictions alone are n_alphas x n_val x n_targets_batch,
    # which at 4000 targets in float64 was ~7 GB and pushed the job past 32G.
    Xtr = ((Xtr - mu) / sd).astype(np.float32)
    Xte = ((Xte - mu) / sd).astype(np.float32)
    ridge = RidgeCV(alphas=alphas, cv=splits,
                    solver_params=dict(n_targets_batch=500, n_alphas_batch=7, n_targets_batch_refit=500))
    ridge.fit(Xtr, Y_train)
    pred = banded_ridge._as_numpy(ridge.predict(Xte), backend)
    return banded_ridge.r2_per_target(Y_test, pred).astype(np.float32)


def main():
    args = parse_args()
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "8"))))
    s = args.subject
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.model_name}_subj{s:02d}.npz")

    stim_info = nsd_data.load_stim_info()
    split = nsd_data.build_train_test_split(s, stim_info)
    sessions = nsd_data.available_sessions(s)
    if args.quick:
        sessions = sessions[:2]
    mask = nsd_data.load_mask(s, "nsdgeneral")
    nc = nsd_data.load_noise_ceiling(s, mask)
    # Reading a subject's ~30 HDF5 sessions takes ~25 min; six models per
    # subject share it through an uncompressed cache (atomic write).
    bcache = os.path.join(args.beta_cache, f"subj{s:02d}{'_quick' if args.quick else ''}.npz") if args.beta_cache else None
    if bcache and os.path.exists(bcache):
        zb = np.load(bcache)
        betas = zb["betas"]
        print(f"[+{time.time()-t0:6.0f}s] betas {betas.shape} (cache)", flush=True)
    else:
        betas = nsd_data.load_betas(s, mask, sessions=sessions)
        print(f"[+{time.time()-t0:6.0f}s] betas {betas.shape}", flush=True)
        if bcache:
            os.makedirs(args.beta_cache, exist_ok=True)
            tmp = bcache[:-4] + f".tmp{os.getpid()}.npz"
            np.savez(tmp, betas=betas, sessions=np.array(sessions))
            os.replace(tmp, bcache)
    n_trials = betas.shape[0]

    if args.quick:
        # restrict the split to the loaded trials
        keep = split["train_trials"] < n_trials
        split["train_trials"] = split["train_trials"][keep]
        split["train_image_ids"] = split["train_image_ids"][keep]
        ok = (split["test_trials_per_image"] < n_trials).all(1)
        split["test_trials_per_image"] = split["test_trials_per_image"][ok]
        split["test_image_ids"] = split["test_image_ids"][ok]
        vox = np.flatnonzero(nc > 0.1)[:500]
        betas, nc = betas[:, vox], nc[vox]
        mask_idx = np.flatnonzero(mask.reshape(-1))[vox]
    Y_train = betas[split["train_trials"]]
    Y_test = betas[split["test_trials_per_image"]].mean(axis=1)
    del betas

    all_ids = np.unique(np.concatenate([split["train_image_ids"], split["test_image_ids"]]))
    y1, y2 = get_outputs(args, all_ids, device)
    row_of = {int(i): k for k, i in enumerate(all_ids)}
    tr_rows = np.array([row_of[int(i)] for i in split["train_image_ids"]])
    te_rows = np.array([row_of[int(i)] for i in split["test_image_ids"]])
    print(f"[+{time.time()-t0:6.0f}s] outputs {y1.shape} {y2.shape}", flush=True)

    splits = banded_ridge.make_grouped_splits(split["train_image_ids"], n_splits=args.n_splits, seed=args.seed)
    alphas = np.logspace(-2, 8, 21)
    grid = np.linspace(0, 1, 3 if args.quick else args.n_w)
    r2 = np.full((len(grid), Y_train.shape[1]), np.nan, dtype=np.float32)
    for i, w in enumerate(grid):
        r2[i] = fit_r2(w * y1 + (1 - w) * y2, tr_rows, te_rows, Y_train, Y_test, splits, alphas)
        v = nc > 0.1
        print(f"[+{time.time()-t0:6.0f}s] w={w:.2f}  median r2 (nc>0.1) {np.nanmedian(r2[i][v]):.4f}", flush=True)
    r2_concat = fit_r2(np.concatenate([y1, y2], 1), tr_rows, te_rows, Y_train, Y_test, splits, alphas)
    print(f"[+{time.time()-t0:6.0f}s] concat  median r2 (nc>0.1) {np.nanmedian(r2_concat[nc > 0.1]):.4f}", flush=True)

    rois = {}
    for name in ("prf-visualrois", "streams", "Kastner2015", "prf-eccrois"):
        lab = nsd_data.load_roi_labels(s, name, mask)
        rois[f"roi_{name}"] = lab[vox] if args.quick else lab
    np.savez(out_path, subject=s, model_name=args.model_name, grid=grid, r2=r2, r2_concat=r2_concat,
             noise_ceiling=nc, n_train_trials=len(split["train_trials"]), n_test_images=len(split["test_image_ids"]),
             lum_blur=not args.no_lum_blur, lum_cs=not args.no_lum_cs, checkpoint=args.checkpoint, **rois)
    print(f"[+{time.time()-t0:6.0f}s] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
