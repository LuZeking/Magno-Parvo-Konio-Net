"""NSD encoding with the shape- and texture-leaning halves as bands (PLAN_GENERALITY D4).

  python -m src.analysis.generality.cortex_bands --subject 1 \
      --nsd-features DIR/<tag>_nsd.npz --factor DIR/<tag>_unit_factor.npz --out-dir OUT

For one frozen network and one NSD subject: the penultimate units are split
into the shape-leaning half S and the texture-leaning half T (the split defined
on all 1,200 cue-conflict images, unit_factor.npz), and

  * single-band ridge on S, single-band ridge on T, banded ridge on [S, T]
    (same split, grouped folds and alpha grid as Methods 4.5) give per-voxel
    test R^2 and the unique variances U_S = R2_joint - R2_T, U_T = R2_joint - R2_S,
    noise-ceiling-normalised, and the index (U_S - U_T)/(U_S + U_T);
  * the lossy sum z(w) = w*y_S + (1-w)*y_T (halves paired by rank of d) is the
    design of a single-band ridge for w in {0, 0.05, ..., 1}, giving the
    cortex-optimal mixing weight as in Extended Data Fig. 5.

Memory is bounded by float32 designs and small target batches (the 2026-09-13
readout_w job measured 25.6 GB peak for the same kind of fit).
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

from src.analysis.encoding import banded_ridge, nsd_data
from src.analysis.encoding.readout_w_cortex import fit_r2

ALPHAS = np.logspace(-2, 8, 21)
SOLVER = dict(n_targets_batch=500, n_alphas_batch=7, n_targets_batch_refit=500)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--subject", type=int, required=True)
    p.add_argument("--nsd-features", required=True)
    p.add_argument("--factor", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--beta-cache", default="/scratch/byrong/mpk/readout_w/betas")
    p.add_argument("--n-w", type=int, default=21)
    p.add_argument("--n-splits", type=int, default=3)
    p.add_argument("--n-iter", type=int, default=20, help="banded random-search samples")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-w-sweep", action="store_true")
    p.add_argument("--only-w-sweep", action="store_true")
    p.add_argument("--quick", action="store_true", help="smoke test: 2 sessions, 500 voxels, 3 w")
    return p.parse_args()


def fit_single(Xtr, Xte, Y_train, Y_test, splits):
    from himalaya.backend import set_backend
    from himalaya.ridge import RidgeCV
    backend = set_backend("torch", on_error="warn")
    ridge = RidgeCV(alphas=ALPHAS, cv=splits, solver_params=SOLVER)
    ridge.fit(Xtr.astype(np.float32), Y_train)
    pred = banded_ridge._as_numpy(ridge.predict(Xte.astype(np.float32)), backend)
    return banded_ridge.r2_per_target(Y_test, pred).astype(np.float32)


def fit_joint(XS_tr, XT_tr, XS_te, XT_te, Y_train, Y_test, splits, n_iter, seed):
    from himalaya.backend import set_backend
    from himalaya.ridge import BandedRidgeCV
    backend = set_backend("torch", on_error="warn")
    groups = np.r_[np.zeros(XS_tr.shape[1], int), np.ones(XT_tr.shape[1], int)]
    Xtr = np.concatenate([XS_tr, XT_tr], 1).astype(np.float32)
    Xte = np.concatenate([XS_te, XT_te], 1).astype(np.float32)
    banded = BandedRidgeCV(groups=groups, solver="random_search",
                           solver_params=dict(n_iter=n_iter, alphas=ALPHAS, **SOLVER),
                           random_state=seed, cv=splits)
    banded.fit(Xtr, Y_train)
    pred = banded_ridge._as_numpy(banded.predict(Xte), backend)
    return banded_ridge.r2_per_target(Y_test, pred).astype(np.float32)


def main():
    args = parse_args()
    t0 = time.time()
    import torch
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "8"))))
    s = args.subject
    os.makedirs(args.out_dir, exist_ok=True)

    zf = np.load(args.nsd_features, allow_pickle=True)
    tag = str(zf["tag"])
    fac = np.load(args.factor, allow_pickle=True)
    if str(fac["tag"]) != tag:
        raise SystemExit(f"tag mismatch: features {tag} vs factor {fac['tag']}")
    S, d = fac["S_full"].astype(bool), fac["d_full"]
    suffix = "_w" if args.only_w_sweep else ""
    out_path = os.path.join(args.out_dir, f"{tag}_subj{s:02d}{suffix}.npz")

    stim_info = nsd_data.load_stim_info()
    split = nsd_data.build_train_test_split(s, stim_info)
    sessions = nsd_data.available_sessions(s)
    if args.quick:
        sessions = sessions[:2]
    mask = nsd_data.load_mask(s, "nsdgeneral")
    nc = nsd_data.load_noise_ceiling(s, mask)
    bcache = os.path.join(args.beta_cache, f"subj{s:02d}{'_quick' if args.quick else ''}.npz")
    if os.path.exists(bcache):
        betas = np.load(bcache)["betas"]
        print(f"[+{time.time() - t0:6.0f}s] betas {betas.shape} (cache)", flush=True)
    else:
        betas = nsd_data.load_betas(s, mask, sessions=sessions)
        os.makedirs(args.beta_cache, exist_ok=True)
        tmp = bcache[:-4] + f".tmp{os.getpid()}.npz"
        np.savez(tmp, betas=betas, sessions=np.array(sessions))
        os.replace(tmp, bcache)
        print(f"[+{time.time() - t0:6.0f}s] betas {betas.shape}", flush=True)
    n_trials = betas.shape[0]
    vox = None
    if args.quick:
        keep = split["train_trials"] < n_trials
        split["train_trials"], split["train_image_ids"] = split["train_trials"][keep], split["train_image_ids"][keep]
        ok = (split["test_trials_per_image"] < n_trials).all(1)
        split["test_trials_per_image"], split["test_image_ids"] = split["test_trials_per_image"][ok], split["test_image_ids"][ok]
        vox = np.flatnonzero(nc > 0.1)[:500]
        betas, nc = betas[:, vox], nc[vox]
    Y_train = betas[split["train_trials"]]
    Y_test = betas[split["test_trials_per_image"]].mean(axis=1)
    del betas

    row_of = {int(i): k for k, i in enumerate(zf["image_ids"])}
    feats = zf["features"]
    Xtr_all = feats[[row_of[int(i)] for i in split["train_image_ids"]]].astype(np.float32)
    Xte_all = feats[[row_of[int(i)] for i in split["test_image_ids"]]].astype(np.float32)
    del feats
    print(f"[+{time.time() - t0:6.0f}s] design {Xtr_all.shape} / {Xte_all.shape}; S {int(S.sum())} T {int((~S).sum())} units", flush=True)
    splits = banded_ridge.make_grouped_splits(split["train_image_ids"], n_splits=args.n_splits, seed=args.seed)

    out = dict(subject=s, tag=tag, noise_ceiling=nc, n_train_trials=len(split["train_trials"]),
               n_test_images=len(split["test_image_ids"]), n_S=int(S.sum()), n_T=int((~S).sum()))
    for name in ("prf-visualrois", "streams", "Kastner2015", "prf-eccrois"):
        lab = nsd_data.load_roi_labels(s, name, mask)
        out[f"roi_{name}"] = lab[vox] if vox is not None else lab

    if not args.only_w_sweep:
        # the paper's band preparation: centre, drop degenerate dims, unit RMS scale (training statistics)
        XS_tr, XS_te, _ = banded_ridge.prepare_band(Xtr_all[:, S], Xte_all[:, S])
        XT_tr, XT_te, _ = banded_ridge.prepare_band(Xtr_all[:, ~S], Xte_all[:, ~S])
        r2_S = fit_single(XS_tr, XS_te, Y_train, Y_test, splits)
        print(f"[+{time.time() - t0:6.0f}s] S band  median r2 (nc>0.1) {np.nanmedian(r2_S[nc > 0.1]):.4f}", flush=True)
        r2_T = fit_single(XT_tr, XT_te, Y_train, Y_test, splits)
        print(f"[+{time.time() - t0:6.0f}s] T band  median r2 (nc>0.1) {np.nanmedian(r2_T[nc > 0.1]):.4f}", flush=True)
        r2_joint = fit_joint(XS_tr, XT_tr, XS_te, XT_te, Y_train, Y_test, splits, args.n_iter, args.seed)
        print(f"[+{time.time() - t0:6.0f}s] joint   median r2 (nc>0.1) {np.nanmedian(r2_joint[nc > 0.1]):.4f}", flush=True)
        vp = banded_ridge.variance_partition(r2_joint, r2_S, r2_T, noise_ceiling=nc)
        out.update(r2_S=r2_S, r2_T=r2_T, r2_joint=r2_joint,
                   unique_S=vp["unique_magno"], unique_T=vp["unique_parvo"], shared=vp["shared"],
                   unique_S_pre_nc=vp["unique_magno_pre_nc"], unique_T_pre_nc=vp["unique_parvo_pre_nc"],
                   r2_joint_nc=vp["r2_joint_nc"])
        v = (nc > 0.1) & ((out["unique_S"] + out["unique_T"]) > 1e-3)
        idx = (out["unique_S"][v] - out["unique_T"][v]) / (out["unique_S"][v] + out["unique_T"][v])
        print(f"[+{time.time() - t0:6.0f}s] index median over reliable voxels {np.nanmedian(idx):+.3f} "
              f"(U_S {np.nanmedian(out['unique_S'][v]):.3f}, U_T {np.nanmedian(out['unique_T'][v]):.3f})", flush=True)

    if not args.skip_w_sweep:
        S_order = np.flatnonzero(S)[np.argsort(-d[S], kind="stable")]
        T_order = np.flatnonzero(~S)[np.argsort(d[~S], kind="stable")]
        k = min(len(S_order), len(T_order))
        S_order, T_order = S_order[:k], T_order[:k]
        grid = np.linspace(0, 1, 3 if args.quick else args.n_w)
        r2_w = np.full((len(grid), Y_train.shape[1]), np.nan, dtype=np.float32)
        tr_rows, te_rows = np.arange(Xtr_all.shape[0]), np.arange(Xtr_all.shape[0], Xtr_all.shape[0] + Xte_all.shape[0])
        X_all = np.concatenate([Xtr_all, Xte_all], 0)
        for i, w in enumerate(grid):
            Z = w * X_all[:, S_order] + (1 - w) * X_all[:, T_order]
            r2_w[i] = fit_r2(Z, tr_rows, te_rows, Y_train, Y_test, splits, ALPHAS)
            print(f"[+{time.time() - t0:6.0f}s] w={w:.2f}  median r2 (nc>0.1) {np.nanmedian(r2_w[i][nc > 0.1]):.4f}", flush=True)
        out.update(grid=grid, r2_w=r2_w)

    np.savez(out_path, **out)
    print(f"[+{time.time() - t0:6.0f}s] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
