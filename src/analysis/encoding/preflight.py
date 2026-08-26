"""Cheap end-to-end check of every I/O path used by run_encoding.

Exercises the design table, split construction, mask/ncsnr/ROI loading, a
single session of betas, NSD stimulus reads, the DKL front end, feature
extraction and the solver — each at minimal scale — so data-layout problems
surface in seconds instead of after a long fit.

Run:  python -m src.analysis.encoding.preflight --subject 1
"""

from __future__ import annotations

import argparse
import traceback

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, default=1)
    args = ap.parse_args()
    s = args.subject
    failures: list[str] = []

    skipped: list[str] = []

    def check(name, fn):
        try:
            out = fn()
            # A check that quietly declined to do its work must not read as a
            # pass: the first version of the mini fit silently skipped the fit
            # and still printed PREFLIGHT PASSED.
            if isinstance(out, str) and "skip" in out.lower():
                print(f"  SKIP {name}: {out}", flush=True)
                skipped.append(name)
                return False
            print(f"  OK   {name}: {out}", flush=True)
            return True
        except Exception:
            print(f"  FAIL {name}", flush=True)
            traceback.print_exc()
            failures.append(name)
            return False

    from src.analysis.encoding import dkl_frontend, features, nsd_data

    print(f"=== preflight, subject {s} ===", flush=True)

    stim_info = None
    mask = None
    split = None

    def _stim():
        nonlocal stim_info
        stim_info = nsd_data.load_stim_info()
        return f"{len(stim_info)} rows, {int(stim_info['shared1000'].sum())} shared"

    def _sessions():
        return f"{len(nsd_data.available_sessions(s))} sessions on disk"

    def _split():
        nonlocal split
        split = nsd_data.build_train_test_split(s, stim_info)
        return (f"{len(split['train_trials'])} train trials, "
                f"{len(split['test_image_ids'])} test images")

    def _mask():
        nonlocal mask
        mask = nsd_data.load_mask(s, "nsdgeneral")
        return f"{int(mask.sum())} voxels, shape {mask.shape}"

    def _nc():
        nc = nsd_data.load_noise_ceiling(s, mask)
        return (f"NC median {np.nanmedian(nc):.3f}, "
                f">0.1 in {int((nc > 0.1).sum())} voxels")

    def _rois():
        got = {}
        for roi in ("streams", "prf-eccrois", "prf-visualrois"):
            got[roi] = len(np.unique(nsd_data.load_roi_labels(s, roi, mask)))
        return got

    def _betas():
        one = nsd_data.load_betas(s, mask, sessions=[1])
        return f"one session -> {one.shape}, finite={np.isfinite(one).all()}"

    def _images():
        ids = split["test_image_ids"][:4]
        rgb = dkl_frontend.load_nsd_images(ids)
        return f"{rgb.shape} {rgb.dtype}"

    def _frontend():
        ch_min, ch_max = dkl_frontend.compute_dkl_channel_minmax()
        rgb = dkl_frontend.load_nsd_images(split["test_image_ids"][:4])
        x_m, x_p = dkl_frontend.magno_parvo_inputs(rgb, ch_min, ch_max)
        return f"magno {tuple(x_m.shape)}, parvo {tuple(x_p.shape)}"

    def _features():
        ch_min, ch_max = dkl_frontend.compute_dkl_channel_minmax()
        rgb = dkl_frontend.load_nsd_images(split["test_image_ids"][:4])
        x_m, x_p = dkl_frontend.magno_parvo_inputs(rgb, ch_min, ch_max)
        model = features.build_model(checkpoint=None, seed=0)
        f = features.extract_stream_features(model, x_m, x_p, batch_size=2)
        return {k: v.shape for k, v in f.items()}

    def _mini_fit():
        """Small end-to-end fit on REAL features and REAL voxels.

        The synthetic solver test uses well-conditioned Gaussian features and
        therefore cannot reproduce the rank deficiency of a heavily blurred
        magno band, which made torch's SVD fail to converge on the first real
        run. Fitting a few hundred real images against a few hundred real
        voxels reproduces that regime in ~a minute.
        """
        from src.analysis.encoding import banded_ridge

        # Sessions to load, and how many components to request. The requested
        # component count must be high enough to expose a rank-deficient band
        # (at 120 components both bands were trivially full rank and the
        # production failure at 1000 was invisible).
        sessions = [1, 2, 3, 4]
        n_components, n_vox = 1000, 300
        n_trials_loaded = len(sessions) * nsd_data.TRIALS_PER_SESSION

        # Choose images by whether their trials actually fall in the loaded
        # sessions, rather than taking the first N image ids and hoping.
        trial_ids = split["trial_image_ids"]
        train_trials = split["train_trials"]
        train_trials = train_trials[train_trials < n_trials_loaded]
        train_ids, first_row = np.unique(
            trial_ids[train_trials], return_index=True
        )
        rows_tr_all = train_trials[first_row]

        test_rows_all = split["test_trials_per_image"]
        test_ok = test_rows_all[:, 0] < n_trials_loaded
        test_ids = split["test_image_ids"][test_ok]
        rows_te_all = test_rows_all[test_ok, 0]

        if len(train_ids) < 200 or len(test_ids) < 20:
            raise RuntimeError(
                f"not enough trials in sessions {sessions}: "
                f"{len(train_ids)} train / {len(test_ids)} test images"
            )

        ch_min, ch_max = dkl_frontend.compute_dkl_channel_minmax()
        model = features.build_model(checkpoint=None, seed=0)

        def feats_for(ids):
            rgb = dkl_frontend.load_nsd_images(ids)
            x_m, x_p = dkl_frontend.magno_parvo_inputs(rgb, ch_min, ch_max)
            return features.extract_stream_features(
                model, x_m, x_p, batch_size=16
            )

        print(f"    mini_fit: {len(train_ids)} train / {len(test_ids)} test "
              f"images from sessions {sessions}", flush=True)
        f_tr, f_te = feats_for(train_ids), feats_for(test_ids)

        ranks, Xtr, Xte = {}, [], []
        for band in ("magno", "parvo"):
            pca = features.fit_pca(
                f_tr[band], n_components=n_components, seed=0
            )
            # Effective rank: components carrying non-negligible variance.
            ranks[band] = {
                "requested": int(pca.n_components_),
                "effective": int(
                    np.sum(pca.explained_variance_ratio_ > 1e-8)
                ),
                "var_explained": round(
                    float(pca.explained_variance_ratio_.sum()), 4
                ),
            }
            Xtr.append(pca.transform(f_tr[band]))
            Xte.append(pca.transform(f_te[band]))

        betas = nsd_data.load_betas(s, mask, sessions=sessions)
        vox = np.argsort(-nsd_data.load_noise_ceiling(s, mask))[:n_vox]
        Y_tr = betas[rows_tr_all][:, vox]
        Y_te = betas[rows_te_all][:, vox]
        del betas

        splits_small = banded_ridge.make_grouped_splits(
            train_ids, n_splits=3
        )
        res = banded_ridge.fit_banded(
            Xtr, Y_tr.astype(np.float32), Xte, Y_te.astype(np.float32),
            splits_small, backend="torch", n_iter=10,
        )
        return {
            "pca_ranks": ranks,
            "r2_joint_median": round(float(np.nanmedian(res["r2_joint"])), 4),
            "r2_magno_median": round(float(np.nanmedian(res["r2_magno"])), 4),
            "r2_parvo_median": round(float(np.nanmedian(res["r2_parvo"])), 4),
        }

    check("load_stim_info", _stim)
    check("available_sessions", _sessions)
    if stim_info is not None:
        check("build_train_test_split", _split)
    check("load_mask", _mask)
    if mask is not None:
        check("load_noise_ceiling", _nc)
        check("load_roi_labels", _rois)
        check("load_betas(1 session)", _betas)
    if split is not None:
        check("load_nsd_images", _images)
        check("dkl_frontend", _frontend)
        check("extract_stream_features", _features)
    if split is not None and mask is not None:
        check("mini_fit(real features + real voxels)", _mini_fit)

    print()
    if failures:
        print(f"PREFLIGHT FAILED: {failures}", flush=True)
        return 1
    if skipped:
        print(f"PREFLIGHT INCOMPLETE (skipped: {skipped}) — do not treat as "
              f"validation", flush=True)
        return 2
    print("PREFLIGHT PASSED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
