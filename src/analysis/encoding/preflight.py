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

    def check(name, fn):
        try:
            out = fn()
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

    print()
    if failures:
        print(f"PREFLIGHT FAILED: {failures}", flush=True)
        return 1
    print("PREFLIGHT PASSED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
