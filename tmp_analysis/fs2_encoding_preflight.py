"""Preflight for encoding the Stage-2 fusion TSNs on NSD (2026-10-16).

The NSD encoding (run_encoding.py) and the readout-weight fit
(readout_w_cortex.py) build their own front end from RGB. For the fs2 runs that
front end must be the one the trainer scored, and for the DVD-diet arms no
blur/CS flag combination expresses it. On the first N MiniEcoset val images,
for each run, this checks:

  1. dkl_frontend.rgb_to_dkl3(RGB source) == the DKL uint8 h5 the runs trained on
  2. magno_parvo_inputs with the flags the Slurm arrays pass
     == StreamInputBuilder(the checkpoint's own config), bit for bit
  3. features.build_model(checkpoint_final.pt) trunks + the run's tied head
     (fp32, CPU) against the trainer's final_logits.npz (bf16, GPU):
     argmax agreement >= 0.95
  4. run_encoding.check_frontend_matches_checkpoint accepts those flags and
     refuses the other diet's
  5. run_encoding.features_from_rgb and readout_w_cortex.stream_outputs run
     with those flags, and stream_outputs equals the pooled trunk output of 3

Exits 1 on any failure, so the encoding arrays can depend on it (afterok).
"""
from __future__ import annotations

import copy
import json
import sys
import types

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from src.analysis.encoding import dkl_frontend, features, readout_w_cortex, run_encoding
from src.training.standalone_trainer import StreamInputBuilder

N = 64
RGB_H5 = "/scratch/byrong/VST_data/VST_datasets/texture2shape_miniecoset_256px.h5"
DKL_H5 = "/scratch/byrong/VST_data/VST_datasets/miniecoset_256px_dkl_uint8.h5"
C = "/scratch/byrong/TSN/fusion_stage1/checkpoints/2026-10-12_stage2"
RUNS = ([f"fs2_dvd300_avg_w050_ls_seed{s}" for s in range(3)]
        + ["fs2_dvdfix300_avg_w050_ls_seed0"]
        + [f"fs2_bcs_avg_w050_ls_seed{s}" for s in range(3)])
fails: list[str] = []


def check(ok: bool, msg: str) -> None:
    print(("PASS " if ok else "FAIL ") + msg, flush=True)
    if not ok:
        fails.append(msg)


def slurm_flags(run: str) -> types.SimpleNamespace:
    """The arguments slurm/encode_fs2.sh and slurm/readout_w_fs2.sh pass."""
    dvd = 300.0 if "_dvd" in run else None
    return types.SimpleNamespace(
        checkpoint=f"{C}/{run}/checkpoint_final.pt", dvd_age_months=dvd,
        no_lum_blur=dvd is not None, no_lum_cs=dvd is not None, age_months=48,
        transform_target="magno", parvo_highsf=False, chroma_channel="rg",
        chroma_rescale="none", features="network", batch_size=32)


def pooled(block, x):
    return F.adaptive_avg_pool2d(block(x), 1).flatten(1)


@torch.no_grad()
def main() -> None:
    torch.set_num_threads(4)
    with h5py.File(RGB_H5, "r") as f:
        rgb, lab_rgb = f["val"]["data"][:N], np.asarray(f["val"]["labels"][:N]).reshape(-1)
    with h5py.File(DKL_H5, "r") as f:
        dkl, lab = f["val"]["data"][:N], np.asarray(f["val"]["labels"][:N]).reshape(-1)
    ch_min, ch_max = dkl_frontend.compute_dkl_channel_minmax()

    x3 = dkl_frontend.rgb_to_dkl3(rgb, ch_min, ch_max)
    ours = (x3 * 255.0).round().to(torch.uint8).permute(0, 2, 3, 1).numpy()
    d = np.abs(ours.astype(int) - dkl.astype(int))
    check(d.max() <= 1 and (d > 0).mean() < 1e-3,
          f"1 rgb_to_dkl3 vs training h5: max|d| {d.max()} (uint8 steps), "
          f"fraction differing {(d > 0).mean():.2e}")
    check(np.array_equal(lab_rgb, lab), "1 RGB and DKL h5 val label order agree")
    x_train = torch.from_numpy(dkl).permute(0, 3, 1, 2).float() / 255.0

    for run in RUNS:
        a = slurm_flags(run)
        ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
        cfg = ck["config"]
        print(f"--- {run}: weights={ck.get('weights')} "
              f"dvd_end_months={cfg.get('dvd_end_months')} blur={cfg['blur']} "
              f"cs={cfg['cs']} w={cfg['w']}", flush=True)
        age = json.load(open(f"{C}/{run}/config.json"))["age_months"]
        builder = StreamInputBuilder(cfg, age_months=age)

        xb_m, xb_p = builder(x3[:, :2].clone())
        xe_m, xe_p = dkl_frontend.magno_parvo_inputs(
            rgb, ch_min, ch_max, lum_blur=not a.no_lum_blur,
            lum_cs=not a.no_lum_cs, age_months=a.age_months,
            dvd_age=a.dvd_age_months)
        check(torch.equal(xb_m, xe_m) and torch.equal(xb_p, xe_p),
              f"2 {run} encoding front end == StreamInputBuilder "
              f"(max|d| magno {(xb_m - xe_m).abs().max():.2e})")

        model = features.build_model(checkpoint=a.checkpoint)
        lin = model.classifier[-1]
        xt_m, xt_p = builder(x_train)
        h_m, h_p = pooled(model.stream1_block1, xt_m), pooled(model.stream2_block1, xt_p)
        z = lin(cfg["w"] * h_m + (1 - cfg["w"]) * h_p).numpy()
        zl = np.load(f"{C}/{run}/final_logits.npz")
        ref = zl["val_joint"][:N]
        agree = (z.argmax(1) == ref.argmax(1)).mean()
        check(agree >= 0.95,
              f"3 {run} argmax agreement with trainer logits {agree:.3f}, "
              f"max|dz| {np.abs(z - ref).max():.3f} (logit sd {ref.std():.2f}); "
              f"top-1 here {(z.argmax(1) == lab).mean():.3f}, "
              f"trainer {(ref.argmax(1) == lab).mean():.3f}")

        run_encoding.check_frontend_matches_checkpoint(a)
        wrong = copy.copy(a)
        if a.dvd_age_months is None:
            wrong.dvd_age_months, wrong.no_lum_blur, wrong.no_lum_cs = 300.0, True, True
        else:
            wrong.dvd_age_months, wrong.no_lum_blur, wrong.no_lum_cs = None, False, False
        try:
            run_encoding.check_frontend_matches_checkpoint(wrong)
            check(False, f"4 {run} guard accepted the wrong front end")
        except ValueError:
            check(True, f"4 {run} guard accepts the run's flags, refuses the other diet's")

        fm, fp = run_encoding.features_from_rgb(a, rgb[:8], ch_min, ch_max, model, "cpu")
        y1, y2 = readout_w_cortex.stream_outputs(model, rgb[:8], ch_min, ch_max, a, "cpu")
        e1, e2 = pooled(model.stream1_block1, xe_m[:8]), pooled(model.stream2_block1, xe_p[:8])
        check(fm.shape == (8, 15360) and fp.shape == (8, 15360)
              and np.allclose(y1, e1.numpy(), atol=1e-5)
              and np.allclose(y2, e2.numpy(), atol=1e-5),
              f"5 {run} features_from_rgb {fm.shape}, stream_outputs == pooled trunks")

    print(f"\n{len(fails)} failure(s)", flush=True)
    for m in fails:
        print("  " + m)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
