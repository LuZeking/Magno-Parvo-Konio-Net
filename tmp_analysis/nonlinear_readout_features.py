"""Dump pooled per-stream penultimate features for the §104 nonlinear readout test.

§2.3's w-sweep works offline from stored *logits* because the head is affine:
``clf(w*F1 + (1-w)*F2) = w*clf(F1) + (1-w)*clf(F2)``. A NONLINEAR head breaks
that identity by construction, so the logits are useless for §104 and the
pooled features have to be extracted and stored.

What is stored, per run: the 512-d pooled pre-classifier feature of each stream
for the MiniEcoset train and val splits (to fit and select the head) and for the
1,200 Geirhos cue-conflict images (to score shape bias). Nothing else changes:
the front end is rebuilt from the run's own ``config.json`` rather than passed
on the command line, so a mismatch between the extraction and the training
condition is not expressible.

Usage: PYTHONPATH=.:tmp_analysis python tmp_analysis/nonlinear_readout_features.py \
           --run-dir .../dkl_bcs_seed0 --checkpoint-name checkpoint_125.pt \
           --tag dkl_bcs_seed0_best --out-dir ...
--run-dir also takes a Stage-2 run name (fs2_dvd300_avg_w050_ls_seed0, with
--checkpoint-name checkpoint_final.pt); the cue-conflict front end is
dkl_frontend.checkpoint_frontend of the run's config (the DVD diet for those).
"""
from __future__ import annotations

import argparse, json, os, time

import h5py
import numpy as np
import torch
import torch.nn as nn

from src.analysis.encoding import dkl_frontend, features
from src.analysis.behavior.cue_conflict import (
    GEIRHOS_CLASSES, list_cue_conflict_images, load_class_index, load_images,
)
from src.training.standalone_trainer import StreamInputBuilder
from fusion_sweep import stream_features

POOL = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())


def pooled(fmap: torch.Tensor) -> torch.Tensor:
    """(B,512,H,W) -> (B,512), the exact tensor the trained Linear head sees."""
    return POOL(fmap)


def extract_h5(model, builder, h5_path, split, device, batch=256, limit=None):
    f1, f2, ys = [], [], []
    with h5py.File(h5_path, "r") as f:
        d, lab = f[split]["data"], np.asarray(f[split]["labels"][:]).reshape(-1)
        n = d.shape[0] if limit is None else min(limit, d.shape[0])
        t0 = time.time()
        for i in range(0, n, batch):
            arr = d[i:i + batch]                       # (B,H,W,C) uint8
            x = torch.from_numpy(np.ascontiguousarray(arr)).permute(0, 3, 1, 2)
            x = x.to(device).float() / 255.0           # identical to the val path
            x1, x2 = builder(x)
            with torch.no_grad():
                a = model.stream1_block1(x1)
                b = model.stream2_block1(x2)
                if model.stream_concat:
                    a, b = model._mix_streams(a, b)
                    a, b = model.stream1_block2(a), model.stream2_block2(b)
                f1.append(pooled(a.float()).cpu().half())
                f2.append(pooled(b.float()).cpu().half())
            ys.append(torch.from_numpy(lab[i:i + batch].copy()))
            if i % (batch * 100) == 0:
                print(f"  {split} {i}/{n}  {time.time()-t0:.0f}s", flush=True)
    return torch.cat(f1), torch.cat(f2), torch.cat(ys)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--checkpoint-name", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--train-limit", type=int, default=None)
    args = p.parse_args()

    args.run_dir = dkl_frontend.resolve_run_dir(args.run_dir)  # or an fs2 run name
    cfg = json.load(open(os.path.join(args.run_dir, "config.json")))
    ccfg = cfg["condition_cfg"]
    # The g017 runs name a node-local copy of the training h5.
    cfg["dataset"] = dkl_frontend.run_dataset(cfg["dataset"])
    ckpt = os.path.join(args.run_dir, args.checkpoint_name)
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{args.tag}: condition={cfg['condition']} cfg={ccfg} ckpt={ckpt}",
          flush=True)

    model = features.build_model(
        # Stage-2 configs record like_resnet null and the arch as _resolved
        checkpoint=ckpt, like_resnet=(cfg.get("like_resnet")
                                      or cfg.get("like_resnet_resolved") or "resnet18"),
        stream_concat=False, stream1_weight=0.5, seed=cfg.get("seed", 0),
        device=device, in_channels=tuple(cfg["in_channels"]))
    model.eval()
    builder = StreamInputBuilder(ccfg, age_months=cfg.get("age_months", 48))

    out = {}
    for split in ("train", "val"):
        a, b, y = extract_h5(model, builder, cfg["dataset"], split, device,
                             limit=args.train_limit if split == "train" else None)
        out[f"{split}_F1"], out[f"{split}_F2"], out[f"{split}_y"] = a, b, y
        print(f"  {split}: {tuple(a.shape)}", flush=True)

    # cue-conflict comes through the RGB->DKL path, normalised with the same
    # channel min/max as the training set (dkl_frontend docstring), so the two
    # feature sets are on one scale.
    items = list_cue_conflict_images()
    paths = [i[0] for i in items]
    ch_min, ch_max = dkl_frontend.compute_dkl_channel_minmax()
    cf1, cf2 = [], []
    for s in range(0, len(paths), 128):
        rgb = load_images(paths[s:s + 128])
        # The builder's front end as magno_parvo_inputs kwargs: a DVD-diet run
        # keeps blur/cs = true beside dvd_end_months in its config.
        x_m, x_p = dkl_frontend.magno_parvo_inputs(
            rgb, ch_min, ch_max, device=device,
            **dkl_frontend.checkpoint_frontend(cfg))
        a, b = stream_features(model, x_m, x_p, device=device)
        cf1.append(pooled(a.float()).cpu()); cf2.append(pooled(b.float()).cpu())
    cls = load_class_index()
    out.update(
        cc_F1=torch.cat(cf1), cc_F2=torch.cat(cf2),
        cc_stems=np.array([os.path.splitext(os.path.basename(q))[0] for q in paths]),
        cc_shapes=np.array([i[1] for i in items]),
        cc_textures=np.array([i[2] for i in items]),
        geirhos_cols=np.array([cls[c] for c in GEIRHOS_CLASSES]),
        geirhos_classes=np.array(GEIRHOS_CLASSES),
        tag=args.tag, checkpoint=ckpt, condition=cfg["condition"])
    print(f"  cue-conflict: {tuple(out['cc_F1'].shape)}", flush=True)

    dest = os.path.join(args.out_dir, f"{args.tag}_features.npz")
    np.savez(dest, **{k: (v.numpy() if torch.is_tensor(v) else v)
                      for k, v in out.items()})
    print(f"saved -> {dest} ({os.path.getsize(dest)/1e9:.2f} GB)", flush=True)


if __name__ == "__main__":
    main()
