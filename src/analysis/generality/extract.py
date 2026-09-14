"""Dump a frozen network's penultimate features for one stimulus set.

  python -m src.analysis.generality.extract --model run:rgb_tsn_seed0 --set cue_conflict --out-dir DIR
  python -m src.analysis.generality.extract --model run:dkl_bcs_single34_seed0 --set miniecoset --out-dir DIR
  python -m src.analysis.generality.extract --model tv:resnet50 --set nsd --out-dir DIR

Writes DIR/<tag>_<set>.npz with
  features   (N, D) float16          penultimate units, the input of the last linear layer
  W, b       (C, D), (C,) float32    the network's own head
  cue_conflict: shapes, textures, stems, logits (N, C) float32 at the trained readout, and the
                trained shape bias by the model's scorer (sanity check against results/cue_conflict*)
  miniecoset:   train_features / train_y / val_features / val_y  (the full class-sorted h5 splits)
  nsd:          image_ids (0-based NSD ids, the union over subjects 1-8 of train and test images)
"""
from __future__ import annotations

import argparse
import json
import os
import time

import h5py
import numpy as np

from src.analysis.generality.models import FrozenNet
from src.analysis.generality.scoring import Scorer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="run:<name>[:ckpt] | concat:<name>[:ckpt] | tv:<arch>")
    p.add_argument("--set", required=True, choices=("cue_conflict", "miniecoset", "nsd"))
    p.add_argument("--out-dir", required=True)
    p.add_argument("--subjects", default="1,2,3,4,5,6,7,8")
    p.add_argument("--limit", type=int, default=None, help="smoke test: first N images per split")
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    import torch
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "8"))))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = FrozenNet(args.model, device=device)
    os.makedirs(args.out_dir, exist_ok=True)
    dest = os.path.join(args.out_dir, f"{net.tag}_{args.set}.npz")
    out = dict(tag=net.tag, spec=args.model, checkpoint=net.checkpoint, scorer=net.scorer,
               W=net.W.astype(np.float32), b=net.b.astype(np.float32))
    if net.stream_of_unit is not None:
        out["stream_of_unit"] = net.stream_of_unit
    print(f"[extract] {net.tag} set={args.set} D={net.n_units} C={net.W.shape[0]} device={device}", flush=True)

    if args.set == "cue_conflict":
        from src.analysis.behavior.cue_conflict import list_cue_conflict_images, load_images
        items = list_cue_conflict_images()
        if args.limit:
            items = items[:args.limit]
        paths = [i[0] for i in items]
        feats = []
        for s in range(0, len(paths), 128):
            feats.append(net.features_from_rgb(load_images(paths[s:s + 128])))
        y = np.concatenate(feats)
        logits = y @ net.W.T + net.b
        shapes = [i[1] for i in items]
        textures = [i[2] for i in items]
        sc = Scorer(net.scorer).score(logits, shapes, textures)
        print(f"[extract] trained readout: shape bias {sc['shape_bias']:.3f} "
              f"(forced {sc['forced_two_way']:.3f}, decided {sc['n_shape'] + sc['n_texture']}/{sc['n_images']})", flush=True)
        out.update(features=y.astype(np.float16), logits=logits.astype(np.float32),
                   shapes=np.array(shapes), textures=np.array(textures),
                   stems=np.array([os.path.splitext(os.path.basename(q))[0] for q in paths]),
                   trained_score=json.dumps(sc))

    elif args.set == "miniecoset":
        h5 = net.cfg["dataset"]
        for split in ("train", "val"):
            with h5py.File(h5, "r") as f:
                d, lab = f[split]["data"], np.asarray(f[split]["labels"][:]).reshape(-1)
                n = d.shape[0] if args.limit is None else min(args.limit, d.shape[0])
                feats = []
                for i in range(0, n, 256):
                    feats.append(net.features_from_h5(d[i:i + 256]))
                    if (i // 256) % 100 == 0:
                        print(f"  {split} {i}/{n}  {time.time() - t0:.0f}s", flush=True)
            out[f"{split}_features"] = np.concatenate(feats).astype(np.float16)
            out[f"{split}_y"] = lab[:n].astype(np.int64)
            print(f"[extract] {split}: {out[f'{split}_features'].shape}", flush=True)

    else:  # nsd
        from src.analysis.encoding import dkl_frontend, nsd_data
        stim_info = nsd_data.load_stim_info()
        ids = []
        for s in [int(x) for x in args.subjects.split(",")]:
            sp = nsd_data.build_train_test_split(s, stim_info)
            ids.append(sp["train_image_ids"]); ids.append(sp["test_image_ids"])
        image_ids = np.unique(np.concatenate(ids))
        if args.limit:
            image_ids = image_ids[:args.limit]
        print(f"[extract] {len(image_ids)} unique NSD images", flush=True)
        feats = []
        for s in range(0, len(image_ids), 1024):
            rgb = dkl_frontend.load_nsd_images(image_ids[s:s + 1024])
            feats.append(net.features_from_rgb(rgb))
            print(f"  nsd {min(s + 1024, len(image_ids))}/{len(image_ids)}  {time.time() - t0:.0f}s", flush=True)
        out.update(features=np.concatenate(feats).astype(np.float16), image_ids=image_ids)

    tmp = dest[:-4] + f".tmp{os.getpid()}.npz"  # np.savez appends .npz to any other suffix
    np.savez(tmp, **out)
    os.replace(tmp, dest)
    print(f"[extract] wrote {dest} ({os.path.getsize(dest) / 1e6:.0f} MB) in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
