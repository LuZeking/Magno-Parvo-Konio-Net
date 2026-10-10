"""Cue-conflict cache for the fusion Stage-1 trainer (per-epoch shape bias).

Writes the 1,200 Geirhos cue-conflict images (same-cue images excluded) as
(N, 256, 256, 3) uint8 DKL, the exact format the training h5 stores
(dkl_frontend.rgb_to_dkl3: resize, DKL, training min/max, uint8), with the
shape / texture labels and the MiniEcoset columns of the 16 Geirhos classes.

Parity check before saving: the trainer's StreamInputBuilder applied to the
cached tensor must reproduce dkl_frontend.magno_parvo_inputs -- the front end
of src/analysis/behavior/cue_conflict.py -- for the dieted and parvo_highsf
conditions, so per-epoch shape bias is scored on the inputs the published
cue-conflict numbers used.

    PYTHONPATH=. python scripts/fusion_stage1_cue_conflict_cache.py --out <path.npz>
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from src.analysis.behavior.cue_conflict import (GEIRHOS_CLASSES,
                                                list_cue_conflict_images,
                                                load_class_index, load_images)
from src.analysis.encoding import dkl_frontend
from src.training.standalone_trainer import CONDITIONS, StreamInputBuilder


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    args = p.parse_args()

    items = list_cue_conflict_images()
    ch_min, ch_max = dkl_frontend.compute_dkl_channel_minmax()
    xs = []
    for s in range(0, len(items), 128):
        rgb = load_images([i[0] for i in items[s:s + 128]])
        x = dkl_frontend.rgb_to_dkl3(rgb, ch_min, ch_max)  # u8 / 255
        xs.append((x * 255.0).round().to(torch.uint8)
                  .permute(0, 2, 3, 1).numpy())
        print(f"  {min(s + 128, len(items))}/{len(items)}", flush=True)
    x = np.concatenate(xs)

    rgb = load_images([i[0] for i in items[:8]])
    x8 = torch.from_numpy(x[:8]).permute(0, 3, 1, 2).float() / 255.0
    for cond, kw in (("dkl_bcs", {}), ("dkl_bcs_parvo_highsf",
                                       {"parvo_highsf": True})):
        ref_m, ref_p = dkl_frontend.magno_parvo_inputs(rgb, ch_min, ch_max,
                                                       **kw)
        got_m, got_p = StreamInputBuilder(CONDITIONS[cond])(x8)
        d = max(float((ref_m - got_m).abs().max()),
                float((ref_p - got_p).abs().max()))
        print(f"  parity {cond}: max |diff| = {d:.2e}", flush=True)
        if d > 1e-5:
            raise RuntimeError(f"{cond}: trainer input differs from the "
                               f"cue_conflict.py front end by {d}")

    ci = load_class_index()
    tmp = args.out + ".tmp.npz"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(tmp, x=x,
             shapes=np.array([i[1] for i in items]),
             textures=np.array([i[2] for i in items]),
             classes=np.array(GEIRHOS_CLASSES),
             cols=np.array([ci[c] for c in GEIRHOS_CLASSES]),
             paths=np.array([i[0] for i in items]))
    os.replace(tmp, args.out)
    print(f"saved {x.shape} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
