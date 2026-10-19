"""Stream features of the two-stream network, stored, and the exemplar-held-out probes on them
(docs/PLAN_EXEMPLAR_2026-09-21.md, array B).

Extraction is `stream_shape_texture.py`'s, step for step and with its defaults (same front end, same model
builder, same seed for untrained networks, same stage-pooled features), so the paper's probe (Q0) on the
stored float32 features must reproduce `results/shape_texture/<model-name>.json` exactly; the task fails if
it does not. Then Q1 (exemplar-balanced, PCA inside the folds), Q1b and Q2 per stream.

  python -m src.analysis.behavior.stream_features --model-name dkl_bcs_seed0_trained \
      --checkpoint RUNS/dkl_bcs_seed0/checkpoint_last.pt --feat-dir FEAT --out-dir OUT [front-end flags]
  --checkpoint also takes a Stage-2 run name (fs2_dvd300_avg_w050_ls_seed0); its DVD-diet arms need
  --dvd-age-months 300, and dkl_frontend.check_frontend refuses any other front end for them.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from sklearn.decomposition import PCA

from src.analysis.behavior.cue_conflict import list_cue_conflict_images, load_images, stream_in_channels
from src.analysis.behavior.stream_shape_texture import probe
from src.analysis.encoding import dkl_frontend, features
from src.analysis.generality.exemplar import all_probes, parse_exemplars


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--no-lum-blur", action="store_true")
    p.add_argument("--no-lum-cs", action="store_true")
    p.add_argument("--transform-target", default="magno", choices=("magno", "parvo"))
    p.add_argument("--stream1-weight", type=float, default=0.5)
    p.add_argument("--parvo-highsf", action="store_true")
    p.add_argument("--rgb-mode", action="store_true")
    p.add_argument("--dvd-age-months", type=float, default=None,
                   help="DVD magno diet at this age (fs2_dvd* runs: 300) instead of blur + CS; "
                        "also the untrained anchor on that front end")
    p.add_argument("--pca-dim", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--reference-dir", default="results/shape_texture")
    p.add_argument("--feat-dir", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    fe = dkl_frontend.frontend_from_args(args)  # resolves an fs2 run name; refuses a wrong diet
    t0 = time.time()
    os.makedirs(args.feat_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    feat_path = os.path.join(args.feat_dir, f"{args.model_name}_streams.npz")
    items = list_cue_conflict_images()
    paths = [a for a, _, _ in items]
    shapes = np.array([b for _, b, _ in items]); textures = np.array([c for _, _, c in items])
    stems = np.array([os.path.splitext(os.path.basename(a))[0] for a in paths])
    if os.path.exists(feat_path):
        z = np.load(feat_path, allow_pickle=True)
        assert list(z["stems"]) == list(stems)
        feats = {"magno": z["magno"], "parvo": z["parvo"]}
        print(f"[stream_features] {args.model_name}: reusing {feat_path}", flush=True)
    else:
        if args.checkpoint:
            features.assert_converged(args.checkpoint, False)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        rgb = load_images(paths)
        ch_min, ch_max = dkl_frontend.compute_dkl_channel_minmax()
        x_m, x_p = dkl_frontend.magno_parvo_inputs(rgb, ch_min, ch_max, device="cpu", **fe)
        model = features.build_model(checkpoint=args.checkpoint, stream1_weight=args.stream1_weight,
                                     seed=args.seed, device=device, in_channels=stream_in_channels(args))
        model.eval()
        feats = features.extract_stream_features(model, x_m, x_p, batch_size=args.batch_size, device=device)
        np.savez(feat_path, magno=feats["magno"], parvo=feats["parvo"], shapes=shapes, textures=textures,
                 stems=stems, model_name=args.model_name, checkpoint=str(args.checkpoint))
        print(f"[stream_features] {args.model_name}: extracted {feats['magno'].shape} per stream "
              f"({time.time() - t0:.0f}s) -> {feat_path}", flush=True)

    obj, tex = parse_exemplars(stems)
    ref_path = os.path.join(args.reference_dir, f"{args.model_name}.json")
    ref = json.load(open(ref_path)) if os.path.exists(ref_path) else None
    out = dict(model_name=args.model_name, checkpoint=args.checkpoint, n_images=len(stems),
               plan="docs/PLAN_EXEMPLAR_2026-09-21.md", streams={})
    for stream in ("magno", "parvo"):
        f = feats[stream]
        # Q0 exactly as stream_shape_texture.main ran it: PCA on the float32 features
        k = min(args.pca_dim, f.shape[0] - 1, f.shape[1])
        zc = PCA(n_components=k, random_state=args.seed).fit_transform(f)
        a_s, _ = probe(zc, list(shapes), seed=args.seed)
        a_t, _ = probe(zc, list(textures), seed=args.seed)
        q0 = dict(shape_acc=float(a_s), texture_acc=float(a_t), bias=float(a_s - a_t))
        if ref is not None:
            want = ref["streams"][stream]["bias"]
            d = abs(q0["bias"] - want)
            print(f"  {stream}: Q0 {q0['bias']:+.4f} vs stored {want:+.4f} (|d| = {d:.1e})", flush=True)
            if d > 1e-9:
                raise SystemExit(f"Q0 does not reproduce {ref_path} for {stream}: |d| = {d}")
            q0["stored_bias"] = want
        pr = all_probes(f.astype(np.float64), shapes, textures, obj, tex)
        pr["Q0"] = q0
        out["streams"][stream] = pr
        print(f"  {stream}: Q1 {pr['Q1']['bias']:+.3f} [{pr['Q1']['bias_p05']:+.3f}, {pr['Q1']['bias_p95']:+.3f}] "
              f"(shape {pr['Q1']['shape_acc']:.3f} tex {pr['Q1']['texture_acc']:.3f}) | Q1b {pr['Q1b']['bias']:+.3f} "
              f"| Q2 {pr['Q2']['bias']:+.3f}  {time.time() - t0:.0f}s", flush=True)
    dest = os.path.join(args.out_dir, f"{args.model_name}_stream_exemplar.json")
    json.dump(out, open(dest, "w"), indent=1)
    print(f"[stream_features] wrote {dest} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
