"""Do the input channels actually induce shape- vs texture-biased streams?

The project's title claim is that biologically constrained magno/parvo inputs
induce specialised shape-/texture-biased streams. That claim has so far been
supported only indirectly, through the classifier's behaviour. This measures
it directly, in the representations themselves.

Method. Geirhos cue-conflict images carry a shape label and a texture label
that are orthogonal by construction (image `<shape>i-<texture>j.png`, i != j).
A feature space therefore cannot score well on both by trivially encoding
"which image this is": shape and texture are separately recoverable only to
the extent that the features represent them separately. For each stream we
fit a cross-validated linear probe for the shape label and another for the
texture label, and report

    bias = acc(shape) - acc(texture)

per stream, plus the between-stream contrast that the claim predicts:

    bias(magno) - bias(parvo)  >  0

Why a probe rather than the classifier head: the head is trained on object
class and mixes the streams, so it cannot say what each stream *represents*;
and the probe works on an untrained network, which is the control the rest of
this project is built on.

Both streams are reduced to the same number of PCA components before probing,
so a difference cannot come from one stream having more usable dimensions --
the general-factor confound that defeated the thesis's RSA.

Usage:
  python -m src.analysis.behavior.stream_shape_texture \
      --model-name untrained_seed0 [--checkpoint ...] \
      [--no-lum-blur] [--no-lum-cs] [--out results/shape_texture]
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.analysis.behavior.cue_conflict import (
    list_cue_conflict_images,
    load_images,
)
from src.analysis.encoding import dkl_frontend, features


def probe(x: np.ndarray, y: list[str], n_splits: int = 5,
          seed: int = 0) -> tuple[float, np.ndarray]:
    """Cross-validated linear-probe accuracy; also returns per-fold scores."""
    y = np.asarray(y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for tr, te in skf.split(x, y):
        sc = StandardScaler().fit(x[tr])
        clf = LogisticRegression(max_iter=2000, C=1.0, n_jobs=1)
        clf.fit(sc.transform(x[tr]), y[tr])
        scores.append(clf.score(sc.transform(x[te]), y[te]))
    return float(np.mean(scores)), np.asarray(scores)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--no-lum-blur", action="store_true")
    p.add_argument("--no-lum-cs", action="store_true")
    p.add_argument("--transform-target", default="magno",
                   choices=("magno", "parvo"))
    p.add_argument("--stream1-weight", type=float, default=0.5)
    p.add_argument("--parvo-highsf", action="store_true",
                   help="biologically corrected parvo input: stream 2 gets "
                        "untransformed luminance + rg (2 channels). Sets "
                        "in_channels=(1,2); required for the "
                        "dkl_bcs_parvo_highsf checkpoints and wrong for "
                        "every other condition.")
    p.add_argument("--pca-dim", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--out", default="results/shape_texture")
    p.add_argument("--allow-partial", action="store_true",
                   help="probe a still-training checkpoint on purpose")
    args = p.parse_args()

    # Before any compute: a mid-training checkpoint produces output that is
    # indistinguishable from a converged one (PAPER_PLAN §10.1).
    if args.checkpoint:
        features.assert_converged(args.checkpoint, args.allow_partial)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    items = list_cue_conflict_images()
    paths = [a for a, _, _ in items]
    shapes = [b for _, b, _ in items]
    textures = [c for _, _, c in items]
    print(f"{len(items)} cue-conflict images | "
          f"{len(set(shapes))} shape classes, {len(set(textures))} texture "
          f"classes | device={device}", flush=True)

    rgb = load_images(paths)
    ch_min, ch_max = dkl_frontend.compute_dkl_channel_minmax()
    x_m, x_p = dkl_frontend.magno_parvo_inputs(
        rgb, ch_min, ch_max,
        lum_blur=not args.no_lum_blur, lum_cs=not args.no_lum_cs,
        transform_target=args.transform_target, device="cpu",
        parvo_highsf=args.parvo_highsf,
    )

    model = features.build_model(
        checkpoint=args.checkpoint, stream1_weight=args.stream1_weight,
        seed=args.seed, device=device,
        in_channels=(1, 2) if args.parvo_highsf else (1, 1),
    )
    model.eval()
    feats = features.extract_stream_features(
        model, x_m, x_p, batch_size=args.batch_size, device=device,
    )

    # Chance is 1/n_classes for both probes because both label sets use the
    # same 16 Geirhos categories, so the two accuracies are comparable.
    chance = 1.0 / len(set(shapes))
    out: dict = {"model_name": args.model_name, "n_images": len(items),
                 "chance": chance, "pca_dim": args.pca_dim,
                 "checkpoint": args.checkpoint, "streams": {}}

    for stream in ("magno", "parvo"):
        f = feats[stream]
        k = min(args.pca_dim, f.shape[0] - 1, f.shape[1])
        z = PCA(n_components=k, random_state=args.seed).fit_transform(f)
        acc_s, fold_s = probe(z, shapes, seed=args.seed)
        acc_t, fold_t = probe(z, textures, seed=args.seed)
        out["streams"][stream] = {
            "shape_acc": acc_s, "texture_acc": acc_t,
            "bias": acc_s - acc_t,
            "shape_folds": fold_s.tolist(),
            "texture_folds": fold_t.tolist(),
            "n_components": int(k),
        }
        print(f"  {stream:<6} shape {acc_s:.4f}  texture {acc_t:.4f}  "
              f"bias {acc_s - acc_t:+.4f}  (chance {chance:.4f}, k={k})",
              flush=True)

    m, pv = out["streams"]["magno"], out["streams"]["parvo"]
    out["bias_magno_minus_parvo"] = m["bias"] - pv["bias"]
    # Paired across folds: the same CV partition is used for both streams,
    # so folds are matched and the difference is a within-fold contrast.
    d = ((np.array(m["shape_folds"]) - np.array(m["texture_folds"]))
         - (np.array(pv["shape_folds"]) - np.array(pv["texture_folds"])))
    out["fold_diffs"] = d.tolist()
    print(f"\n  bias(magno) - bias(parvo) = "
          f"{out['bias_magno_minus_parvo']:+.4f}   "
          f"per-fold {np.round(d, 4).tolist()}")
    print("  (positive = magno is relatively more shape-biased, "
          "the predicted direction)")

    os.makedirs(args.out, exist_ok=True)
    dest = os.path.join(args.out, f"{args.model_name}.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
