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

Instance repeats. The 1,200 images are 160 content photos x 48 texture
source images; each texture source appears in 15-35 images (median 25), each
content photo in 3-12 (median 8). The original split (`streams`, 5-fold
StratifiedKFold, no grouping) therefore tests on images whose texture source
and content photo were both in training, and a probe can score by
recognising the repeated instance rather than the category -- texture most,
because texture sources repeat ~3x more. `splits` adds two grouped splits:
`per_label` (shape probe holds out content photos, texture probe holds out
texture sources) and `double` (each test image's content photo AND texture
source are unseen). The grouped texture probe trains on 2 sources per class
against 8 photos for shape, which handicaps texture; read grouped biases
with that in mind.

Usage:
  python -m src.analysis.behavior.stream_shape_texture \
      --model-name untrained_seed0 [--checkpoint ...] \
      [--no-lum-blur] [--no-lum-cs] [--out results/shape_texture]
  python -m src.analysis.behavior.stream_shape_texture \
      --model-name fs1_bcs_avg_seed0 --stage1-run <run dir>
  python -m src.analysis.behavior.stream_shape_texture \
      --model-name fs2_dvd300_avg_w050_ls_seed0 \
      --checkpoint fs2_dvd300_avg_w050_ls_seed0 --dvd-age-months 300

--checkpoint takes a Stage-2 run name for its checkpoint_final.pt; a DVD-diet
run probed without --dvd-age-months is refused (dkl_frontend.check_frontend).
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
    stream_in_channels,
)
from src.analysis.encoding import dkl_frontend, features


def _fit_score(x: np.ndarray, y: np.ndarray, tr: np.ndarray,
               te: np.ndarray) -> float:
    sc = StandardScaler().fit(x[tr])
    clf = LogisticRegression(max_iter=2000, C=1.0, n_jobs=1)
    clf.fit(sc.transform(x[tr]), y[tr])
    return clf.score(sc.transform(x[te]), y[te])


def probe(x: np.ndarray, y: list[str], n_splits: int = 5, seed: int = 0,
          folds: np.ndarray | None = None) -> tuple[float, np.ndarray]:
    """Cross-validated linear-probe accuracy; also returns per-fold scores.

    ``folds=None`` is the original ungrouped split that Table 1 reports.
    Per-image fold ids (``folds_within_class``) hold whole groups out instead.
    The accuracy is pooled over images; with ungrouped folds of equal size
    (1,200 / 5) that equals the mean of the fold scores reported before.
    """
    y = np.asarray(y)
    if folds is None:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                              random_state=seed)
        splits = list(skf.split(x, y))
    else:
        splits = [(np.flatnonzero(folds != f), np.flatnonzero(folds == f))
                  for f in np.unique(folds)]
    scores = [_fit_score(x, y, tr, te) for tr, te in splits]
    n_te = [len(te) for _, te in splits]
    return float(np.dot(scores, n_te) / sum(n_te)), np.asarray(scores)


def probe_double(x: np.ndarray, y: list[str], content_fold: np.ndarray,
                 source_fold: np.ndarray) -> tuple[float, np.ndarray]:
    """Both groups held out: no test image shares its content photo or its
    texture source with any training image. 5 x 3 = 15 cells, every image
    tested once; each fit trains on (4/5)(2/3) of the images."""
    y = np.asarray(y)
    scores, n_te = [], []
    for i in np.unique(content_fold):
        for j in np.unique(source_fold):
            te = np.flatnonzero((content_fold == i) & (source_fold == j))
            tr = np.flatnonzero((content_fold != i) & (source_fold != j))
            scores.append(_fit_score(x, y, tr, te))
            n_te.append(len(te))
    return float(np.dot(scores, n_te) / sum(n_te)), np.asarray(scores)


def cue_conflict_groups(paths: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(content photo, texture source) of each image `<shape>i-<texture>j`."""
    stems = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    return (np.array([s.split("-")[0] for s in stems]),
            np.array([s.split("-")[1] for s in stems]))


def folds_within_class(groups: np.ndarray, labels: list[str],
                       k: int) -> np.ndarray:
    """Per-image fold id: each class's groups are dealt round-robin into k
    folds, so every fold holds every class and no group spans two folds.
    160 content photos = 10 per class -> k=5; 48 texture sources = 3 per
    class -> k=3 (one source per class per fold)."""
    by_class: dict[str, set] = {}
    for g, lab in zip(groups, labels):
        by_class.setdefault(lab, set()).add(g)
    fold = {}
    for lab, gs in by_class.items():
        if len(gs) < k:
            raise ValueError(f"class {lab} has {len(gs)} groups < {k} folds")
        for i, g in enumerate(sorted(gs)):
            fold[g] = i % k
    return np.array([fold[g] for g in groups])


def probe_all_splits(z: np.ndarray, shapes: list[str], textures: list[str],
                     content_fold: np.ndarray, source_fold: np.ndarray,
                     seed: int) -> dict:
    """Shape and texture accuracy of one feature space under the three
    splits."""
    out = {}
    for split in ("stratified", "per_label", "double"):
        if split == "stratified":
            acc_s, _ = probe(z, shapes, seed=seed)
            acc_t, _ = probe(z, textures, seed=seed)
        elif split == "per_label":
            acc_s, _ = probe(z, shapes, folds=content_fold)
            acc_t, _ = probe(z, textures, folds=source_fold)
        else:
            acc_s, _ = probe_double(z, shapes, content_fold, source_fold)
            acc_t, _ = probe_double(z, textures, content_fold, source_fold)
        out[split] = {"shape_acc": acc_s, "texture_acc": acc_t,
                      "bias": acc_s - acc_t}
    return out


def stage1_model_and_inputs(run_dir: str, cc_npz: str | None = None):
    """A Stage-1 late-fusion run (standalone_trainer `fs1_*` conditions).

    Rebuilds the network the way the trainer did, loads the EMA weights of
    `checkpoint_final.pt` (the network every logged metric scores) with
    strict key matching, and builds the inputs from the run's own
    cue-conflict cache through the trainer's StreamInputBuilder, so the
    diet doses and the 64-px magno stem are reproduced rather than
    approximated. Returns (trunk, fusion net, x_magno, x_parvo, paths,
    shapes, textures, logged final logits, MiniEcoset columns of the 16
    Geirhos classes).
    """
    from src.models.late_fusion import LateFusionNetwork, make_magno_stem64
    from src.models.pytorch_models import TwoStreamNetwork
    from src.training.standalone_trainer import (
        StreamInputBuilder,
        patch_stream_input_channels,
    )

    with open(os.path.join(run_dir, "config.json")) as fh:
        run_cfg = json.load(fh)
    ck = torch.load(os.path.join(run_dir, "checkpoint_final.pt"),
                    map_location="cpu", weights_only=False)
    if ck.get("weights") != "ema":
        raise RuntimeError(f"{run_dir}: checkpoint_final.pt is not EMA")
    cfg = ck["config"]
    c1, c2 = cfg["in_channels"]
    trunk = TwoStreamNetwork(
        num_classes=run_cfg["num_classes"],
        like_resnet=run_cfg["like_resnet_resolved"],
        seperate_from_layer=0, baseline=False, dvd_strategy="dkl",
        stream_concat=False, stream1_weight=cfg.get("stream1_weight", 0.5),
        legacy_asymmetric_mixing=False,
    )
    patch_stream_input_channels(trunk, c1, c2)
    if cfg.get("magno_stem64"):
        make_magno_stem64(trunk.stream1_block1, c1)
    net = LateFusionNetwork(trunk, cfg["fusion"], w=cfg.get("w", 0.5),
                            k=cfg.get("k", 3), lam=cfg.get("lam", 0.0),
                            norm_scale=cfg.get("norm_scale"))
    net.load_state_dict(ck["state_dict"])  # strict: a lookalike fails loudly
    net.eval()

    # The g017 runs recorded a node-local copy of the cache; `cc_npz` points
    # at the same file elsewhere.
    z = np.load(cc_npz or run_cfg["cue_conflict_npz"], allow_pickle=False)
    builder = StreamInputBuilder(cfg, age_months=run_cfg["age_months"])
    bs = run_cfg["batch_size"]  # the trainer's cue-conflict batches
    xs_m, xs_p = [], []
    with torch.no_grad():
        for s in range(0, len(z["x"]), bs):
            x = torch.from_numpy(z["x"][s:s + bs]).permute(0, 3, 1, 2)
            x_m, x_p = builder(x.float() / 255.0)
            xs_m.append(x_m)
            xs_p.append(x_p)
    logged = np.load(os.path.join(run_dir, "final_logits.npz"))
    return (trunk, net, torch.cat(xs_m), torch.cat(xs_p),
            list(z["paths"].astype(str)), list(z["shapes"].astype(str)),
            list(z["textures"].astype(str)),
            {h: logged[f"cc_{h}"] for h in ("magno", "parvo")},
            np.asarray(z["cols"]).astype(np.int64))


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
    p.add_argument("--rgb-mode", action="store_true",
                   help="rgb_tsn architecture-only control: both streams get "
                        "the identical 3-channel image, in_channels=(3,3). "
                        "Despite the name the channels are DKL lum/rg/by -- "
                        "every training condition read the DKL h5.")
    p.add_argument("--dvd-age-months", type=float, default=None,
                   help="DVD magno diet at this age instead of blur + CS: "
                        "300 for the fs2_dvd* runs (their dvd_end_months) "
                        "and for the untrained anchor on their front end")
    p.add_argument("--pca-dim", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--out", default="results/shape_texture")
    p.add_argument("--allow-partial", action="store_true",
                   help="probe a still-training checkpoint on purpose")
    p.add_argument("--stage1-run", default=None,
                   help="a Stage-1 late-fusion run directory; replaces "
                        "--checkpoint and every front-end flag, which are "
                        "read from the run itself")
    p.add_argument("--cue-conflict-npz", default=None,
                   help="with --stage1-run: the cue-conflict cache, if the "
                        "path in the run's config.json is gone")
    p.add_argument("--save-features", default=None,
                   help="write the PCA and readout features + labels + "
                        "groups to this .npz")
    args = p.parse_args()
    # An fs2 run name -> its checkpoint; flags that are not the checkpoint's
    # own front end (blur + CS on a DVD-diet run) are refused here.
    fe = dkl_frontend.frontend_from_args(args)

    # Before any compute: a mid-training checkpoint produces output that is
    # indistinguishable from a converged one (PAPER_PLAN §10.1).
    if args.checkpoint:
        features.assert_converged(args.checkpoint, args.allow_partial)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logged = None
    if args.stage1_run:
        (model, net, x_m, x_p, paths, shapes, textures, logged,
         cols) = stage1_model_and_inputs(args.stage1_run,
                                         args.cue_conflict_npz)
        n_images = len(paths)
    else:
        items = list_cue_conflict_images()
        paths = [a for a, _, _ in items]
        shapes = [b for _, b, _ in items]
        textures = [c for _, _, c in items]
        n_images = len(items)
    print(f"{n_images} cue-conflict images | "
          f"{len(set(shapes))} shape classes, {len(set(textures))} texture "
          f"classes | device={device}", flush=True)

    if not args.stage1_run:
        rgb = load_images(paths)
        ch_min, ch_max = dkl_frontend.compute_dkl_channel_minmax()
        x_m, x_p = dkl_frontend.magno_parvo_inputs(
            rgb, ch_min, ch_max, device="cpu", **fe)

        model = features.build_model(
            checkpoint=args.checkpoint, stream1_weight=args.stream1_weight,
            seed=args.seed, device=device,
            in_channels=stream_in_channels(args),
        )
    model.eval()
    feats = features.extract_stream_features(
        model, x_m, x_p, batch_size=args.batch_size, device=device,
    )

    # Chance is 1/n_classes for both probes because both label sets use the
    # same 16 Geirhos categories, so the two accuracies are comparable.
    chance = 1.0 / len(set(shapes))
    out: dict = {"model_name": args.model_name, "n_images": n_images,
                 "chance": chance, "pca_dim": args.pca_dim,
                 "checkpoint": args.checkpoint,
                 "stage1_run": args.stage1_run, "streams": {}}

    content, source = cue_conflict_groups(paths)
    content_fold = folds_within_class(content, shapes, 5)
    source_fold = folds_within_class(source, textures, 3)
    out["groups"] = {"n_content_photos": len(set(content)),
                     "n_texture_sources": len(set(source)),
                     "content_folds": 5, "texture_source_folds": 3}

    # The readout features are what the classifier heads see: the last
    # stage pooled globally. The stage-pooled vector stores that stage as
    # (512, POOL_GRID^2) cells; when the map side is a multiple of POOL_GRID
    # (8 x 8 at 256 px) the cells are equal and their mean is the global
    # average pool exactly. ResNet18 only (stages 64+128+256+512 channels).
    n_last = 512 * features.POOL_GRID ** 2
    assert feats["magno"].shape[1] == 960 * features.POOL_GRID ** 2
    readout = {s: feats[s][:, -n_last:].reshape(n_images, 512, -1).mean(-1)
               for s in ("magno", "parvo")}
    if logged is not None:
        # Same weights, same inputs -> the heads applied to these features
        # must reproduce the logits the trainer logged (bf16 on the GPU, so
        # equality is to bf16 precision, and the 16-way decision must agree).
        out["logit_check"] = {}
        heads = {"magno": net.head_m, "parvo": net.head_p}
        for s, head in heads.items():
            with torch.no_grad():
                mine = head(torch.from_numpy(readout[s])).numpy()
            ref = logged[s]
            agree = float((mine[:, cols].argmax(1)
                           == ref[:, cols].argmax(1)).mean())
            out["logit_check"][s] = {
                "max_abs_diff": float(np.abs(mine - ref).max()),
                "logit_sd": float(ref.std()),
                "decision_agreement_16way": agree,
            }
            print(f"  logit check {s}: max |diff| "
                  f"{out['logit_check'][s]['max_abs_diff']:.4f} "
                  f"(logit sd {ref.std():.3f}), 16-way agreement {agree:.4f}",
                  flush=True)

    pcs = {}
    for stream in ("magno", "parvo"):
        f = feats[stream]
        k = min(args.pca_dim, f.shape[0] - 1, f.shape[1])
        z = PCA(n_components=k, random_state=args.seed).fit_transform(f)
        pcs[stream] = z
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

    # The same probe under the three splits, on the PCA features (the
    # instrument above; `both` = the two streams' components side by side)
    # and on the readout features the heads see.
    spaces = {
        "pca": {**pcs, "both": np.hstack([pcs["magno"], pcs["parvo"]])},
        "readout": {**readout,
                    "both": np.hstack([readout["magno"], readout["parvo"]])},
    }
    out["splits"] = {}
    for space, fs in spaces.items():
        res: dict = {}
        for stream, z in fs.items():
            for split, r in probe_all_splits(z, shapes, textures,
                                             content_fold, source_fold,
                                             args.seed).items():
                res.setdefault(split, {})[stream] = r
        for split, r in res.items():
            r["bias_magno_minus_parvo"] = (r["magno"]["bias"]
                                           - r["parvo"]["bias"])
            print(f"  [{space:<7} {split:<10}] " + "  ".join(
                f"{s} {r[s]['shape_acc']:.3f}/{r[s]['texture_acc']:.3f} "
                f"({r[s]['bias']:+.3f})" for s in ("magno", "parvo", "both")),
                flush=True)
        out["splits"][space] = res

    if args.save_features:
        os.makedirs(os.path.dirname(args.save_features) or ".", exist_ok=True)
        np.savez(args.save_features,
                 pca_magno=pcs["magno"].astype(np.float32),
                 pca_parvo=pcs["parvo"].astype(np.float32),
                 readout_magno=readout["magno"].astype(np.float32),
                 readout_parvo=readout["parvo"].astype(np.float32),
                 shapes=np.asarray(shapes), textures=np.asarray(textures),
                 content=content, source=source,
                 content_fold=content_fold, source_fold=source_fold,
                 paths=np.asarray(paths))
        print(f"wrote {args.save_features}")

    os.makedirs(args.out, exist_ok=True)
    dest = os.path.join(args.out, f"{args.model_name}.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
