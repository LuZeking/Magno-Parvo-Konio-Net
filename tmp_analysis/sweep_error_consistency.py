"""Error consistency against 10 human observers across the ENTIRE readout sweep.

§2.6.2 currently bounds trial-level agreement for TWO models (diet vs no diet).
But §2.3 shows the readout weight `w` spans the whole behavioural range over
frozen representations, so the sharper question is about the whole family:

    Sweeping `w` moves the shape-bias RATE from 0.40 to 0.90 without touching
    either representation. Does trial-level agreement with humans move with it?

This is the test the field's practice invites. The standard move is to tune a
model until its cue-conflict rate approaches the human rate and then call it
human-like. Here the rate can be dialled to any value in that range by changing
one scalar in the readout, with `y1` and `y2` fixed -- so if agreement does not
track the rate, then rate-matching is not evidence of decision-matching, shown
WITHIN one network rather than across a model zoo.

Reports, per seed and checkpoint policy:
  * kappa(w) against each of the 10 Geirhos observers, mean over observers
  * the shape-bias rate at the same w (so kappa can be plotted against rate)
  * the best kappa over the whole sweep -- the strongest statement available
    for ANY readout of these components
  * kappa at the w whose rate is closest to the human rate

Validation before any of it is trusted (§57 practice): the human loading here
must reproduce the published human shape bias 0.958 and the human-human ceiling
kappa 0.234 computed by tmp_analysis/error_consistency.py. Both are asserted.

Usage: PYTHONPATH=. python tmp_analysis/sweep_error_consistency.py \
           --checkpoint ... --model-name ...
"""
from __future__ import annotations

import argparse
import csv
import glob
import itertools
import json
import os
import re

import numpy as np
import torch

from src.analysis.encoding import dkl_frontend, features
from src.analysis.behavior.cue_conflict import (
    GEIRHOS_CLASSES,
    list_cue_conflict_images,
    load_class_index,
    load_images,
    shape_bias,
    stream_in_channels,
)
from fusion_sweep import stream_features   # identical feature path to §2.3

HUMAN_DIR = "/scratch/byrong/VST_data/geirhos_human_data/cue_conflict"
STEM = re.compile(r"([a-z]+)\d+-([a-z]+)\d+$")


def kappa(a, b):
    """Cohen's kappa on aligned 'S'/'T'/None choice vectors (as §2.6.2)."""
    m = np.array([x is not None and y is not None for x, y in zip(a, b)])
    if m.sum() < 30:
        return np.nan
    x = np.array([v == "S" for v in np.asarray(a, dtype=object)[m]])
    y = np.array([v == "S" for v in np.asarray(b, dtype=object)[m]])
    c_obs = float((x == y).mean())
    px, py = float(x.mean()), float(y.mean())
    c_exp = px * py + (1 - px) * (1 - py)
    return np.nan if c_exp >= 1.0 else (c_obs - c_exp) / (1 - c_exp)


def load_humans(stems, shapes, textures):
    index = {st: i for i, st in enumerate(stems)}
    humans = {}
    for f in sorted(glob.glob(f"{HUMAN_DIR}/*subject-*.csv")):
        subj = re.search(r"(subject-\d+)", os.path.basename(f)).group(1)
        choice = [None] * len(stems)
        with open(f) as fh:
            for row in csv.DictReader(fh):
                m = STEM.search(os.path.splitext(row["imagename"])[0])
                if not m:
                    continue
                i = index.get(m.group(0))
                if i is None:
                    continue
                r = row["object_response"]
                if r == shapes[i]:
                    choice[i] = "S"
                elif r == textures[i]:
                    choice[i] = "T"
        humans[subj] = choice
    return humans


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--out-dir", default="results/sweep_error_consistency")
    p.add_argument("--no-lum-blur", action="store_true")
    p.add_argument("--no-lum-cs", action="store_true")
    p.add_argument("--transform-target", default="magno",
                   choices=["magno", "parvo"])
    p.add_argument("--parvo-highsf", action="store_true")
    p.add_argument("--age-months", type=int, default=48)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--w-step", type=float, default=0.05)
    p.add_argument("--allow-partial", action="store_true")
    args = p.parse_args()

    features.assert_converged(args.checkpoint, args.allow_partial)
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    items = list_cue_conflict_images()
    paths = [i[0] for i in items]
    shapes = [i[1] for i in items]
    textures = [i[2] for i in items]
    stems = [os.path.splitext(os.path.basename(p_))[0] for p_ in paths]
    sh_a = np.asarray(shapes, dtype=object)
    tx_a = np.asarray(textures, dtype=object)

    # ---- humans, and the two anchors that say the loading is right ----------
    humans = load_humans(stems, shapes, textures)
    hsb = []
    for c in humans.values():
        nS = sum(v == "S" for v in c); nT = sum(v == "T" for v in c)
        hsb.append(nS / (nS + nT))
    hsb = np.array(hsb)
    hh = np.array([kappa(humans[a], humans[b])
                   for a, b in itertools.combinations(sorted(humans), 2)],
                  dtype=float)
    CEIL = float(np.nanmean(hh))
    print(f"humans: n={len(humans)}  shape bias {hsb.mean():.4f} "
          f"(range {hsb.min():.3f}-{hsb.max():.3f})", flush=True)
    print(f"human-human ceiling: kappa = {CEIL:.4f} "
          f"({np.isfinite(hh).sum()} pairs)", flush=True)
    assert abs(hsb.mean() - 0.958) < 0.002, f"human shape bias {hsb.mean()}"
    assert abs(CEIL - 0.234) < 0.005, f"human-human ceiling {CEIL}"

    # ---- model features, computed once; only `w` changes afterwards ---------
    ch_min, ch_max = dkl_frontend.compute_dkl_channel_minmax()
    model = features.build_model(
        checkpoint=args.checkpoint, stream_concat=False, stream1_weight=0.5,
        seed=args.seed, device=device, in_channels=stream_in_channels(args))
    F1, F2 = [], []
    for start in range(0, len(paths), 128):
        rgb = load_images(paths[start:start + 128])
        x_m, x_p = dkl_frontend.magno_parvo_inputs(
            rgb, ch_min, ch_max, lum_blur=not args.no_lum_blur,
            lum_cs=not args.no_lum_cs, age_months=args.age_months,
            device=device, transform_target=args.transform_target,
            parvo_highsf=args.parvo_highsf, rgb_mode=False)
        a, b = stream_features(model, x_m, x_p, device=device)
        F1.append(a); F2.append(b)
    F1, F2 = torch.cat(F1), torch.cat(F2)

    cls = load_class_index()
    cols = [cls[c] for c in GEIRHOS_CLASSES]
    clf = model.classifier.float().cpu()
    grid = [round(w, 4) for w in np.arange(0.0, 1.0 + 1e-9, args.w_step)]
    rows = []
    with torch.no_grad():
        for w in grid:
            lg = clf(w * F1 + (1.0 - w) * F2).numpy()
            res = shape_bias(lg, shapes, textures, cls)
            dec = np.asarray(GEIRHOS_CLASSES, dtype=object)[lg[:, cols].argmax(1)]
            ch = np.full(len(stems), None, dtype=object)
            ch[dec == sh_a] = "S"
            ch[dec == tx_a] = "T"
            ch = list(ch)
            ks = {h: kappa(ch, humans[h]) for h in sorted(humans)}
            kbar = float(np.nanmean(list(ks.values())))
            rows.append({
                "w": w, "shape_bias": res["shape_bias"],
                "decision_rate": (res["n_shape"] + res["n_texture"]) / res["n_images"],
                "kappa_mean": kbar, "kappa_pct_ceiling": 100 * kbar / CEIL,
                "kappa_per_human": ks,
            })
            print(f"  w={w:.2f}  rate={res['shape_bias']:.3f}  "
                  f"kappa={kbar:+.4f}  ({100*kbar/CEIL:5.1f}% of ceiling)",
                  flush=True)

    sb = np.array([r["shape_bias"] for r in rows])
    kp = np.array([r["kappa_mean"] for r in rows])
    best = int(np.nanargmax(kp))
    near = int(np.nanargmin(np.abs(sb - hsb.mean())))
    print(f"\nrate spans {sb.min():.3f}-{sb.max():.3f} "
          f"({sb.max()-sb.min():.3f}) across w")
    print(f"kappa spans {np.nanmin(kp):+.4f}-{np.nanmax(kp):+.4f} "
          f"({100*np.nanmin(kp)/CEIL:.1f}%-{100*np.nanmax(kp)/CEIL:.1f}% of ceiling)")
    print(f"BEST kappa over the whole readout family: {kp[best]:+.4f} at w={grid[best]:.2f} "
          f"(rate {sb[best]:.3f}) = {100*kp[best]/CEIL:.1f}% of ceiling")
    print(f"kappa at the w closest to the human rate ({hsb.mean():.3f}): "
          f"{kp[near]:+.4f} at w={grid[near]:.2f} (rate {sb[near]:.3f})")

    out = {"model_name": args.model_name, "checkpoint": args.checkpoint,
           "human_shape_bias": float(hsb.mean()), "human_human_ceiling": CEIL,
           "sweep": rows}
    path = os.path.join(args.out_dir, f"{args.model_name}_kappa_sweep.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved -> {path}", flush=True)


if __name__ == "__main__":
    main()
