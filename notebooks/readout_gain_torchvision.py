"""Reproduce the Fig. 6b ResNet-50 curve from a fresh download, on a laptop CPU, in minutes.

    pip install -r requirements.txt
    PYTHONPATH=. python notebooks/readout_gain_torchvision.py [--model tv:resnet50] [--data ./data]

What it does (Methods 4.8; docs/PLAN_SUBSPACE_2026-09-15.md):
  1. downloads the Geirhos et al. (2019) cue-conflict stimuli from the authors' GitHub
     repository (style-transfer-preprocessed-512; ~1,280 PNGs) unless CUE_CONFLICT_DIR is set;
  2. loads torchvision ResNet-50 (IMAGENET1K_V1 weights, downloaded by torchvision) and takes
     the penultimate features y of the 1,200 images whose shape and texture categories differ;
  3. scores the trained readout with the Geirhos 16-class rule;
  4. fits the shape- and texture-predictive subspaces on one stratified half of the images,
     sweeps the gain w on the other half (10 splits, both directions) with the head untouched,
     and prints the held-out curve, its span and Spearman rho(w, rate);
  5. compares every number with the committed results/figures_v2/subspace_behaviour.json;
  6. if IMAGENET_VAL points at the ILSVRC-2012 validation images, also runs the accuracy gate
     (stratified 10-per-class subset, 10,000 images; ~20 min on CPU) and prints the gated span.
Writes notebooks/readout_gain_<tag>.png and .json.

Expected on a 2024 laptop CPU: download ~200 MB, ResNet-50 features ~2 min, sweeps ~2 min.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import zipfile

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

ZIP_URL = "https://github.com/rgeirhos/texture-vs-shape/archive/refs/heads/master.zip"
STIM_SUBDIR = "stimuli/style-transfer-preprocessed-512"


def ensure_stimuli(data_dir: str) -> str:
    """Return the stimuli directory, downloading the authors' repository archive if needed."""
    if os.environ.get("CUE_CONFLICT_DIR") and os.path.isdir(os.environ["CUE_CONFLICT_DIR"]):
        return os.environ["CUE_CONFLICT_DIR"]
    dest = os.path.join(data_dir, "geirhos_cue_conflict", STIM_SUBDIR)
    if os.path.isdir(dest) and len(os.listdir(dest)) >= 16:
        return dest
    import urllib.request
    print(f"downloading {ZIP_URL} ...", flush=True)
    t0 = time.time()
    with urllib.request.urlopen(ZIP_URL, timeout=600) as r:
        blob = r.read()
    print(f"  {len(blob) / 1e6:.0f} MB in {time.time() - t0:.0f}s; extracting {STIM_SUBDIR}", flush=True)
    zf = zipfile.ZipFile(io.BytesIO(blob))
    n = 0
    for info in zf.infolist():
        parts = info.filename.split("/", 1)
        if len(parts) < 2 or not parts[1].startswith(STIM_SUBDIR + "/") or info.is_dir():
            continue
        rel = parts[1][len(STIM_SUBDIR) + 1:]
        out = os.path.join(dest, rel)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "wb") as fh:
            fh.write(zf.read(info))
        n += 1
    print(f"  {n} files -> {dest}", flush=True)
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tv:resnet50")
    ap.add_argument("--data", default=os.path.join(REPO, "data"))
    ap.add_argument("--n-splits", type=int, default=10)
    ap.add_argument("--threads", type=int, default=0, help="torch threads (0 = default)")
    args = ap.parse_args()

    stim = ensure_stimuli(args.data)
    os.environ["CUE_CONFLICT_DIR"] = stim                     # read at import time by cue_conflict.py
    import numpy as np
    import torch
    if args.threads:
        torch.set_num_threads(args.threads)
    from scipy.stats import spearmanr
    from src.analysis.behavior.cue_conflict import list_cue_conflict_images, load_images
    from src.analysis.generality.models import FrozenNet
    from src.analysis.generality.scoring import Scorer
    from src.analysis.generality.subspace_factor import WS, gain_sweep, subspace_bases
    from src.analysis.generality.unit_factor import stratified_halves

    items = list_cue_conflict_images()
    assert len(items) == 1200, f"expected 1,200 cue-conflict images with shape != texture, found {len(items)}"
    shapes = np.array([i[1] for i in items]); textures = np.array([i[2] for i in items])

    t0 = time.time()
    net = FrozenNet(args.model, device="cpu")
    feats = []
    for s in range(0, len(items), 100):
        feats.append(net.features_from_rgb(load_images([i[0] for i in items[s:s + 100]])))
        print(f"  features {min(s + 100, len(items))}/1200  {time.time() - t0:.0f}s", flush=True)
    y = np.concatenate(feats).astype(np.float64)
    W, b = net.W.astype(np.float64), net.b.astype(np.float64)
    scorer = Scorer(net.scorer)
    trained = scorer.score(y @ W.T + b, shapes, textures)
    print(f"\n{net.tag}: D = {y.shape[1]}; trained shape-bias rate {trained['shape_bias']:.3f} "
          f"({trained['n_shape'] + trained['n_texture']}/1200 images decided)", flush=True)

    rng = np.random.default_rng(0)
    curves = []
    for k in range(args.n_splits):
        A, B = stratified_halves(shapes, textures, rng)
        for fit, ev in ((A, B), (B, A)):
            B_S, B_T, _ = subspace_bases(y, shapes, textures, fit, 256, 15, "lowdin")
            curve = gain_sweep(y[ev], W, b, B_S, B_T, scorer, shapes[ev], textures[ev])
            curves.append([r["shape_bias"] for r in curve])
    M = np.array(curves)
    mean_curve = M.mean(0)
    span = float((M.max(1) - M.min(1)).mean())
    rho = float(np.mean([spearmanr(WS, row).correlation for row in M]))
    print(f"held-out sweep, w = 0 -> 1: rate {mean_curve[0]:.3f} -> {mean_curve[-1]:.3f}; "
          f"span {span:.3f}; rho(w, rate) {rho:.2f}", flush=True)

    out = dict(model=args.model, tag=net.tag, trained=trained, ws=WS.tolist(), mean_curve=mean_curve.tolist(),
               span=span, rho=rho, n_splits=args.n_splits)

    # ---- compare with the committed numbers
    ref_path = os.path.join(REPO, "results/figures_v2/subspace_behaviour.json")
    key = args.model.split(":")[1] if args.model.startswith("tv:") else None
    if key and os.path.exists(ref_path):
        fam = {f["key"]: f for f in json.load(open(ref_path))["families"]}
        if key in fam:
            ref = fam[key]
            dev = float(np.max(np.abs(np.array(ref["seeds"][0]["mean_curve"]) - mean_curve)))
            print(f"committed (results/figures_v2/subspace_behaviour.json): trained {ref['trained']:.3f}, span {ref['span']:.3f}, rho {ref['rho']:.2f}; "
                  f"largest deviation of the mean curve here from the committed one {dev:.3f} "
                  f"(splits are seeded identically; differences come from the machine's BLAS and image decoding)", flush=True)
            out["committed"] = dict(trained=ref["trained"], span=ref["span"], rho=ref["rho"], gated_span=ref.get("gated_span"), max_curve_dev=dev)

    # ---- optional accuracy gate
    val_root = os.environ.get("IMAGENET_VAL")
    if val_root and os.path.isdir(val_root) and net.scorer == "imagenet16":
        from PIL import Image
        lab = np.load(os.path.join(REPO, "src/analysis/generality/imagenet_val_labels.npz"), allow_pickle=True)
        names, labels = lab["names"], lab["labels"].astype(int)
        rng2 = np.random.default_rng(0)
        pick = np.sort(np.concatenate([rng2.permutation(np.flatnonzero(labels == c))[:10] for c in range(1000)]))
        yv = []
        for s in range(0, len(pick), 64):
            imgs = [np.asarray(Image.open(os.path.join(val_root, str(names[k]))).convert("RGB").resize((256, 256), Image.BILINEAR), dtype=np.uint8)
                    for k in pick[s:s + 64]]
            yv.append(net.features_from_rgb(np.stack(imgs)))
            if (s // 64) % 25 == 0:
                print(f"  imagenet_val {s}/{len(pick)}  {time.time() - t0:.0f}s", flush=True)
        yv = np.concatenate(yv).astype(np.float64)
        from src.analysis.generality.subspace_factor import gain_features
        B_S, B_T, _ = subspace_bases(y, shapes, textures, np.arange(len(y)), 256, 15, "lowdin")
        acc = np.array([(np.argmax(gain_features(yv, B_S, B_T, float(w)) @ W.T + b, 1) == labels[pick]).mean() for w in WS])
        gate = acc >= acc[10] - 0.05
        gspan = float(mean_curve[gate].max() - mean_curve[gate].min())
        print(f"accuracy gate: trained top-1 {acc[10]:.3f}; within 5 points for w in [{WS[gate].min():.2f}, {WS[gate].max():.2f}]; gated span {gspan:.3f}", flush=True)
        out["gate"] = dict(top1=acc.tolist(), trained_top1=float(acc[10]), w_within_5pts=WS[gate].tolist(), gated_span=gspan)
    else:
        print("accuracy gate skipped (set IMAGENET_VAL to the ILSVRC-2012 validation directory to run it)", flush=True)

    # ---- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(3.6, 2.8), dpi=150)
    ax.fill_between(WS, M.min(0), M.max(0), color="#BBBBBB", alpha=0.4, lw=0, label="range over 20 half-splits")
    ax.plot(WS, mean_curve, color="#1A1A1A", lw=1.6, label="held-out mean")
    ax.plot(0.5, trained["shape_bias"], "o", color="#D55E00", ms=5, label="trained readout (w = 0.5)")
    if "committed" in out:
        ax.plot(WS, fam[key]["seeds"][0]["mean_curve"], color="#0072B2", lw=1, ls="--", label="committed curve (Fig. 6b)")
    ax.set_xlabel("gain w on the shape-predictive subspace"); ax.set_ylabel("cue-conflict shape-bias rate")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_title(f"{net.tag}: span {span:.2f}, ρ {rho:.2f}", fontsize=9)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    fig.tight_layout()
    dest = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"readout_gain_{net.tag}")
    fig.savefig(dest + ".png")
    json.dump(out, open(dest + ".json", "w"), indent=1)
    print(f"wrote {dest}.png and .json  ({time.time() - t0:.0f}s in total)")


if __name__ == "__main__":
    main()
