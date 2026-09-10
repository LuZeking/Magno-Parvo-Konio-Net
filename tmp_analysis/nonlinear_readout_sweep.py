"""§104: does the readout span survive a nonlinear head?

§2.3's sweep is an identity of the affine head. This refits a head on the frozen
per-stream features at each of the same 21 values of w and asks whether the
cue-conflict shape-bias span survives.

THREE heads are fitted at every w, not one, because "nonlinear" and "refitted"
are different manipulations and the registered question is about the first:

  affine   -- the trained classifier's own Linear, applied unchanged (this
              reproduces §2.3 exactly and is the baseline)
  linear   -- a Linear(512,112) REFITTED at this w  (isolates refitting)
  mlp      -- Linear(512,512) -> ReLU -> Linear(512,112) refitted at this w
              (isolates nonlinearity, given the linear control)

The linear control is an addition to §104.3's registered design A. It can only
strengthen the test: without it, a change in span could not be attributed to
nonlinearity rather than to the refit.

Every head gets an identical budget (§104.3's tuning-fairness commitment) and an
identical init seed across w, so differences come from w and not from the draw.
MiniEcoset val accuracy is reported at every w: §104.4 makes it a gate, not a
footnote -- a span that appears only where accuracy has collapsed is not a
readout result.

Usage: PYTHONPATH=.:tmp_analysis python tmp_analysis/nonlinear_readout_sweep.py \
           --features .../dkl_bcs_seed0_best_features.npz --out-dir ...
"""
from __future__ import annotations

import argparse, json, os, time

import numpy as np
import torch
import torch.nn as nn

from src.analysis.behavior.cue_conflict import shape_bias, load_class_index

# fixed now, identical for every w and every head (§104.3)
EPOCHS, BATCH, LR, HIDDEN, INIT_SEED = 20, 1024, 1e-3, 512, 0
WS = np.round(np.linspace(0.0, 1.0, 21), 3)


def make_head(kind: str, d: int, n_cls: int) -> nn.Module:
    torch.manual_seed(INIT_SEED)          # same draw at every w
    if kind == "linear":
        return nn.Linear(d, n_cls)
    return nn.Sequential(nn.Linear(d, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, n_cls))


def fit(head, X, y, device):
    head = head.to(device).train()
    opt = torch.optim.Adam(head.parameters(), lr=LR)
    lossf = nn.CrossEntropyLoss()
    n = X.shape[0]
    g = torch.Generator().manual_seed(INIT_SEED)
    for _ in range(EPOCHS):
        perm = torch.randperm(n, generator=g)
        for s in range(0, n, BATCH):
            idx = perm[s:s + BATCH]
            xb = X[idx].to(device, non_blocking=True).float()
            yb = y[idx].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            lossf(head(xb), yb).backward()
            opt.step()
    return head.eval()


@torch.no_grad()
def accuracy(head, X, y, device, batch=4096) -> float:
    ok = 0
    for s in range(0, X.shape[0], batch):
        xb = X[s:s + batch].to(device).float()
        ok += int((head(xb).argmax(1).cpu() == y[s:s + batch]).sum())
    return ok / X.shape[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    z = np.load(args.features, allow_pickle=True)
    tag = str(z["tag"])
    trF1 = torch.from_numpy(z["train_F1"]); trF2 = torch.from_numpy(z["train_F2"])
    try_ = torch.from_numpy(z["train_y"]).long()
    vaF1 = torch.from_numpy(z["val_F1"]);   vaF2 = torch.from_numpy(z["val_F2"])
    vay = torch.from_numpy(z["val_y"]).long()
    ccF1 = torch.from_numpy(z["cc_F1"]).float()
    ccF2 = torch.from_numpy(z["cc_F2"]).float()
    shapes, textures = list(z["cc_shapes"]), list(z["cc_textures"])
    cls = load_class_index()
    n_cls = int(try_.max()) + 1
    print(f"{tag}: train {tuple(trF1.shape)} val {tuple(vaF1.shape)} "
          f"cc {tuple(ccF1.shape)} classes={n_cls} device={device}", flush=True)

    rows, t0 = [], time.time()
    for w in WS:
        Xtr = (w * trF1.float() + (1 - w) * trF2.float()).half()
        Xva = (w * vaF1.float() + (1 - w) * vaF2.float()).half()
        Xcc = (w * ccF1 + (1 - w) * ccF2)
        for kind in ("linear", "mlp"):
            head = fit(make_head(kind, Xtr.shape[1], n_cls), Xtr, try_, device)
            acc = accuracy(head, Xva, vay, device)
            with torch.no_grad():
                logits = head(Xcc.to(device)).cpu().numpy()
            sb = shape_bias(logits, shapes, textures, cls)
            rows.append({"tag": tag, "w": float(w), "head": kind,
                         "val_top1": acc, "shape_bias": sb["shape_bias"],
                         "n_other": int(sb["n_other"])})
            print(f"  w={w:.2f} {kind:6s} val={acc:.4f} sb={sb['shape_bias']:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    dest = os.path.join(args.out_dir, f"{tag}_nonlinear_sweep.json")
    json.dump(rows, open(dest, "w"), indent=2)
    print(f"saved -> {dest}", flush=True)


if __name__ == "__main__":
    main()
