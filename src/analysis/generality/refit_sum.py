"""Lossy-sum mixture with refitted heads (PLAN_GENERALITY D3).

  python -m src.analysis.generality.refit_sum --cc DIR/<tag>_cue_conflict.npz \
      --mini DIR/<tag>_miniecoset.npz --out-dir DIR

A per-unit gain can be undone by a refitted linear head, so the refit test
mixes the two halves the way the two-stream model does: z(w) = w*y_S + (1-w)*y_T,
with y_S the shape-leaning half ordered by descending d and y_T the
texture-leaning half ordered by ascending d (paired by rank, the analogue of
the two streams' index pairing). A linear and a two-layer head are refitted on
MiniEcoset training features at every w with the budget of Methods 4.3
(Adam, lr 1e-3, batch 1024, 20 epochs, one init seed), validation accuracy is
recorded, and the cue-conflict rate is read on the half of the images that did
not define the split. Both directions of the first stratified split are run.
A refit on the unmixed full y gives the accuracy ceiling of the refit protocol.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from src.analysis.generality.scoring import Scorer
from src.analysis.generality.unit_factor import WS, median_split, shape_score, stratified_halves

EPOCHS, BATCH, LR, HIDDEN, INIT_SEED = 20, 1024, 1e-3, 512, 0


def make_head(kind: str, d: int, n_cls: int) -> nn.Module:
    torch.manual_seed(INIT_SEED)
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
            opt.zero_grad(set_to_none=True)
            lossf(head(X[idx].to(device)), y[idx].to(device)).backward()
            opt.step()
    return head.eval()


@torch.no_grad()
def accuracy(head, X, y, device, batch=4096) -> float:
    ok = 0
    for s in range(0, X.shape[0], batch):
        ok += int((head(X[s:s + batch].to(device)).argmax(1).cpu() == y[s:s + batch]).sum())
    return ok / X.shape[0]


@torch.no_grad()
def logits_of(head, X, device) -> np.ndarray:
    return head(X.to(device)).cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cc", required=True)
    p.add_argument("--mini", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--heads", default="linear,mlp")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "8"))))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cc = np.load(args.cc, allow_pickle=True)
    mi = np.load(args.mini, allow_pickle=True)
    tag, kind = str(cc["tag"]), str(cc["scorer"])
    if kind != "miniecoset112":
        raise SystemExit("refit needs a MiniEcoset-trained network")
    scorer = Scorer(kind)
    ycc = cc["features"].astype(np.float32)
    shapes, textures = cc["shapes"], cc["textures"]
    trF = torch.from_numpy(mi["train_features"].astype(np.float32))
    trY = torch.from_numpy(mi["train_y"]).long()
    vaF = torch.from_numpy(mi["val_features"].astype(np.float32))
    vaY = torch.from_numpy(mi["val_y"]).long()
    n_cls = int(trY.max()) + 1
    heads = args.heads.split(",")
    print(f"[refit_sum] {tag}: train {tuple(trF.shape)} val {tuple(vaF.shape)} cc {ycc.shape} heads {heads}", flush=True)

    rows, t0 = [], time.time()
    # ceiling of the refit protocol: unmixed full y
    for kind_h in heads:
        h = fit(make_head(kind_h, trF.shape[1], n_cls), trF, trY, device)
        acc = accuracy(h, vaF, vaY, device)
        sc = scorer.score(logits_of(h, torch.from_numpy(ycc), device), shapes, textures)
        rows.append(dict(tag=tag, direction="full", w=None, head=kind_h, val_top1=acc, **sc))
        print(f"  full y {kind_h:6s} val={acc:.4f} sb={sc['shape_bias']:.3f} ({time.time() - t0:.0f}s)", flush=True)

    rng = np.random.default_rng(args.seed)
    A, B = stratified_halves(shapes, textures, rng)  # the first split of unit_factor (same seed)
    for name, fitidx, evidx in (("A>B", A, B), ("B>A", B, A)):
        d = shape_score(ycc[fitidx].astype(np.float64), shapes[fitidx], textures[fitidx])
        S = median_split(d)
        S_order = np.flatnonzero(S)[np.argsort(-d[S], kind="stable")]
        T_order = np.flatnonzero(~S)[np.argsort(d[~S], kind="stable")]
        k = min(len(S_order), len(T_order))
        S_order, T_order = S_order[:k], T_order[:k]
        for w in WS:
            w = float(w)
            Xtr = w * trF[:, S_order] + (1 - w) * trF[:, T_order]
            Xva = w * vaF[:, S_order] + (1 - w) * vaF[:, T_order]
            Xcc = torch.from_numpy(w * ycc[evidx][:, S_order] + (1 - w) * ycc[evidx][:, T_order])
            for kind_h in heads:
                h = fit(make_head(kind_h, k, n_cls), Xtr, trY, device)
                acc = accuracy(h, Xva, vaY, device)
                sc = scorer.score(logits_of(h, Xcc, device), shapes[evidx], textures[evidx])
                rows.append(dict(tag=tag, direction=name, w=w, head=kind_h, val_top1=acc, n_eval=int(len(evidx)), **sc))
                print(f"  {name} w={w:.2f} {kind_h:6s} val={acc:.4f} sb={sc['shape_bias']:.3f} "
                      f"forced={sc['forced_two_way']:.3f} decided={sc['n_shape'] + sc['n_texture']} ({time.time() - t0:.0f}s)", flush=True)

    dest = os.path.join(args.out_dir, f"{tag}_refit_sum.json")
    json.dump(rows, open(dest, "w"), indent=1)
    print(f"[refit_sum] wrote {dest}", flush=True)


if __name__ == "__main__":
    main()
