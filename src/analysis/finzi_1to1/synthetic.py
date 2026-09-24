"""Synthetic end-to-end check of mapping.py + stats.py (plan amendment 1).

A planted magno->dorsal preference must come out positive, a null network near
zero, both LOSO positive controls must pass and the negative control must be
calibrated. Writes into FINZI_OUT_ROOT -- point it at a scratch directory:

    FINZI_OUT_ROOT=/tmp/f1to1_synth python -m src.analysis.finzi_1to1.synthetic
    # then mapping --mode main / --mode pc for diet_s0 nodiet_s0 tdann_a025_s0, and stats
"""
import json
import os

import numpy as np

from src.analysis.finzi_1to1 import betas_cache_path, features_path

rng = np.random.default_rng(1)
K, NTR, NTE = 20, 600, 150
N_POS, N_CH = 64, 48
LAB = {"ventral": 5, "lateral": 6, "dorsal": 7}


def voxels(sub_rng):
    hemi, lab, nc, load = [], [], [], []
    for h in (0, 1):
        for name, n in (("ventral", 1200), ("lateral", 1000), ("dorsal", 700)):
            w = np.zeros((n, K))
            if name == "ventral":
                w[:, :10] = sub_rng.standard_normal((n, 10))
            elif name == "dorsal":
                w[:, 10:] = sub_rng.standard_normal((n, 10))
            else:
                w[:] = sub_rng.standard_normal((n, K)) * 0.7
            hemi += [h] * n
            lab += [LAB[name]] * n
            # dorsal gets the highest NC, as in the real data
            base = {"ventral": 20, "lateral": 25, "dorsal": 35}[name]
            nc += list(np.clip(sub_rng.normal(base, 8, n), 10, 80))
            load.append(w)
    return np.array(hemi, np.int8), np.array(lab), np.array(nc), np.concatenate(load)


def units(kind, u_rng):
    if kind == "tdann":  # single stream, half the units dorsal-loading (planted)
        stream = np.zeros(49 * 64, np.int8)
        pos = np.tile(np.arange(49), 64)
        w = np.zeros((len(pos), K))
        for i in range(len(pos)):
            dors = u_rng.random() < 0.5
            w[i, 10:] = u_rng.standard_normal(10) * (1.0 if dors else 0.2)
            w[i, :10] = u_rng.standard_normal(10) * (0.2 if dors else 1.0)
        return stream, pos, w
    stream = np.repeat([0, 1], N_POS * N_CH).astype(np.int8)
    pos = np.tile(np.repeat(np.arange(N_POS), N_CH), 2)
    w = np.zeros((len(pos), K))
    for i, s in enumerate(stream):
        if kind == "planted":
            dors = u_rng.random() < (0.7 if s == 0 else 0.3)
            w[i, 10:] = u_rng.standard_normal(10) * (1.0 if dors else 0.2)
            w[i, :10] = u_rng.standard_normal(10) * (0.2 if dors else 1.0)
        else:
            w[i] = u_rng.standard_normal(K)
    return stream, pos, w


Z_tr = rng.standard_normal((NTR, K))
Z_te = rng.standard_normal((NTE, K))
nets = {"diet_s0": "planted", "nodiet_s0": "null", "tdann_a025_s0": "tdann"}
net_units = {n: units(k, np.random.default_rng(hash(n) % 2**32)) for n, k in nets.items()}
tr_ids, te_ids = np.arange(NTR), np.arange(10000, 10000 + NTE)
for s in range(1, 9):
    srng = np.random.default_rng(100 + s)
    hemi, lab, nc, load = voxels(srng)
    noise_sd = np.sqrt((100 - nc) / nc)[None, :]
    btr = Z_tr @ load.T + srng.standard_normal((NTR, len(nc))) * noise_sd * 3
    bte = Z_te @ load.T + srng.standard_normal((NTE, len(nc))) * noise_sd
    os.makedirs(os.path.dirname(betas_cache_path(s)), exist_ok=True)
    np.savez(betas_cache_path(s), train_image_ids=tr_ids, train_betas=btr.astype(np.float32),
             train_n_reps=np.ones(NTR, int), test_image_ids=te_ids,
             test_betas=bte.astype(np.float32), hemi=hemi, label=lab, nc=nc,
             zyx=np.zeros((len(nc), 3), np.int16))
    for n, (stream, pos, w) in net_units.items():
        ftr = np.maximum(Z_tr @ w.T + 0.5 * srng.standard_normal((NTR, len(pos))), 0)
        fte = np.maximum(Z_te @ w.T + 0.5 * srng.standard_normal((NTE, len(pos))), 0)
        os.makedirs(os.path.dirname(features_path(n, s)), exist_ok=True)
        np.savez(features_path(n, s), train_feats=ftr.astype(np.float16),
                 test_feats=fte.astype(np.float16), train_image_ids=tr_ids,
                 test_image_ids=te_ids, unit_stream=stream, unit_pos=pos.astype(np.int16),
                 unit_chan=np.zeros(len(pos), np.int16), provenance=json.dumps({"synthetic": n}))
print("synthetic data written")
