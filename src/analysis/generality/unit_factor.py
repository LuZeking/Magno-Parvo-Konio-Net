"""Unit factorisation and the fixed-head gain sweep (PLAN_GENERALITY D1-D2).

  python -m src.analysis.generality.unit_factor --features DIR/<tag>_cue_conflict.npz --out-dir DIR

For each unit j of the penultimate layer, eta2_s(j) and eta2_t(j) are the
fractions of its variance over the 1,200 cue-conflict images that lie between
the 16 shape categories and between the 16 texture categories (one-way).
d_j = eta2_s - eta2_t; the median split of d gives the shape-leaning half S and
the texture-leaning half T.

Behaviour: the split is defined on one half of the images (stratified by
shape x texture pair) and the gain sweep  logits(w) = head(g(w) * y),
g = 2w on S and 2(1-w) on T, is scored on the other half; 10 random splits,
both directions. w = 0.5 is the trained network exactly.

Also written: the split defined on ALL images (for the cortex analysis, which
has no circularity with the cue-conflict set) and, when the features carry
`stream_of_unit`, the fraction of S drawn from stream 1 (validation P3).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy.stats import spearmanr

from src.analysis.generality.scoring import Scorer

WS = np.round(np.linspace(0.0, 1.0, 21), 3)


def eta2(y: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Per-column fraction of variance between the groups of `labels` (one-way)."""
    y = y.astype(np.float64)
    mu = y.mean(0, keepdims=True)
    ss_tot = ((y - mu) ** 2).sum(0)
    ss_between = np.zeros(y.shape[1])
    for g in np.unique(labels):
        m = labels == g
        ss_between += m.sum() * (y[m].mean(0) - mu[0]) ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        out = ss_between / ss_tot
    out[~np.isfinite(out)] = 0.0
    return out


def shape_score(y, shapes, textures) -> np.ndarray:
    return eta2(y, shapes) - eta2(y, textures)


def median_split(d: np.ndarray) -> np.ndarray:
    """Boolean S mask: the top half of units by d (ties broken by index)."""
    order = np.lexsort((np.arange(len(d)), -d))  # descending d, stable
    S = np.zeros(len(d), bool)
    S[order[: len(d) // 2]] = True
    return S


def stratified_halves(shapes, textures, rng) -> tuple[np.ndarray, np.ndarray]:
    pair = np.array([f"{s}|{t}" for s, t in zip(shapes, textures)])
    a, b = [], []
    for p in np.unique(pair):
        idx = rng.permutation(np.flatnonzero(pair == p))
        k = len(idx) // 2
        if len(idx) % 2 and rng.random() < 0.5:
            k += 1
        a.extend(idx[:k]); b.extend(idx[k:])
    return np.array(sorted(a)), np.array(sorted(b))


def gain_sweep(y, W, b, S, scorer: Scorer, shapes, textures) -> list[dict]:
    rows = []
    for w in WS:
        g = np.where(S, 2 * w, 2 * (1 - w)).astype(np.float64)
        logits = (y * g) @ W.T + b
        sc = scorer.score(logits, shapes, textures)
        rows.append(dict(w=float(w), **sc))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-splits", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    z = np.load(args.features, allow_pickle=True)
    tag, kind = str(z["tag"]), str(z["scorer"])
    y = z["features"].astype(np.float64)
    W, b = z["W"].astype(np.float64), z["b"].astype(np.float64)
    shapes, textures = z["shapes"], z["textures"]
    scorer = Scorer(kind)
    N, D = y.shape
    print(f"[unit_factor] {tag}: {N} images, {D} units, scorer {kind}", flush=True)

    # trained readout, every image (must reproduce the stored trained_score)
    trained = scorer.score(y @ W.T + b, shapes, textures)
    print(f"  trained: shape bias {trained['shape_bias']:.3f} forced {trained['forced_two_way']:.3f}", flush=True)

    # split on all images (cortex use) + descriptives
    d_full = shape_score(y, shapes, textures)
    S_full = median_split(d_full)
    e_s, e_t = eta2(y, shapes), eta2(y, textures)
    desc = dict(frac_units_shape_leaning=float((d_full > 0).mean()),
                median_eta2_shape=float(np.median(e_s)), median_eta2_texture=float(np.median(e_t)),
                median_d_S=float(np.median(d_full[S_full])), median_d_T=float(np.median(d_full[~S_full])))
    stream_of_unit = z["stream_of_unit"] if "stream_of_unit" in z.files else None
    if stream_of_unit is not None:
        desc["frac_S_from_stream1_full"] = float((stream_of_unit[S_full] == 0).mean())
        desc["frac_T_from_stream2_full"] = float((stream_of_unit[~S_full] == 1).mean())

    # in-sample sweep on all images with the all-image split (reference only)
    insample = gain_sweep(y, W, b, S_full, scorer, shapes, textures)

    # held-out: split on half A, score on half B, both directions
    rng = np.random.default_rng(args.seed)
    held = []      # list of per-split curves
    frac_s1 = []
    for k in range(args.n_splits):
        A, B = stratified_halves(shapes, textures, rng)
        for name, fit, ev in (("A>B", A, B), ("B>A", B, A)):
            d = shape_score(y[fit], shapes[fit], textures[fit])
            S = median_split(d)
            if stream_of_unit is not None:
                frac_s1.append(float((stream_of_unit[S] == 0).mean()))
            curve = gain_sweep(y[ev], W, b, S, scorer, shapes[ev], textures[ev])
            held.append(dict(split=k, direction=name, n_eval=int(len(ev)), curve=curve,
                             overlap_with_full=float((S == S_full).mean())))
            if k == 0:
                print(f"  split {k} {name}: rate at w=0 {curve[0]['shape_bias']:.3f}, "
                      f"w=0.5 {curve[10]['shape_bias']:.3f}, w=1 {curve[-1]['shape_bias']:.3f}", flush=True)

    def summarise(key):
        M = np.array([[r[key] for r in h["curve"]] for h in held])  # (n_curves, n_w)
        spans = M.max(1) - M.min(1)
        rhos = np.array([spearmanr(WS, row).correlation for row in M])
        return dict(mean_curve=M.mean(0).tolist(), min_curve=M.min(0).tolist(), max_curve=M.max(0).tolist(),
                    span_mean=float(spans.mean()), span_min=float(spans.min()), span_max=float(spans.max()),
                    rho_mean=float(np.nanmean(rhos)), rho_min=float(np.nanmin(rhos)),
                    rate_w0=float(M[:, 0].mean()), rate_w05=float(M[:, 10].mean()), rate_w1=float(M[:, -1].mean()))

    summary = dict(tag=tag, scorer=kind, n_images=int(N), n_units=int(D), n_splits=args.n_splits,
                   trained=trained, trained_score_stored=json.loads(str(z["trained_score"])),
                   descriptives=desc, ws=WS.tolist(),
                   primary=summarise("shape_bias"), forced=summarise("forced_two_way"),
                   decided_mean_curve=np.array([[r["n_shape"] + r["n_texture"] for r in h["curve"]] for h in held]).mean(0).tolist(),
                   insample_primary=[r["shape_bias"] for r in insample],
                   insample_forced=[r["forced_two_way"] for r in insample])
    if frac_s1:
        summary["descriptives"]["frac_S_from_stream1_heldout_mean"] = float(np.mean(frac_s1))
        summary["descriptives"]["frac_S_from_stream1_heldout_min"] = float(np.min(frac_s1))
    print(f"  held-out span primary {summary['primary']['span_mean']:.3f} "
          f"[{summary['primary']['span_min']:.3f}, {summary['primary']['span_max']:.3f}], "
          f"rho {summary['primary']['rho_mean']:.2f}; forced span {summary['forced']['span_mean']:.3f}; "
          f"units shape-leaning {desc['frac_units_shape_leaning']:.2f}", flush=True)

    json.dump(dict(summary=summary, held_out=held), open(os.path.join(args.out_dir, f"{tag}_unit_factor.json"), "w"), indent=1)
    np.savez(os.path.join(args.out_dir, f"{tag}_unit_factor.npz"), tag=tag, d_full=d_full, S_full=S_full,
             eta2_shape=e_s, eta2_texture=e_t, ws=WS)
    print(f"[unit_factor] wrote {args.out_dir}/{tag}_unit_factor.{{json,npz}}", flush=True)


if __name__ == "__main__":
    main()
