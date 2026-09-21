"""Exemplar-held-out probes and spans (docs/PLAN_EXEMPLAR_2026-09-21.md, pre-registered 2026-09-21).

  python -m src.analysis.generality.exemplar --features FEAT/<tag>_cue_conflict.npz \
      --behaviour-dir BEH --out-dir OUT

The 1,200 cue-conflict images are built from 160 object exemplars (10 per category) and 48 texture
exemplars (3 per category); the stem `<shape><i>-<texture><j>` names both. The paper's probe folds and its
held-out halves are stratified on labels, so the scored images share exemplars with the fitted ones. This
module re-derives the same quantities with the exemplars held out. Bases, gain and scoring are the paper's
own (subspace_factor, scoring); only the fit/score partition changes.

Spans:  E0 the paper's pair-stratified halves (must reproduce the stored span), E1 fully
        exemplar-disjoint 3x3 folds (primary), E2 texture-exemplar folds, E3 object-exemplar folds.
Probes: Q0 the paper's probe (must reproduce the stored bias), Q1 exemplar-balanced with PCA inside the
        folds (primary), Q1b the same with PCA on all images, Q2 exemplar-grouped 5-fold.

Writes <tag>_exemplar.json and <tag>_exemplar.npz (per-image trained decisions for the ranking's
test-retest floor).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import warnings

import numpy as np
from scipy.stats import spearmanr

from src.analysis.generality.scoring import Scorer
from src.analysis.generality.subspace_factor import gain_features, subspace_bases
from src.analysis.generality.unit_factor import WS, stratified_halves

N_REPS = 10        # E1-E3 repetitions
N_DRAWS = 20       # Q1 / Q1b exemplar draws
N_EX_BAL = 3       # exemplars per category in the balanced probe (the texture set has three)


# ----------------------------------------------------------------------------------------------- exemplars
def parse_exemplars(stems) -> tuple[np.ndarray, np.ndarray]:
    """Object and texture exemplar ids from `<shape><i>-<texture><j>` stems."""
    obj, tex = [], []
    for s in stems:
        m = re.search(r"([a-z]+)(\d+)-([a-z]+)(\d+)", str(s))
        if m is None:
            raise ValueError(f"unparseable stem {s!r}")
        obj.append(m.group(1) + m.group(2))
        tex.append(m.group(3) + m.group(4))
    return np.array(obj), np.array(tex)


def exemplar_folds(labels: np.ndarray, exemplars: np.ndarray, rng, n_folds: int = 3) -> np.ndarray:
    """Per image, the fold of its exemplar: each category's exemplars permuted, fold = position mod n_folds."""
    fold = np.empty(len(labels), int)
    for c in np.unique(labels):
        ex = rng.permutation(sorted(set(exemplars[labels == c])))
        for i, e in enumerate(ex):
            fold[exemplars == e] = i % n_folds
    return fold


# ---------------------------------------------------------------------------------------------- decisions
def decisions(scorer: Scorer, logits: np.ndarray, s_idx: np.ndarray, t_idx: np.ndarray):
    """Per-image (shape hit, texture hit, forced shape) under the scorer's decision rule."""
    r = scorer._restricted(np.asarray(logits, dtype=np.float64))
    dec = r.argmax(1)
    rows = np.arange(len(s_idx))
    return dec == s_idx, dec == t_idx, r[rows, s_idx] > r[rows, t_idx]


def rates(hs, ht, fo):
    n = hs.sum() + ht.sum()
    return (float(hs.sum() / n) if n else float("nan")), float(fo.mean())


# ------------------------------------------------------------------------------------------------- spans
def curve_on(y, W, b, B_S, B_T, idx, scorer, s_idx, t_idx):
    """Per-w per-image decisions on rows idx: list over WS of (hs, ht, fo)."""
    out = []
    for w in WS:
        L = gain_features(y[idx], B_S, B_T, float(w)) @ W.T + b
        out.append(decisions(scorer, L, s_idx[idx], t_idx[idx]))
    return out


def pooled_curve(parts):
    """Rates per w over the union of several folds' decisions."""
    prim, forc = [], []
    for j in range(len(WS)):
        hs = np.concatenate([p[j][0] for p in parts])
        ht = np.concatenate([p[j][1] for p in parts])
        fo = np.concatenate([p[j][2] for p in parts])
        a, f = rates(hs, ht, fo)
        prim.append(a); forc.append(f)
    return prim, forc


def summarise(curves: list[list[float]], gate) -> dict:
    """Span as in subspace_factor.summarise (mean of per-curve max - min) and, as in the manuscript's
    tables, the gated span of the mean curve (max - min inside the gate; shape end = largest gated w)."""
    M = np.array(curves)
    spans = M.max(1) - M.min(1)
    rhos = np.array([spearmanr(WS, row).correlation for row in M])
    mc = M.mean(0)
    rho_mc = float(spearmanr(WS, mc).correlation)
    m = (WS >= min(gate) - 1e-9) & (WS <= max(gate) + 1e-9)
    c = mc[m]
    return dict(mean_curve=mc.tolist(), span_mean=float(spans.mean()), span_min=float(spans.min()),
                span_max=float(spans.max()), rho_mean=float(np.nanmean(rhos)), rho_min=float(np.nanmin(rhos)),
                rho_mean_curve=rho_mc, gated_span=float(c.max() - c.min()), shape_end=float(c[-1]),
                texture_end=float(c[0]))


def span_E0(y, W, b, shapes, textures, scorer, s_idx, t_idx, gate, n_splits=10, seed=0):
    """The paper's held-out estimator: pair-stratified halves, same RNG path as subspace_factor.main."""
    rng = np.random.default_rng(seed)
    prim, forc = [], []
    for _ in range(n_splits):
        A, Bh = stratified_halves(shapes, textures, rng)
        for fit, ev in ((A, Bh), (Bh, A)):
            B_S, B_T, _ = subspace_bases(y, shapes, textures, fit)
            p, f = pooled_curve([curve_on(y, W, b, B_S, B_T, ev, scorer, s_idx, t_idx)])
            prim.append(p); forc.append(f)
    return dict(primary=summarise(prim, gate), forced=summarise(forc, gate), n_curves=len(prim))


def span_folds(y, W, b, shapes, textures, obj, tex, scorer, s_idx, t_idx, gate, mode, n_reps=N_REPS):
    """E1 ('both'), E2 ('texture'), E3 ('object'): 3 exemplar folds, pooled per repetition."""
    prim, forc, n_fit, n_eval = [], [], [], []
    for r in range(n_reps):
        rng = np.random.default_rng(r)
        tf = exemplar_folds(textures, tex, rng)
        of = exemplar_folds(shapes, obj, rng)
        parts = []
        for k in range(3):
            if mode == "both":
                ev = np.flatnonzero((tf == k) & (of == k)); fit = np.flatnonzero((tf != k) & (of != k))
            elif mode == "texture":
                ev = np.flatnonzero(tf == k); fit = np.flatnonzero(tf != k)
            elif mode == "object":
                ev = np.flatnonzero(of == k); fit = np.flatnonzero(of != k)
            else:
                raise ValueError(mode)
            B_S, B_T, _ = subspace_bases(y, shapes, textures, fit)
            parts.append(curve_on(y, W, b, B_S, B_T, ev, scorer, s_idx, t_idx))
            n_fit.append(len(fit)); n_eval.append(len(ev))
        p, f = pooled_curve(parts)
        prim.append(p); forc.append(f)
    return dict(primary=summarise(prim, gate), forced=summarise(forc, gate), n_reps=n_reps,
                n_fit_mean=float(np.mean(n_fit)), n_eval_pooled_mean=float(np.mean(n_eval) * 3))


# ------------------------------------------------------------------------------------------------ probes
def _fit_score(Xtr, ytr, Xte, yte, pca_inside: bool, k_max: int = 256, std: bool = True) -> float:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    if pca_inside:
        k = int(min(k_max, Xtr.shape[0] - 1, Xtr.shape[1]))
        pca = PCA(n_components=k, random_state=0).fit(Xtr)
        Xtr, Xte = pca.transform(Xtr), pca.transform(Xte)
    if std:
        sc = StandardScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(Xtr, ytr)
    return float(clf.score(Xte, yte))


def _loeo(X, y, groups, fold, pca_inside, std=True):
    return float(np.mean([_fit_score(X[fold != k], y[fold != k], X[fold == k], y[fold == k], pca_inside, std=std)
                          for k in range(3)]))


def probe_Q0(y, shapes, textures, seed=0):
    """The paper's probe exactly (probe_bias.py): PCA-256 on all images, StratifiedKFold(5), LR C=1."""
    from sklearn.decomposition import PCA
    from src.analysis.behavior.stream_shape_texture import probe
    k = int(min(256, y.shape[0] - 1, y.shape[1]))
    zc = PCA(n_components=k, random_state=seed).fit_transform(y)
    a_s, _ = probe(zc, list(shapes), seed=seed)
    a_t, _ = probe(zc, list(textures), seed=seed)
    return dict(shape_acc=float(a_s), texture_acc=float(a_t), bias=float(a_s - a_t))


def probe_Q1(y, shapes, textures, obj, tex, pca_inside: bool, n_draws=N_DRAWS, k_all: int = 256, std: bool = True):
    """Exemplar-balanced: 3 exemplars per category for both labels, leave-one-exemplar-out,
    texture images subsampled per category to the shape subset's count. pca_inside=False fits PCA
    (k_all components) once on all images; std toggles the per-fold StandardScaler."""
    X_all = y
    if not pca_inside:
        from sklearn.decomposition import PCA
        X_all = PCA(n_components=int(min(k_all, y.shape[0] - 1, y.shape[1])), random_state=0).fit_transform(y)
    S, T = [], []
    for d in range(n_draws):
        rng = np.random.default_rng(d)
        keep = np.zeros(len(shapes), bool)
        for c in np.unique(shapes):
            ex = rng.choice(sorted(set(obj[shapes == c])), N_EX_BAL, replace=False)
            keep |= np.isin(obj, ex)
        fs = exemplar_folds(shapes[keep], obj[keep], rng)
        S.append(_loeo(X_all[keep], shapes[keep], obj[keep], fs, pca_inside, std))
        tk = np.zeros(len(textures), bool)
        for c in np.unique(textures):
            n = int((keep & (shapes == c)).sum()); idx = np.flatnonzero(textures == c)
            tk[rng.choice(idx, min(n, len(idx)), replace=False)] = True
        ft = exemplar_folds(textures[tk], tex[tk], rng)
        T.append(_loeo(X_all[tk], textures[tk], tex[tk], ft, pca_inside, std))
    S, T = np.array(S), np.array(T)
    D = S - T
    return dict(shape_acc=float(S.mean()), texture_acc=float(T.mean()), bias=float(D.mean()),
                bias_p05=float(np.percentile(D, 5)), bias_p95=float(np.percentile(D, 95)),
                bias_draws=D.tolist(), n_draws=n_draws, pca_inside=pca_inside, k_all=k_all, std=std)


def probe_Q2(y, shapes, textures, obj, tex, seed=0):
    """Exemplar-grouped StratifiedGroupKFold(5), PCA inside the folds."""
    from sklearn.model_selection import StratifiedGroupKFold
    out = {}
    for name, lab, grp in (("shape", shapes, obj), ("texture", textures, tex)):
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
        acc = [_fit_score(y[tr], lab[tr], y[te], lab[te], True) for tr, te in cv.split(y, lab, grp)]
        out[f"{name}_acc"] = float(np.mean(acc))
    out["bias"] = out["shape_acc"] - out["texture_acc"]
    return out


def all_probes(y, shapes, textures, obj, tex) -> dict:
    """Q0-Q2 as pre-registered, plus two post-hoc preprocessing sensitivities of the balanced design
    (added after the one-network test runs showed Q1b, PCA-256 on all images then standardised, to be the
    outlier; see the plan's note of 2026-09-21): S64, PCA-64 on all images, standardised; S256n, PCA-256 on
    all images, not standardised."""
    return dict(Q0=probe_Q0(y, shapes, textures), Q1=probe_Q1(y, shapes, textures, obj, tex, True),
                Q1b=probe_Q1(y, shapes, textures, obj, tex, False), Q2=probe_Q2(y, shapes, textures, obj, tex),
                S64=probe_Q1(y, shapes, textures, obj, tex, False, k_all=64),
                S256n=probe_Q1(y, shapes, textures, obj, tex, False, std=False))


# -------------------------------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True)
    p.add_argument("--behaviour-dir", required=True, help="holds <tag>_subspace.json and the gate")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--skip-probes", action="store_true")
    args = p.parse_args()
    t0 = time.time()
    os.makedirs(args.out_dir, exist_ok=True)

    z = np.load(args.features, allow_pickle=True)
    tag, kind = str(z["tag"]), str(z["scorer"])
    y = z["features"].astype(np.float64)
    W, b = z["W"].astype(np.float64), z["b"].astype(np.float64)
    shapes = np.array([str(s) for s in z["shapes"]]); textures = np.array([str(t) for t in z["textures"]])
    obj, tex = parse_exemplars(z["stems"])
    scorer = Scorer(kind)
    s_idx = np.array([scorer.cat_of[s] for s in shapes]); t_idx = np.array([scorer.cat_of[t] for t in textures])
    print(f"[exemplar] {tag}: {y.shape[0]} images, {y.shape[1]} units, scorer {kind}; "
          f"{len(set(obj))} object / {len(set(tex))} texture exemplars", flush=True)

    f_gate = os.path.join(args.behaviour_dir, f"{tag}_gain_accuracy_subspace.json")
    gate = json.load(open(f_gate))["w_within_5pts"] if os.path.exists(f_gate) else [0.0, 1.0]
    gate = [float(min(gate)), float(max(gate))]

    # trained readout, every image
    hs, ht, fo = decisions(scorer, y @ W.T + b, s_idx, t_idx)
    trained, trained_forced = rates(hs, ht, fo)

    # E0 must reproduce the stored held-out span
    stored = json.load(open(os.path.join(args.behaviour_dir, f"{tag}_subspace.json")))["summary"]
    e0 = span_E0(y, W, b, shapes, textures, scorer, s_idx, t_idx, gate)
    d0 = abs(e0["primary"]["span_mean"] - stored["primary"]["span_mean"])
    print(f"  E0 span {e0['primary']['span_mean']:.6f} vs stored {stored['primary']['span_mean']:.6f} "
          f"(|d| = {d0:.1e}); gated {e0['primary']['gated_span']:.3f}  {time.time() - t0:.0f}s", flush=True)
    if d0 > 1e-9:
        raise SystemExit(f"E0 does not reproduce the stored span for {tag}: |d| = {d0}")

    spans = dict(E0=e0)
    for name, mode in (("E1", "both"), ("E2", "texture"), ("E3", "object")):
        spans[name] = span_folds(y, W, b, shapes, textures, obj, tex, scorer, s_idx, t_idx, gate, mode)
        s = spans[name]
        print(f"  {name} ({mode}): span {s['primary']['span_mean']:.3f} gated {s['primary']['gated_span']:.3f} "
              f"forced {s['forced']['span_mean']:.3f} rho(mean curve) {s['primary']['rho_mean_curve']:.2f} "
              f"n_fit {s['n_fit_mean']:.0f} n_eval {s['n_eval_pooled_mean']:.0f}  {time.time() - t0:.0f}s", flush=True)

    out = dict(tag=tag, scorer=kind, n_images=int(y.shape[0]), n_units=int(y.shape[1]), gate=gate,
               trained=trained, trained_forced=trained_forced, stored_span_E0=stored["primary"]["span_mean"],
               spans=spans, plan="docs/PLAN_EXEMPLAR_2026-09-21.md")

    if not args.skip_probes:
        out["probes"] = all_probes(y, shapes, textures, obj, tex)
        f_q0 = os.path.join(args.behaviour_dir, f"{tag}_probe_bias.json")
        if os.path.exists(f_q0):
            q0s = json.load(open(f_q0))["probe_bias"]
            dq = abs(out["probes"]["Q0"]["bias"] - q0s)
            out["stored_probe_Q0"] = q0s
            print(f"  Q0 bias {out['probes']['Q0']['bias']:+.4f} vs stored {q0s:+.4f} (|d| = {dq:.1e})", flush=True)
            if dq > 1e-9:
                raise SystemExit(f"Q0 does not reproduce the stored probe for {tag}: |d| = {dq}")
        q = out["probes"]
        print(f"  probes: Q0 {q['Q0']['bias']:+.3f} | Q1 {q['Q1']['bias']:+.3f} [{q['Q1']['bias_p05']:+.3f}, "
              f"{q['Q1']['bias_p95']:+.3f}] (shape {q['Q1']['shape_acc']:.3f} tex {q['Q1']['texture_acc']:.3f}) | "
              f"Q1b {q['Q1b']['bias']:+.3f} | Q2 {q['Q2']['bias']:+.3f} | S64 {q['S64']['bias']:+.3f} | S256n {q['S256n']['bias']:+.3f}  "
              f"{time.time() - t0:.0f}s", flush=True)
        if "stream_of_unit" in z.files:
            su = z["stream_of_unit"]
            out["stream_probes"] = {name: all_probes(y[:, su == k], shapes, textures, obj, tex)
                                    for name, k in (("stream1", 0), ("stream2", 1))}
            for name, v in out["stream_probes"].items():
                print(f"  {name}: Q0 {v['Q0']['bias']:+.3f} Q1 {v['Q1']['bias']:+.3f} Q2 {v['Q2']['bias']:+.3f}", flush=True)

    json.dump(out, open(os.path.join(args.out_dir, f"{tag}_exemplar.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(args.out_dir, f"{tag}_exemplar.npz"), tag=tag, hit_shape=hs, hit_texture=ht,
                        forced_shape=fo, object_exemplar=obj, texture_exemplar=tex)
    print(f"[exemplar] wrote {args.out_dir}/{tag}_exemplar.{{json,npz}} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
