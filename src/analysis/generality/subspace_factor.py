"""Shape- and texture-predictive subspaces and the fixed-head subspace gain sweep
(PLAN_SUBSPACE D1-D2, pre-registered 2026-09-15).

  python -m src.analysis.generality.subspace_factor --features DIR/<tag>_cue_conflict.npz --out-dir DIR

D1: on the fitting images, centre, PCA to k = min(256, D, n_fit - 17), a 16-class LDA
(eigen solver, Ledoit-Wolf shrinkage) for shape and one for texture, 15 discriminant
directions each mapped back to unit space and orthonormalised, then the two bases made
mutually orthogonal by symmetric (Lowdin) orthogonalisation: B_S, B_T (D x 15 each),
P_R = I - P_S - P_T.
D2: y(w) = P_R y + 2w P_S y + 2(1-w) P_T y through the network's own head; w = 0.5 is the
trained network. Held-out: bases fit on one stratified half of the 1,200 cue-conflict
images (the same splits as unit_factor.py), scored on the other, 10 splits both
directions. The all-image bases (for cortex and the accuracy gate) are written to
<tag>_subspace.npz; per-split summaries and the sequential-order sensitivity to
<tag>_subspace.json.
"""
from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np
from scipy.stats import spearmanr

from src.analysis.generality.scoring import Scorer
from src.analysis.generality.unit_factor import WS, stratified_halves

N_DIR = 15


def pca_basis(Yfit: np.ndarray, k: int):
    """Centred PCA basis V (D x k) and the mean; fit on the fitting rows only."""
    mu = Yfit.mean(0)
    Xc = Yfit - mu
    # thin SVD of the centred data; right singular vectors are the PCs
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    return vt[:k].T, mu


def lda_directions(Z: np.ndarray, labels: np.ndarray, n_dir: int = N_DIR) -> np.ndarray:
    """The leading discriminant directions (k x n_dir) of a shrinkage LDA in PCA coordinates."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto", n_components=n_dir)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lda.fit(Z, labels)
    A = np.asarray(lda.scalings_)[:, :n_dir]
    return A.astype(np.float64)


def orthonormal(U: np.ndarray) -> np.ndarray:
    Q, R = np.linalg.qr(U)
    return Q * np.sign(np.diag(R))[None, :]  # sign-fixed so directions keep their orientation


def lowdin(U_S: np.ndarray, U_T: np.ndarray):
    """Symmetric orthogonalisation of [U_S, U_T]; returns B_S, B_T and the alignment cosines."""
    U = np.concatenate([U_S, U_T], 1)
    M = U.T @ U
    evals, evecs = np.linalg.eigh(M)
    if evals.min() <= 1e-10:
        raise ValueError("shape and texture bases are not jointly full rank")
    M_inv_sqrt = evecs @ np.diag(evals ** -0.5) @ evecs.T
    B = U @ M_inv_sqrt
    k = U_S.shape[1]
    B_S, B_T = B[:, :k], B[:, k:]
    align = dict(S=float(np.mean(np.abs(np.sum(U_S * B_S, 0)))), T=float(np.mean(np.abs(np.sum(U_T * B_T, 0)))))
    return B_S, B_T, align


def gram_schmidt_seq(U_first: np.ndarray, U_second: np.ndarray):
    """Sequential orthogonalisation: the first basis kept, the second projected off it."""
    P = U_first @ U_first.T
    V = U_second - P @ U_second
    return U_first, orthonormal(V)


def subspace_bases(y: np.ndarray, shapes, textures, fit_idx, k_pca: int = 256, n_dir: int = N_DIR, method: str = "lowdin"):
    """B_S, B_T (D x n_dir, orthonormal, mutually orthogonal) fit on rows fit_idx; plus descriptives."""
    Yfit = y[fit_idx]
    k = int(min(k_pca, y.shape[1], len(fit_idx) - n_dir - 2))
    V, mu = pca_basis(Yfit, k)
    Z = (Yfit - mu) @ V
    A_S = lda_directions(Z, shapes[fit_idx], n_dir)
    A_T = lda_directions(Z, textures[fit_idx], n_dir)
    U_S, U_T = orthonormal(V @ A_S), orthonormal(V @ A_T)
    overlap = float(np.mean(np.linalg.svd(U_S.T @ U_T, compute_uv=False)))  # mean cosine of principal angles
    if method == "lowdin":
        B_S, B_T, align = lowdin(U_S, U_T)
    elif method == "S_first":
        B_S, B_T = gram_schmidt_seq(U_S, U_T); align = dict(S=1.0, T=float(np.mean(np.abs(np.sum(U_T * B_T, 0)))))
    elif method == "T_first":
        B_T, B_S = gram_schmidt_seq(U_T, U_S); align = dict(T=1.0, S=float(np.mean(np.abs(np.sum(U_S * B_S, 0)))))
    else:
        raise ValueError(method)
    info = dict(k_pca=k, n_dir=n_dir, method=method, overlap_mean_cos=overlap, align=align)
    return B_S, B_T, info


def gain_features(y: np.ndarray, B_S: np.ndarray, B_T: np.ndarray, w: float) -> np.ndarray:
    """y(w) = P_R y + 2w P_S y + 2(1-w) P_T y = y + (2w-1) P_S y + (1-2w) P_T y."""
    return y + (2 * w - 1) * ((y @ B_S) @ B_S.T) + (1 - 2 * w) * ((y @ B_T) @ B_T.T)


def gain_sweep(y, W, b, B_S, B_T, scorer: Scorer, shapes, textures) -> list[dict]:
    rows = []
    for w in WS:
        logits = gain_features(y, B_S, B_T, float(w)) @ W.T + b
        rows.append(dict(w=float(w), **scorer.score(logits, shapes, textures)))
    return rows


def summarise(held: list[dict], key: str) -> dict:
    M = np.array([[r[key] for r in h["curve"]] for h in held])
    spans = M.max(1) - M.min(1)
    rhos = np.array([spearmanr(WS, row).correlation for row in M])
    return dict(mean_curve=M.mean(0).tolist(), min_curve=M.min(0).tolist(), max_curve=M.max(0).tolist(),
                span_mean=float(spans.mean()), span_min=float(spans.min()), span_max=float(spans.max()),
                rho_mean=float(np.nanmean(rhos)), rho_min=float(np.nanmin(rhos)),
                rate_w0=float(M[:, 0].mean()), rate_w05=float(M[:, 10].mean()), rate_w1=float(M[:, -1].mean()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-splits", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k-pca", type=int, default=256)
    p.add_argument("--n-dir", type=int, default=N_DIR)
    p.add_argument("--out-suffix", default="")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    z = np.load(args.features, allow_pickle=True)
    tag, kind = str(z["tag"]), str(z["scorer"])
    y = z["features"].astype(np.float64)
    W, b = z["W"].astype(np.float64), z["b"].astype(np.float64)
    shapes, textures = z["shapes"], z["textures"]
    scorer = Scorer(kind)
    N, D = y.shape
    print(f"[subspace] {tag}: {N} images, {D} units, scorer {kind}", flush=True)

    trained = scorer.score(y @ W.T + b, shapes, textures)
    print(f"  trained: shape bias {trained['shape_bias']:.3f} forced {trained['forced_two_way']:.3f}", flush=True)

    # all-image bases (cortex, accuracy gate) + descriptives
    all_idx = np.arange(N)
    B_S, B_T, info = subspace_bases(y, shapes, textures, all_idx, args.k_pca, args.n_dir, "lowdin")
    yc = y - y.mean(0)
    tot = float((yc ** 2).sum())
    desc = dict(**info, var_frac_S=float(((yc @ B_S) ** 2).sum() / tot), var_frac_T=float(((yc @ B_T) ** 2).sum() / tot))
    desc["var_frac_R"] = 1.0 - desc["var_frac_S"] - desc["var_frac_T"]
    stream_of_unit = z["stream_of_unit"] if "stream_of_unit" in z.files else None
    if stream_of_unit is not None:
        s1 = stream_of_unit == 0
        desc["energy_S_on_stream1_full"] = float((B_S[s1] ** 2).sum() / B_S.shape[1])
        desc["energy_T_on_stream2_full"] = float((B_T[~s1] ** 2).sum() / B_T.shape[1])
    insample = gain_sweep(y, W, b, B_S, B_T, scorer, shapes, textures)
    print(f"  all-image bases: k_pca {info['k_pca']}, overlap cos {info['overlap_mean_cos']:.2f}, "
          f"align S {info['align']['S']:.2f} T {info['align']['T']:.2f}; var frac S {desc['var_frac_S']:.3f} T {desc['var_frac_T']:.3f}", flush=True)

    # held-out sweeps: primary (Lowdin) + sequential-order sensitivity
    rng = np.random.default_rng(args.seed)
    held = {m: [] for m in ("lowdin", "S_first", "T_first")}
    e_s1 = []
    for k in range(args.n_splits):
        A, Bh = stratified_halves(shapes, textures, rng)
        for name, fit, ev in (("A>B", A, Bh), ("B>A", Bh, A)):
            for m in held:
                bs, bt, inf = subspace_bases(y, shapes, textures, fit, args.k_pca, args.n_dir, m)
                curve = gain_sweep(y[ev], W, b, bs, bt, scorer, shapes[ev], textures[ev])
                held[m].append(dict(split=k, direction=name, n_eval=int(len(ev)), curve=curve, info=inf))
                if m == "lowdin" and stream_of_unit is not None:
                    e_s1.append(float((bs[stream_of_unit == 0] ** 2).sum() / bs.shape[1]))
            if k == 0:
                c = held["lowdin"][-1]["curve"]
                print(f"  split {k} {name}: rate at w=0 {c[0]['shape_bias']:.3f}, w=0.5 {c[10]['shape_bias']:.3f}, w=1 {c[-1]['shape_bias']:.3f}", flush=True)

    summary = dict(tag=tag, scorer=kind, n_images=int(N), n_units=int(D), n_splits=args.n_splits, ws=WS.tolist(),
                   trained=trained, trained_score_stored=json.loads(str(z["trained_score"])), descriptives=desc,
                   primary=summarise(held["lowdin"], "shape_bias"), forced=summarise(held["lowdin"], "forced_two_way"),
                   sensitivity={m: dict(primary=summarise(held[m], "shape_bias"), forced=summarise(held[m], "forced_two_way"))
                                for m in ("S_first", "T_first")},
                   decided_mean_curve=np.array([[r["n_shape"] + r["n_texture"] for r in h["curve"]] for h in held["lowdin"]]).mean(0).tolist(),
                   insample_primary=[r["shape_bias"] for r in insample], insample_forced=[r["forced_two_way"] for r in insample])
    if e_s1:
        summary["descriptives"]["energy_S_on_stream1_heldout_mean"] = float(np.mean(e_s1))
        summary["descriptives"]["energy_S_on_stream1_heldout_min"] = float(np.min(e_s1))
    pr = summary["primary"]
    print(f"  held-out span primary {pr['span_mean']:.3f} [{pr['span_min']:.3f}, {pr['span_max']:.3f}], rho {pr['rho_mean']:.2f} "
          f"(min {pr['rho_min']:.2f}); forced span {summary['forced']['span_mean']:.3f}; "
          f"S_first {summary['sensitivity']['S_first']['primary']['span_mean']:.3f} T_first {summary['sensitivity']['T_first']['primary']['span_mean']:.3f}", flush=True)

    out_json = os.path.join(args.out_dir, f"{tag}_subspace{args.out_suffix}.json")
    json.dump(dict(summary=summary, held_out=held["lowdin"]), open(out_json, "w"), indent=1)
    np.savez(os.path.join(args.out_dir, f"{tag}_subspace{args.out_suffix}.npz"), tag=tag, B_S=B_S, B_T=B_T,
             k_pca=info["k_pca"], n_dir=args.n_dir, method="lowdin", ws=WS)
    print(f"[subspace] wrote {args.out_dir}/{tag}_subspace{args.out_suffix}.{{json,npz}}", flush=True)


if __name__ == "__main__":
    main()
