"""What sets the span: the per-image evidence the gain acts on, and a random-direction null.

  python -m src.analysis.generality.span_model --features-dir FEAT --out-dir BEH   # whole zoo

`docs/PLAN_SPAN_2026-09-18.md` (predictions S-1 to S-7). For every network, at the trained
readout (w = 1/2):

- the forced two-way margin of each cue-conflict image, Delta = log p_shape - log p_texture,
  with p_c the Geirhos mean probability over category c's ImageNet classes;
- its derivative with respect to the logits, u = q_shape - q_texture (softmax within each
  category; the global normaliser cancels), and the evidence the head receives through each
  subspace, e_S = u . W P_S y and e_T = u . W P_T y; the leverage g = e_S - e_T is the
  first-order change of Delta per unit of s = 2w - 1, since the gain adds s W (P_S - P_T) y
  to the logits exactly;
- the first-order curve F0(w) = mean 1{Delta + (2w - 1) g > 0} (model M0) and the two-moment
  curve Phi(z + (2w - 1) lambda), z = mean(Delta)/sd(Delta), lambda = mean(g)/sd(Delta) (M1);
- the decision-free head-weighted subspace share H, on the cue-conflict and on the ImageNet
  validation features;
- a null: ten pairs of random orthonormal 15-dimensional subspaces inside the network's own
  256-dimensional PCA space of the cue-conflict features, under the same gain, estimators and
  5-point top-1 gate.

Nothing is refitted: features, head, bases and gate are those of Methods 4.8 and of the kappa
sweep. Writes <tag>_span_model.json and <tag>_span_model.npz (per-image Delta, e_S, e_T).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

from src.analysis.generality.scoring import Scorer
from src.analysis.generality.subspace_factor import N_DIR, orthonormal, pca_basis

WS = [round(0.05 * i, 2) for i in range(21)]
N_RANDOM = 10
K_PCA = 256


def category_margin(logits, scorer, s_idx, t_idx):
    """Delta = log p_s - log p_t under the Geirhos rule, and u = dDelta/dlogits (N x 1000)."""
    z = logits - logits.max(1, keepdims=True)
    n, c = z.shape
    lse = np.stack([logsumexp(z[:, m], 1) - np.log(len(m)) for m in scorer.members], 1)   # log p_c + log Z
    rows = np.arange(n)
    delta = lse[rows, s_idx] - lse[rows, t_idx]
    u = np.zeros((n, c))
    for i in range(n):
        ms, mt = scorer.members[s_idx[i]], scorer.members[t_idx[i]]
        qs = np.exp(z[i, ms] - logsumexp(z[i, ms]))
        qt = np.exp(z[i, mt] - logsumexp(z[i, mt]))
        u[i, ms] += qs
        u[i, mt] -= qt
    return delta, u


def head_share(X, A, B_S, B_T):
    """Fraction of the head's category-contrast variance tr(A Sigma A^T) carried by each subspace."""
    Xc = X - X.mean(0)
    tot = float(np.sum((Xc @ A.T) ** 2))
    s = float(np.sum((((Xc @ B_S) @ B_S.T) @ A.T) ** 2))
    t = float(np.sum((((Xc @ B_T) @ B_T.T) @ A.T) ** 2))
    return dict(S=s / tot, T=t / tot, H=(s + t) / tot)


def gated_span(curve, top1):
    """The gate of Methods 4.8 (top-1 within 5 points of w = 1/2) and the rate change across it."""
    t0 = top1[10]
    inside = [i for i, a in enumerate(top1) if a >= t0 - 0.05]
    lo, hi = min(inside), max(inside)
    return [WS[lo], WS[hi]], float(curve[hi] - curve[lo])


def run_one(cc_path, val_path, fac_path, gate_path, kappa_path, out_dir):
    t0 = time.time()
    z = np.load(cc_path, allow_pickle=True)
    tag, kind = str(z["tag"]), str(z["scorer"])
    scorer = Scorer(kind)
    shapes = [str(s) for s in z["shapes"]]
    textures = [str(t) for t in z["textures"]]
    s_idx = np.array([scorer.cat_of[s] for s in shapes])
    t_idx = np.array([scorer.cat_of[t] for t in textures])
    y = z["features"].astype(np.float64)
    W, b = z["W"].astype(np.float64), z["b"].astype(np.float64)
    fac = np.load(fac_path, allow_pickle=True)
    B_S, B_T = fac["B_S"].astype(np.float64), fac["B_T"].astype(np.float64)
    gate = json.load(open(gate_path))
    ksw = json.load(open(kappa_path))
    forced_measured = [r["forced"] for r in ksw["sweep"]]
    rate_measured = [r["rate"] for r in ksw["sweep"]]

    # the trained readout: margins, the gradient, and the evidence through each subspace
    L = y @ W.T + b
    delta, u = category_margin(L, scorer, s_idx, t_idx)
    f_trained = float(np.mean(delta > 0))
    # the same decision as the stored sweep's p_s > p_t, up to floating-point ties (at most one image)
    assert abs(f_trained - forced_measured[10]) <= 1.5 / len(y), (tag, f_trained, forced_measured[10])
    PS_y, PT_y = (y @ B_S) @ B_S.T, (y @ B_T) @ B_T.T
    e_S = np.sum(u * (PS_y @ W.T), 1)
    e_T = np.sum(u * (PT_y @ W.T), 1)
    g = e_S - e_T
    mu, sd = float(delta.mean()), float(delta.std())
    zloc, lam = mu / sd, float(g.mean()) / sd

    m0 = [float(np.mean(delta + (2 * w - 1) * g > 0)) for w in WS]
    m1 = [float(norm.cdf(zloc + (2 * w - 1) * lam)) for w in WS]

    # exact margins along the sweep (diagnostic only; the model is first order by the plan)
    Ld = (PS_y - PT_y) @ W.T
    exact = []
    for w in (0.0, 1.0):
        d_w, _ = category_margin(L + (2 * w - 1) * Ld, scorer, s_idx, t_idx)
        exact.append(dict(w=w, mean=float(d_w.mean()), sd=float(d_w.std()), forced=float(np.mean(d_w > 0)),
                          corr_with_first_order=float(np.corrcoef(d_w, delta + (2 * w - 1) * g)[0, 1])))

    # the decision-free structural share: category-contrast rows of the head
    C = np.zeros((16, W.shape[0]))
    for c, m in enumerate(scorer.members):
        C[c, m] = 1.0 / len(m)
    A = (np.eye(16) - 1.0 / 16) @ C @ W
    v = np.load(val_path, allow_pickle=True)
    yv, lab = v["features"].astype(np.float64), v["labels"].astype(int)
    H_cc = head_share(y, A, B_S, B_T)
    H_val = head_share(yv, A, B_S, B_T)

    print(f"[span_model] {tag}: margins, leverage and H in {time.time() - t0:.0f}s", flush=True)
    # random-direction null inside the network's own PCA space of the cue-conflict features
    k = int(min(K_PCA, y.shape[1], len(y) - N_DIR - 2))
    V, _ = pca_basis(y, k)
    Lv = yv @ W.T + b
    top1_trained = float(np.mean(np.argmax(Lv, 1) == lab))
    rand = []
    for seed in range(N_RANDOM):
        rng = np.random.default_rng(seed)
        Q = orthonormal(rng.standard_normal((k, 2 * N_DIR)))
        R_S, R_T = V @ Q[:, :N_DIR], V @ Q[:, N_DIR:]
        Ldr = (((y @ R_S) @ R_S.T) - ((y @ R_T) @ R_T.T)) @ W.T
        Ldv = (((yv @ R_S) @ R_S.T) - ((yv @ R_T) @ R_T.T)) @ W.T
        rate, forced, top1 = [], [], []
        for w in WS:
            s = 2 * w - 1
            sc = scorer.score(L + s * Ldr, shapes, textures)
            rate.append(sc["shape_bias"]); forced.append(sc["forced_two_way"])
            top1.append(float(np.mean(np.argmax(Lv + s * Ldv, 1) == lab)))
        gw, span = gated_span(rate, top1)
        _, fspan = gated_span(forced, top1)
        rand.append(dict(seed=seed, rate=rate, forced=forced, top1=top1, gate=gw, span=span, forced_span=fspan,
                         span_ungated=float(rate[-1] - rate[0])))
        print(f"[span_model] {tag} random {seed}: gate {gw[0]:.2f}-{gw[1]:.2f} span {span:+.3f} "
              f"(forced {fspan:+.3f}, ungated {rate[-1] - rate[0]:+.3f}) {time.time() - t0:.0f}s", flush=True)

    g_lo, g_hi = float(min(gate["w_within_5pts"])), float(max(gate["w_within_5pts"]))
    i_lo, i_hi = WS.index(round(g_lo, 2)), WS.index(round(g_hi, 2))
    out = dict(
        tag=tag, scorer=kind, n_images=len(y), n_units=int(y.shape[1]), k_pca=k,
        gate=[g_lo, g_hi], top1_trained_val=top1_trained, top1_trained_stored=float(gate["trained_top1"]),
        forced_trained=f_trained, rate_trained=rate_measured[10],
        mu=mu, sd=sd, z=zloc, lam=lam, Phi_z=float(norm.cdf(zloc)),
        mean_eS=float(e_S.mean()), mean_eT=float(e_T.mean()), mean_g=float(g.mean()), sd_g=float(g.std()),
        corr_delta_g=float(np.corrcoef(delta, g)[0, 1]),
        forced_curve=forced_measured, rate_curve=rate_measured, m0_curve=m0, m1_curve=m1,
        T_f=float(forced_measured[i_hi] - forced_measured[i_lo]),
        m0_span=float(m0[i_hi] - m0[i_lo]), m1_span=float(m1[i_hi] - m1[i_lo]),
        m0_max_abs_err=float(np.max(np.abs(np.array(m0) - np.array(forced_measured)))),
        m1_max_abs_err=float(np.max(np.abs(np.array(m1) - np.array(forced_measured)))),
        exact_end_margins=exact, H_cc=H_cc, H_val=H_val, random=rand,
    )
    os.makedirs(out_dir, exist_ok=True)
    dest = os.path.join(out_dir, f"{tag}_span_model.json")
    json.dump(out, open(dest, "w"), indent=1)
    np.savez_compressed(os.path.join(out_dir, f"{tag}_span_model.npz"), delta=delta, e_S=e_S, e_T=e_T)
    print(f"[span_model] {tag}: z {zloc:+.3f} lambda {lam:.3f} | T_f {out['T_f']:.3f} M0 {out['m0_span']:.3f} "
          f"M1 {out['m1_span']:.3f} | H_cc {H_cc['H']:.3f} | M0 max err {out['m0_max_abs_err']:.3f}", flush=True)
    return dest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features-dir", required=True)
    p.add_argument("--out-dir", required=True, help="the behaviour directory holding the stored bases, gates and kappa sweeps")
    p.add_argument("--only", default=None, help="comma-separated tags (testing)")
    args = p.parse_args()
    only = set(args.only.split(",")) if args.only else None
    done = 0
    for cc in sorted(glob.glob(os.path.join(args.features_dir, "*_cue_conflict.npz"))):
        tag = os.path.basename(cc)[: -len("_cue_conflict.npz")]
        if only and tag not in only:
            continue
        paths = [os.path.join(args.features_dir, f"{tag}_imagenet_val.npz")] + [
            os.path.join(args.out_dir, f"{tag}{s}") for s in
            ("_subspace.npz", "_gain_accuracy_subspace.json", "_kappa_sweep.json")]
        missing = [q for q in paths if not os.path.exists(q)]
        if missing:
            raise SystemExit(f"{tag}: missing {missing}")
        run_one(cc, *paths, args.out_dir)
        done += 1
    print(f"[span_model] {done} networks written", flush=True)


if __name__ == "__main__":
    main()
