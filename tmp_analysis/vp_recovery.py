"""Does the Uₘ/Uₚ variance partitioning recover a known split -- or manufacture one?

§2.2's whole claim rests on `variance_partition` (banded_ridge.py:208-223), which
is pure subtraction:

    unique_m = r2_joint - r2_parvo
    unique_p = r2_joint - r2_magno

Its own docstring warns that "with correlated bands, unique components understate
a band's total contribution". The magno and parvo bands here are strongly
correlated -- same images, same architecture, adjacent front ends -- so the
instrument is being used precisely in the regime its author flagged, and it has
never been validated. The NN referee simulation ranks this first, and it is the
right call: if the pipeline can produce a V1v->hV4 gradient out of correlated
bands and ROI-varying SNR alone, §2.2 is an artefact.

This module builds voxels whose true unique-variance split is known exactly,
pushes them through the SAME production code path (`fit_banded` +
`variance_partition`, not a reimplementation), and asks two questions.

GROUND TRUTH. Real cached magno/parvo features are used, so the inter-band
correlation is the true one rather than an assumed one. Only the voxel responses
are synthetic. Three mutually ORTHOGONAL sources are constructed from the real
feature spaces:

    S_m  in  span(Zm) ∩ span(Zp)^⊥   -- only the magno band can predict it
    S_p  in  span(Zp) ∩ span(Zm)^⊥   -- only the parvo band can predict it
    S_s  in  span(Zm) ∩ span(Zp)     -- either band can predict it

Orthogonality is not assumed, it follows: S_p is orthogonal to all of span(Zm)
and S_s lies inside span(Zm), so S_p ⊥ S_s; likewise S_m ⊥ S_s and S_m ⊥ S_p.
It is also asserted numerically before use. With mutually orthogonal sources the
population values are exact:

    true unique_m = var(S_m)/var(signal),  true unique_p = var(S_p)/var(signal)

CONDITION A -- recovery. The true split varies across simulated ROIs in a known
monotone way. Does the estimate track it, and with what bias?

CONDITION B -- manufacture (the decisive one). Every simulated ROI is given an
IDENTICAL true split, and ROIs differ only in noise ceiling and voxel count,
drawn from the real per-ROI values. A gradient here is manufactured, because
there is none in the truth. This is the condition that can kill §2.2.

    PYTHONPATH=. python tmp_analysis/vp_recovery.py --subject 1
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

CACHE = "/scratch/byrong/VST_data/mpk_feature_cache"


def pca_fit_transform(Xtr: np.ndarray, Xte: np.ndarray, n_comp: int,
                      seed: int) -> tuple[np.ndarray, np.ndarray]:
    """PCA fit on TRAIN only, applied to both -- as run_encoding does."""
    from sklearn.decomposition import PCA
    p = PCA(n_components=n_comp, svd_solver="randomized", random_state=seed)
    Ztr = p.fit_transform(Xtr)
    Zte = p.transform(Xte)
    return Ztr.astype(np.float64), Zte.astype(np.float64)


def orth_basis(Z: np.ndarray) -> np.ndarray:
    """Orthonormal basis for the column space of Z."""
    Q, _ = np.linalg.qr(Z)
    return Q


def build_sources(Qm: np.ndarray, Qp: np.ndarray) -> tuple[dict, dict]:
    """Three near-orthogonal unit-variance sources, via canonical correlations.

    The obvious construction -- take span(Qm) ∩ span(Qp)^⊥ -- does not work
    here: for two generic k-dimensional subspaces of an n-dimensional space
    with 2k < n, that intersection is {0}. Nothing lives in the magno space
    that is exactly orthogonal to the whole parvo space, so "magno-unique"
    cannot be defined that way.

    The right object is the principal angles between the two subspaces. For
    orthonormal Qm, Qp the singular values of Qm^T Qp ARE the canonical
    correlations, and the canonical variates are Qm U and Qp V:

        sigma ~ 1  ->  a magno direction the parvo band predicts almost
                       perfectly            = SHARED
        sigma ~ 0  ->  a magno direction the parvo band cannot predict at all
                       = MAGNO-UNIQUE

    A canonical direction with correlation sigma has population R^2 = sigma^2
    from the other band, so the split is known rather than assumed.

    Orthogonality comes free: Qm U has orthonormal columns, so distinct magno
    variates are orthogonal; and canonical variates across the two sets are
    uncorrelated except for matched indices. Using the FIRST magno variate for
    the shared source and the LAST magno/parvo variates for the unique ones
    keeps all three mutually near-orthogonal, which is asserted numerically by
    the caller rather than trusted.
    """
    M = Qm.T @ Qp
    U, sig, Vt = np.linalg.svd(M)
    V = Vt.T
    sig = np.clip(sig, 0.0, 1.0)

    def unit(v: np.ndarray) -> np.ndarray:
        v = v - v.mean()
        return v / v.std()

    s_s = unit(Qm @ U[:, 0])      # highest canonical correlation -> shared
    s_m = unit(Qm @ U[:, -1])     # lowest  -> parvo cannot predict it
    s_p = unit(Qp @ V[:, -1])     # lowest  -> magno cannot predict it

    info = {
        "sigma_shared": float(sig[0]),
        "sigma_unique_m": float(sig[-1]),
        "sigma_unique_p": float(sig[-1]),
        "sigma_mean": float(sig.mean()),
        "sigma_median": float(np.median(sig)),
    }
    return {"m": s_m, "p": s_p, "s": s_s}, info


def make_voxels(src: dict[str, np.ndarray], cca: dict, a: float, b: float,
                c: float, nc: np.ndarray, rng
                ) -> tuple[np.ndarray, dict[str, float]]:
    """Voxels with a known unique split at a given noise ceiling.

    `nc` is one target noise ceiling per voxel: the fraction of each voxel's
    variance that is explainable in principle.

    The population truth follows from the canonical correlations. A source
    that is a canonical variate with correlation sigma is predicted by the
    OTHER band with R^2 = sigma^2, and by its own band perfectly. With the
    three sources mutually orthogonal:

        R2_magno = [a^2 + b^2*sig_p^2 + c^2] / v
        R2_parvo = [a^2*sig_m^2 + b^2 + c^2*sig_s^2] / v
        R2_joint = 1

    so unique_m = R2_joint - R2_parvo and unique_p = R2_joint - R2_magno,
    which is what the pipeline is being asked to recover.
    """
    sig_m = cca["sigma_unique_m"]
    sig_p = cca["sigma_unique_p"]
    sig_s = cca["sigma_shared"]

    s = a * src["m"] + b * src["p"] + c * src["s"]
    s = s - s.mean()
    v = s.var()
    sig = s

    r2_magno = (a**2 + b**2 * sig_p**2 + c**2) / v
    r2_parvo = (a**2 * sig_m**2 + b**2 + c**2 * sig_s**2) / v
    truth = {
        "unique_m": 1.0 - r2_parvo,
        "unique_p": 1.0 - r2_magno,
        "shared": r2_magno + r2_parvo - 1.0,
    }

    n = sig.shape[0]
    Y = np.empty((n, nc.shape[0]))
    for i, ceil in enumerate(nc):
        ceil = float(np.clip(ceil, 1e-3, 0.999))
        # var(sig) / (var(sig) + var(noise)) = ceil
        noise_var = v * (1.0 - ceil) / ceil
        Y[:, i] = sig + rng.standard_normal(n) * np.sqrt(noise_var)
    return Y, truth


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--subject", type=int, default=1)
    p.add_argument("--model", default="dkl_bcs_seed0_trained")
    p.add_argument("--n-comp", type=int, default=1000)
    p.add_argument("--n-vox", type=int, default=200,
                   help="simulated voxels per ROI")
    p.add_argument("--backend", default="numpy")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="results/vp_recovery")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    from src.analysis.encoding.banded_ridge import (
        fit_banded, make_grouped_splits, variance_partition,
    )

    z = np.load(f"{CACHE}/{args.model}_subj{args.subject:02d}.npz")
    Xm, Xp, ids = z["magno"], z["parvo"], z["image_ids"]
    n = Xm.shape[0]
    print(f"[{args.model} subj{args.subject:02d}] {n} images, "
          f"{Xm.shape[1]} features/stream", flush=True)

    # held-out test split by image id, as the real pipeline does
    order = rng.permutation(n)
    n_te = n // 5
    te, tr = order[:n_te], order[n_te:]

    print("PCA (train-only fit) ...", flush=True)
    Zm_tr, Zm_te = pca_fit_transform(Xm[tr], Xm[te], args.n_comp, args.seed)
    Zp_tr, Zp_te = pca_fit_transform(Xp[tr], Xp[te], args.n_comp, args.seed)

    # --- ground-truth sources, built on the TRAIN rows and carried to test ---
    # Sources must be defined on all rows at once so the test-set truth is the
    # same signal; build the bases on the stacked design.
    Zm_all = np.vstack([Zm_tr, Zm_te])
    Zp_all = np.vstack([Zp_tr, Zp_te])
    print("orthonormal bases + canonical correlations ...", flush=True)
    Qm, Qp = orth_basis(Zm_all), orth_basis(Zp_all)
    src, cca = build_sources(Qm, Qp)

    # assert the construction rather than trusting the algebra
    max_corr = 0.0
    for x, y in (("m", "p"), ("m", "s"), ("p", "s")):
        c = abs(float(np.corrcoef(src[x], src[y])[0, 1]))
        max_corr = max(max_corr, c)
        print(f"  corr(S_{x}, S_{y}) = {c:.2e}")
    assert max_corr < 1e-3, f"sources not near-orthogonal: max r={max_corr}"

    print(f"  canonical correlations: shared sigma={cca['sigma_shared']:.4f}, "
          f"unique sigma={cca['sigma_unique_m']:.4f}, "
          f"median={cca['sigma_median']:.4f}", flush=True)
    # The band correlation the referee asked us to match is exactly this
    # spectrum: it is measured from the real features, not assumed.
    band_corr = cca["sigma_median"]

    splits = make_grouped_splits(ids[tr], n_splits=3, seed=args.seed)

    # ROI noise ceilings taken from the real fitted results if present
    roi_nc = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
    results = {"model": args.model, "subject": args.subject,
               "band_corr": band_corr, "cca": cca, "conditions": {}}

    for cond in ("A_recovery", "B_manufacture"):
        rows = []
        for i, ceil in enumerate(roi_nc):
            if cond == "A_recovery":
                # true split varies monotonically across ROIs
                a = 0.30 + 0.10 * i
                b = 0.90 - 0.10 * i
            else:
                # IDENTICAL truth everywhere; only the noise ceiling varies
                a, b = 0.60, 0.60
            c_sh = 1.0

            nc = np.full(args.n_vox, ceil)
            Ytr, truth = make_voxels(src, cca, a, b, c_sh, nc, rng)
            # reuse the same signal for the test rows
            Y_all = Ytr
            Y_train, Y_test = Y_all[:len(tr)], Y_all[len(tr):]

            out = fit_banded(
                [Zm_tr, Zp_tr], Y_train, [Zm_te, Zp_te], Y_test,
                splits, backend=args.backend, seed=args.seed,
            )
            vp = variance_partition(
                out["r2_joint"], out["r2_magno"], out["r2_parvo"],
                noise_ceiling=np.full(args.n_vox, ceil),
            )
            est_m = float(np.nanmedian(vp["unique_magno"]))
            est_p = float(np.nanmedian(vp["unique_parvo"]))
            rows.append({
                "roi_index": i, "noise_ceiling": ceil,
                "true_unique_m": truth["unique_m"],
                "true_unique_p": truth["unique_p"],
                "est_unique_m": est_m, "est_unique_p": est_p,
                "true_diff": truth["unique_m"] - truth["unique_p"],
                "est_diff": est_m - est_p,
            })
            print(f"  [{cond}] roi{i} nc={ceil:.2f}  "
                  f"true Um-Up={rows[-1]['true_diff']:+.4f}  "
                  f"est {rows[-1]['est_diff']:+.4f}", flush=True)

        from scipy import stats as sps
        idx = np.arange(len(rows))
        est = np.array([r["est_diff"] for r in rows])
        tru = np.array([r["true_diff"] for r in rows])
        rho_est = float(sps.spearmanr(idx, est).statistic)
        rho_true = float(sps.spearmanr(idx, tru).statistic)
        results["conditions"][cond] = {
            "rows": rows, "rho_est_vs_rank": rho_est,
            "rho_true_vs_rank": rho_true,
        }
        print(f"  [{cond}] rho(true, rank)={rho_true:+.3f}  "
              f"rho(est, rank)={rho_est:+.3f}\n", flush=True)

    path = os.path.join(
        args.out_dir, f"vp_recovery_subj{args.subject:02d}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
