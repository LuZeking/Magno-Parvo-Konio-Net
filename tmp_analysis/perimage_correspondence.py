"""Does any readout of these two streams track WHICH images humans find hard?

2.6.2 bounds trial-level agreement with individual observers (Cohen's kappa,
human-human ceiling 0.234) and finds no readout above 20.7% of it. That is a
null measured with a weak instrument: one observer's one binary response per
image. This script replaces the instrument.

The target is the per-image human profile averaged over all 10 observers, whose
reliability is far higher (tmp_analysis/human_perimage.py):

    h_i  = P(shape | shape or texture)      ceiling  r = .658 (.624 residual)
    v_i  = P(shape or texture at all)       ceiling  r = .794 (.728 residual)

and the model side is continuous rather than binary, which is where the power
comes from. Two matched model measures, both read off the 16 restricted columns
of LG(w) = w*L1 + (1-w)*L2:

    shape_margin_i(w) = LG[i, shape_i] - LG[i, texture_i]        pairs with h
    cue_margin_i(w)   = max(LG[i, shape_i], LG[i, texture_i])
                        - max over the other 14 categories       pairs with v

Everything is reported raw and after removing shape-category and
texture-category main effects. Only the residual version is evidence about
IMAGES: the raw version can be carried entirely by the 16 categories differing
in difficulty, which is not a claim about image content. Residualisation is a
linear projection applied identically to both sides.

Two w's are then compared, which is the point of the analysis:
    w_rate    the w whose aggregate shape bias is closest to the human 0.958
              -- the operating point the field's standard practice tunes to
    w_choice  the w that best predicts the human profile, chosen out of sample

Usage: PYTHONPATH=.:tmp_analysis python tmp_analysis/perimage_correspondence.py \
           --run-dir <dir>
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

HUMAN_RATE = 0.958   # Geirhos et al. 2018; reproduced at 0.9587 in this loading


# --------------------------------------------------------------------------
def category_projector(shapes, textures):
    """Residual-maker for an additive shape-category + texture-category design.

    Returns R with R @ y = y - X beta_hat. Applying one projection matrix is
    both faster than re-solving per readout weight and exactly equivalent, and
    -- because it is linear -- it commutes with the convex combination over w.
    """
    sh, tx = sorted(set(shapes)), sorted(set(textures))
    X = np.zeros((len(shapes), 1 + len(sh) + len(tx)))
    X[:, 0] = 1.0
    for j, c in enumerate(sh):
        X[np.asarray(shapes) == c, 1 + j] = 1.0
    for j, c in enumerate(tx):
        X[np.asarray(textures) == c, 1 + len(sh) + j] = 1.0
    P = X @ np.linalg.pinv(X)
    return np.eye(len(shapes)) - P


def pearson_rows(M, y):
    """Pearson r between every row of M and vector y (finite entries of y)."""
    m = np.isfinite(y)
    A = M[:, m].astype(float)
    b = y[m].astype(float)
    A = A - A.mean(1, keepdims=True)
    b = b - b.mean()
    num = A @ b
    den = np.sqrt((A * A).sum(1) * (b * b).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def rank_rows(M):
    from scipy.stats import rankdata
    return rankdata(M, axis=1).astype(float)


def build_measures(L1, L2, cols, classes, shapes, textures, grid):
    """(n_grid, n_images) shape margin, cue margin, and shape-bias rate."""
    classes = np.asarray(classes, dtype=object)
    n = L1.shape[0]
    idx = np.arange(n)
    pos = {c: j for j, c in enumerate(classes)}
    si = np.array([pos[c] for c in shapes])
    ti = np.array([pos[c] for c in textures])

    A1, A2 = L1[:, cols], L2[:, cols]          # (n, 16)
    sm = np.empty((len(grid), n))
    cm = np.empty((len(grid), n))
    rates = np.empty(len(grid))
    for j, w in enumerate(grid):
        LG = w * A1 + (1.0 - w) * A2
        ls, lt = LG[idx, si], LG[idx, ti]
        sm[j] = ls - lt
        other = LG.copy()
        other[idx, si] = -np.inf
        other[idx, ti] = -np.inf
        cm[j] = np.maximum(ls, lt) - other.max(1)
        dec = classes[LG.argmax(1)]
        ns = int((dec == shapes).sum()); nt = int((dec == textures).sum())
        rates[j] = ns / (ns + nt) if (ns + nt) else np.nan
    return sm, cm, rates


def analyse(M, target, grid, w_rate, ceiling, rng, n_splits, n_boot, kind):
    """r(w) for one model measure against one human profile, plus OOS and CIs.

    Images the humans never resolved to either cue are dropped BEFORE ranking,
    so the Spearman variant is a rank correlation over exactly the images it
    reports on. Ranking first and masking after would leave non-consecutive
    global ranks in the subset, which is not Spearman's statistic. The split
    halves then inherit those ranks rather than re-ranking, so that every
    split scores the same target.
    """
    keep = np.flatnonzero(np.isfinite(target))
    M = M[:, keep].astype(float)
    target = target[keep].astype(float)
    if kind == "spearman":
        from scipy.stats import rankdata
        M = rank_rows(M)
        target = rankdata(target).astype(float)

    r = pearson_rows(M, target)
    j = int(np.nanargmax(np.abs(r)))
    jr = int(np.argmin(np.abs(grid - w_rate)))

    # out of sample: pick w on half the images, score it on the other half
    fin = np.arange(M.shape[1])
    oos_w, oos_r = [], []
    for _ in range(n_splits):
        perm = rng.permutation(fin)
        tr_i, te_i = perm[:len(perm) // 2], perm[len(perm) // 2:]
        rt = pearson_rows(M[:, tr_i], target[tr_i])
        wj = int(np.nanargmax(np.abs(rt)))
        oos_r.append(float(pearson_rows(M[[wj]][:, te_i], target[te_i])[0]))
        oos_w.append(float(grid[wj]))
    oos_w, oos_r = np.array(oos_w), np.array(oos_r)

    def boot(jj):
        x, y = M[jj], target
        out = np.empty(n_boot)
        for b in range(n_boot):
            k = rng.integers(0, len(x), len(x))
            u, v = x[k] - x[k].mean(), y[k] - y[k].mean()
            d = np.sqrt((u * u).sum() * (v * v).sum())
            out[b] = (u * v).sum() / d if d > 0 else np.nan
        return out
    bc, br = boot(j), boot(jr)

    pc = (lambda z: float(100 * z / ceiling)) if ceiling else (lambda z: None)
    return {
        "w_choice": float(grid[j]), "r_at_w_choice": float(r[j]),
        "r_at_w_choice_ci": [float(np.nanpercentile(bc, 2.5)),
                             float(np.nanpercentile(bc, 97.5))],
        "pct_ceiling_at_w_choice": pc(r[j]),
        "r_at_w_rate": float(r[jr]),
        "r_at_w_rate_ci": [float(np.nanpercentile(br, 2.5)),
                           float(np.nanpercentile(br, 97.5))],
        "pct_ceiling_at_w_rate": pc(r[jr]),
        "r_min": float(np.nanmin(r)), "r_max": float(np.nanmax(r)),
        "r_span": float(np.nanmax(r) - np.nanmin(r)),
        "oos_r_mean": float(oos_r.mean()),
        "oos_r_ci": [float(np.percentile(oos_r, 2.5)),
                     float(np.percentile(oos_r, 97.5))],
        "oos_pct_ceiling": pc(oos_r.mean()),
        "oos_w_mean": float(oos_w.mean()), "oos_w_sd": float(oos_w.std()),
        "ceiling": ceiling,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--n-splits", type=int, default=200)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--w-step", type=float, default=0.005)
    p.add_argument("--expect-n", type=int, default=None,
                   help="fail if this many logit files are not present; a "
                        "partially-finished extraction array otherwise yields "
                        "a table that silently omits configurations")
    args = p.parse_args()

    H = np.load(os.path.join(args.run_dir, "human_perimage.npz"),
                allow_pickle=True)
    ceil = json.load(open(os.path.join(args.run_dir, "human_ceiling.json")))
    shapes = [str(x) for x in H["shapes"]]
    textures = [str(x) for x in H["textures"]]
    R = category_projector(shapes, textures)

    targets = {
        "shape_rate":       (np.asarray(H["h"], float), "shape_rate"),
        "shape_rate_resid": (np.asarray(H["h_resid"], float), "shape_rate_resid"),
        "valid_rate":       (np.asarray(H["v"], float), "valid_rate"),
        "valid_rate_resid": (np.asarray(H["v_resid"], float), "valid_rate_resid"),
    }
    print(f"humans: {H['C'].shape[0]} observers x {len(shapes)} images; "
          f"aggregate shape bias {ceil['aggregate_shape_bias']:.4f}")
    for k, (_, cn) in targets.items():
        c = ceil["ceilings"][cn]
        print(f"  ceiling[{k:17s}] Pearson {c['pearson_mean']:+.4f}  "
              f"Spearman {c['spearman_mean']:+.4f}")

    grid = np.round(np.arange(0.0, 1.0 + 1e-9, args.w_step), 6)
    rng = np.random.default_rng(0)
    report = {"human_ceiling": ceil["ceilings"], "grid_step": args.w_step,
              "models": {}}

    files = sorted(glob.glob(os.path.join(args.run_dir, "logits",
                                          "*_perimage_logits.npz")))
    if args.expect_n is not None and len(files) != args.expect_n:
        raise SystemExit(
            f"expected {args.expect_n} logit files, found {len(files)}:\n  "
            + "\n  ".join(os.path.basename(f) for f in files))
    print(f"\n{len(files)} model configurations\n")
    for f in files:
        Z = np.load(f, allow_pickle=True)
        name = str(Z["model_name"])
        assert [str(x) for x in Z["shapes"]] == shapes, f"{name}: image order"
        sm, cm, rates = build_measures(
            Z["L1"], Z["L2"], Z["geirhos_cols"],
            [str(c) for c in Z["geirhos_classes"]], np.asarray(shapes),
            np.asarray(textures), grid)
        jr = int(np.nanargmin(np.abs(rates - HUMAN_RATE)))
        w_rate = float(grid[jr])

        # Can the readout change WHICH images are hard, or only how many?
        # sm is a convex combination of two fixed vectors; if the streams rank
        # images alike, every readout inherits one ordering and w moves the
        # rate without moving the ordering. That would dissociate the aggregate
        # rate from item-level difficulty inside a single frozen network.
        s0, s1 = sm[0], sm[-1]
        stream_r = float(np.corrcoef(s0, s1)[0, 1])
        stream_rho = float(np.corrcoef(rank_rows(np.vstack([s0, s1])))[0, 1])
        smr = sm @ R.T
        stream_r_res = float(np.corrcoef(smr[0], smr[-1])[0, 1])

        entry = {
            "rate_range": [float(np.nanmin(rates)), float(np.nanmax(rates))],
            "w_rate": w_rate, "rate_at_w_rate": float(rates[jr]),
            "stream_margin_pearson": stream_r,
            "stream_margin_spearman": stream_rho,
            "stream_margin_pearson_resid": stream_r_res,
        }
        print(f"=== {name} ===")
        print(f"  rate spans {entry['rate_range'][0]:.3f}-"
              f"{entry['rate_range'][1]:.3f};  w_rate={w_rate:.3f} "
              f"(rate {entry['rate_at_w_rate']:.3f})")
        print(f"  the two streams rank images alike: r={stream_r:+.4f}  "
              f"rho={stream_rho:+.4f}  r_resid={stream_r_res:+.4f}")

        for tname, (tgt, cname) in targets.items():
            M = cm if tname.startswith("valid") else sm
            if tname.endswith("resid"):
                M = M @ R.T
            for kind in ("pearson", "spearman"):
                C = ceil["ceilings"][cname][f"{kind}_mean"]
                d = analyse(M, tgt, grid, w_rate, C, rng,
                            args.n_splits, args.n_boot, kind)
                entry[f"{tname}__{kind}"] = d
                print(f"  [{tname:17s}/{kind:8s}] "
                      f"w*={d['w_choice']:.3f} r={d['r_at_w_choice']:+.4f} "
                      f"({d['pct_ceiling_at_w_choice']:5.1f}% ceil) | "
                      f"@w_rate {d['r_at_w_rate']:+.4f} "
                      f"({d['pct_ceiling_at_w_rate']:5.1f}%) | "
                      f"oos {d['oos_r_mean']:+.4f} "
                      f"({d['oos_pct_ceiling']:5.1f}%) "
                      f"w={d['oos_w_mean']:.2f}+-{d['oos_w_sd']:.2f} | "
                      f"r(w) span {d['r_span']:.4f}")
        report["models"][name] = entry
        print()

    out = os.path.join(args.run_dir, "perimage_correspondence.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
