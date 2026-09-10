"""Fit the readout weight `w` to each of the ten Geirhos observers (PLAN §106).

Limitation 14 states the gap this closes in its own words: "No w was fitted to
human data and nothing here estimates the weighting a human readout would use."
§2.3 sweeps w and §2.6.2 recomputes kappa at 21 weights, but a sweep reports a
RANGE and a fit reports an ESTIMATE -- and an estimate can come back
unattainable, which says something a range cannot: that the observer lies
outside this family entirely.

TWO ESTIMATORS, both reported, per §8's two-statistics rule (§106.2):

    E1  w_i^rate  = argmin_w | sb(w) - rate_i |     fit to the observer's RATE
    E2  w_i^trial = argmax_w  kappa(model_w, obs_i) fit to the observer's TRIALS

They answer different questions on purpose. E1 asks what weighting reproduces a
person's marginal shape-bias rate; E2 asks what weighting best predicts their
individual choices. Whether they agree IS the result, not a nuisance.

TWO CONTROLS ON E2, added POST-HOC after a smoke run falsified §106.7's third
prediction (see §106.10 -- the sequence is on the record in the order it
happened). E2 is an argmax over 1001 weights, then reported as the best of ten
such maxima, so it is selected twice and some of it is free:

    Control A  split-half cross-validation -- fit w on half the items, score
               kappa on the held-out half. R random splits. The direct test of
               whether a fitted w is real or is fitting 1,200 trials of noise.
    Control B  permutation null -- shuffle the observer's trials across items P
               times and recompute the whole best-over-w statistic, giving the
               distribution of the selected maximum under no correspondence.

WHY A 0.001 GRID IS FREE. The classifier is affine and the fusion weights sum
to one, so

    clf(w*F1 + (1-w)*F2) = w*clf(F1) + (1-w)*clf(F2)

holds EXACTLY -- the two endpoint logit matrices determine every w. The forward
pass therefore runs once and every one of the 1001 grid points is an
interpolation. §2.3 measured the deviation of this identity at 1.1e-6; this
script re-measures it rather than trusting it, because the entire cost argument
rests on it.

WHY THE CONTROLS ARE ALSO FREE. Cohen's kappa needs only four counts, and each
is a dot product between a fixed model indicator and an observer indicator:

    n    = 1[model decided]  .  1[human decided]
    n_ag = 1[model shape]    .  1[human shape]  +  1[model texture] . 1[human texture]
    n_x  = 1[model shape]    .  1[human decided]
    n_y  = 1[model decided]  .  1[human shape]

So P permutations, R splits or B bootstrap replicates are ONE matmul each
against a (n_items x P) matrix of observer vectors, not a loop.

NOTHING VALIDATED IS REIMPLEMENTED. The human loader, the scalar kappa and the
shape-bias scorer are imported from the modules that already produced the
published numbers, and the matmul forms below are asserted against them at
every point of the coarse 0.05 grid before any fit is read. The published
anchors (human shape bias 0.958, human-human ceiling 0.234) are asserted too.

Usage: PYTHONPATH=.:tmp_analysis python tmp_analysis/fit_w_observers.py \
           --checkpoint ... --model-name ...
"""
from __future__ import annotations

import argparse
import itertools
import json
import os

import numpy as np
import torch

from src.analysis.encoding import dkl_frontend, features
from src.analysis.behavior.cue_conflict import (
    GEIRHOS_CLASSES,
    list_cue_conflict_images,
    load_class_index,
    load_images,
    shape_bias,
    stream_in_channels,
)
from fusion_sweep import stream_features          # identical feature path to §2.3
from sweep_error_consistency import kappa, load_humans   # validated, not rewritten

# Decision coding used throughout: 1 = chose shape, 0 = chose texture,
# -1 = decided neither (excluded from every denominator, as §2.6.2 does).
SHAPE, TEXTURE, NEITHER = 1, 0, -1


def code_choices(choice_list) -> np.ndarray:
    """'S'/'T'/None  ->  the integer coding above."""
    out = np.full(len(choice_list), NEITHER, dtype=np.int8)
    for i, c in enumerate(choice_list):
        if c == "S":
            out[i] = SHAPE
        elif c == "T":
            out[i] = TEXTURE
    return out


class ModelDecisions:
    """The model's decision at every w, and kappa against arbitrary observers.

    Holds the three (n_w x n_items) indicator matrices that kappa's sufficient
    statistics dot against. Everything downstream -- point estimates, the
    bootstrap, both controls -- goes through `kappa_against`, so there is one
    kappa in this file and it is the one asserted against the validated scalar.
    """

    def __init__(self, M: np.ndarray):
        self.M = M
        self.dec = (M != NEITHER).astype(np.float32)
        self.s = (M == SHAPE).astype(np.float32)
        self.t = (M == TEXTURE).astype(np.float32)

    def kappa_against(self, h_s, h_t, h_d):
        """kappa for every w against every column of the observer matrices.

        `h_s`, `h_t`, `h_d` are (n_items,) or (n_items, K) float arrays: the
        observer's shape / texture / decided indicators, optionally weighted
        (bootstrap multiplicities) or masked (a cross-validation half).
        Returns (n_w,) or (n_w, K).
        """
        # The matmuls are exact: every operand is a 0/1 indicator or a small
        # integer multiplicity, and the sums cannot exceed n_items, so they sit
        # well inside float32's exactly-representable integers. The RATIOS are
        # not, so the arithmetic below is done in float64 -- without this the
        # matmul form disagrees with the validated scalar kappa at ~1e-7, which
        # is harmless for the fit but would defeat the equality assertion that
        # is the whole reason to trust this code.
        n = np.float64(self.dec @ h_d)
        n_ag = np.float64(self.s @ h_s) + np.float64(self.t @ h_t)
        n_x = np.float64(self.s @ h_d)
        n_y = np.float64(self.dec @ h_s)
        with np.errstate(divide="ignore", invalid="ignore"):
            c_obs = n_ag / n
            px, py = n_x / n, n_y / n
            c_exp = px * py + (1 - px) * (1 - py)
            k = (c_obs - c_exp) / (1 - c_exp)
        # Same degeneracy guards as the scalar form: too few joint decisions,
        # or a chance agreement of exactly 1 (one side constant), give no kappa.
        return np.where((n < 30) | (c_exp >= 1.0), np.nan, k)

    def rate(self, weights=None):
        """Shape-bias rate and decided count for every w."""
        if weights is None:
            n_s, n_t = self.s.sum(1), self.t.sum(1)
        else:
            n_s, n_t = self.s @ weights, self.t @ weights
        with np.errstate(divide="ignore", invalid="ignore"):
            return n_s / (n_s + n_t), n_s + n_t


def obs_indicators(h: np.ndarray, weights=None):
    """Observer code vector -> (shape, texture, decided) float indicators."""
    h_s = (h == SHAPE).astype(np.float32)
    h_t = (h == TEXTURE).astype(np.float32)
    h_d = (h != NEITHER).astype(np.float32)
    if weights is not None:
        w = np.asarray(weights, dtype=np.float32)
        if w.ndim == 2 and h_s.ndim == 1:
            h_s, h_t, h_d = h_s[:, None] * w, h_t[:, None] * w, h_d[:, None] * w
        else:
            h_s, h_t, h_d = h_s * w, h_t * w, h_d * w
    return h_s, h_t, h_d


def nanargmax_cols(a):
    """Per-column argmax that tolerates all-NaN columns (returns 0 there)."""
    return np.argmax(np.where(np.isfinite(a), a, -np.inf), axis=0)


def summarise_fit(grid, curve, target, mode):
    """One observer's fit, with attainability decided BEFORE the argmax is read.

    §106.3 declared that a fit landing at w = 0 or w = 1 is reported as "no
    interior estimate exists" rather than as an estimate of 1.0. The honest
    generalisation, applied here and disclosed: for the RATE fit what matters is
    not where the argmin sits but whether the target is inside the family's
    range at all. `sb(w)` need not be monotonic, so its maximum can be interior
    -- and an argmin sitting at an interior maximum is still not a fit, it is
    the closest miss. Attainability is therefore range membership, which is the
    same test §106.3 intended and a stricter one than a boundary check.
    """
    if not np.isfinite(curve).any():
        return {"attainable": False, "w_hat": None, "reason": "no finite value"}

    lo, hi = float(np.nanmin(curve)), float(np.nanmax(curve))
    if mode == "rate":
        idx = int(np.nanargmin(np.abs(curve - target)))
        attainable = bool(lo <= target <= hi)
    else:
        idx = int(nanargmax_cols(curve[:, None])[0])
        attainable = True
    at_boundary = idx in (0, len(grid) - 1)

    return {
        "attainable": attainable,
        "w_hat": float(grid[idx]),
        "w_index": idx,
        "value_at_w_hat": float(curve[idx]),
        "target": None if target is None else float(target),
        "residual": None if target is None else float(curve[idx] - target),
        "at_w_boundary": at_boundary,
        "curve_min": lo,
        "curve_max": hi,
        "interior_estimate_exists": bool(attainable and not at_boundary),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--out-dir", default="results/fit_w_observers")
    p.add_argument("--no-lum-blur", action="store_true")
    p.add_argument("--no-lum-cs", action="store_true")
    p.add_argument("--transform-target", default="magno",
                   choices=["magno", "parvo"])
    p.add_argument("--parvo-highsf", action="store_true")
    p.add_argument("--age-months", type=int, default=48)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--w-step", type=float, default=0.001)   # §106.4
    p.add_argument("--n-boot", type=int, default=1000)      # §106.4
    p.add_argument("--n-splits", type=int, default=200)     # §106.10 Control A
    p.add_argument("--n-perm", type=int, default=1000)      # §106.10 Control B
    p.add_argument("--boot-seed", type=int, default=20260910)
    p.add_argument("--allow-partial", action="store_true")
    args = p.parse_args()

    features.assert_converged(args.checkpoint, args.allow_partial)
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    items = list_cue_conflict_images()
    paths = [i[0] for i in items]
    shapes = [i[1] for i in items]
    textures = [i[2] for i in items]
    stems = [os.path.splitext(os.path.basename(q))[0] for q in paths]
    sh_a = np.asarray(shapes, dtype=object)
    tx_a = np.asarray(textures, dtype=object)
    n_items = len(stems)

    # ---- humans, and the two anchors that say the loading is right ---------
    humans = load_humans(stems, shapes, textures)
    names = sorted(humans)
    H = np.stack([code_choices(humans[h]) for h in names])          # (10, n)
    h_rate = np.array([(H[i] == SHAPE).sum() / ((H[i] == SHAPE).sum()
                                                + (H[i] == TEXTURE).sum())
                       for i in range(len(names))])
    h_decided = np.array([int((H[i] != NEITHER).sum()) for i in range(len(names))])
    hh = np.array([kappa(humans[a], humans[b])
                   for a, b in itertools.combinations(names, 2)], dtype=float)
    CEIL = float(np.nanmean(hh))
    print(f"humans: n={len(names)}  shape bias {h_rate.mean():.4f} "
          f"(range {h_rate.min():.3f}-{h_rate.max():.3f})", flush=True)
    print(f"human-human ceiling: kappa = {CEIL:.4f} "
          f"({np.isfinite(hh).sum()} pairs, sd {np.nanstd(hh, ddof=1):.4f})",
          flush=True)
    assert abs(h_rate.mean() - 0.958) < 0.002, f"human shape bias {h_rate.mean()}"
    assert abs(CEIL - 0.234) < 0.005, f"human-human ceiling {CEIL}"

    # ---- model: ONE forward pass, then pure interpolation ------------------
    ch_min, ch_max = dkl_frontend.compute_dkl_channel_minmax()
    model = features.build_model(
        checkpoint=args.checkpoint, stream_concat=False, stream1_weight=0.5,
        seed=args.seed, device=device, in_channels=stream_in_channels(args))
    F1, F2 = [], []
    for start in range(0, n_items, 128):
        rgb = load_images(paths[start:start + 128])
        x_m, x_p = dkl_frontend.magno_parvo_inputs(
            rgb, ch_min, ch_max, lum_blur=not args.no_lum_blur,
            lum_cs=not args.no_lum_cs, age_months=args.age_months,
            device=device, transform_target=args.transform_target,
            parvo_highsf=args.parvo_highsf, rgb_mode=False)
        a, b = stream_features(model, x_m, x_p, device=device)
        F1.append(a); F2.append(b)
    F1, F2 = torch.cat(F1), torch.cat(F2)

    cls = load_class_index()
    cols = [cls[c] for c in GEIRHOS_CLASSES]
    clf = model.classifier.float().cpu()
    with torch.no_grad():
        L1 = clf(F1).numpy()
        L2 = clf(F2).numpy()

    # THE identity the cost argument rests on, re-measured rather than trusted.
    dev = 0.0
    with torch.no_grad():
        for w in (0.13, 0.5, 0.87):
            direct = clf(w * F1 + (1.0 - w) * F2).numpy()
            dev = max(dev, float(np.abs(direct - (w * L1 + (1 - w) * L2)).max()))
    print(f"affine identity: max |clf(mix) - mix(clf)| = {dev:.3e}", flush=True)
    assert dev < 1e-4, f"classifier is not affine in w to tolerance: {dev}"

    # ---- decisions on the fine grid, all at once ---------------------------
    grid = np.round(np.arange(0.0, 1.0 + 1e-9, args.w_step), 6)
    A_, B_ = L1[:, cols], L2[:, cols]                    # (n, 16)
    lg = grid[:, None, None] * A_[None] + (1 - grid)[:, None, None] * B_[None]
    dec = np.asarray(GEIRHOS_CLASSES, dtype=object)[lg.argmax(2)]   # (n_w, n)
    del lg
    M = np.full(dec.shape, NEITHER, dtype=np.int8)
    M[dec == sh_a[None, :]] = SHAPE
    M[dec == tx_a[None, :]] = TEXTURE
    md = ModelDecisions(M)
    print(f"grid: {len(grid)} weights x {n_items} items", flush=True)

    sb_curve, dec_count = md.rate()

    # ---- assert the matmul forms against the validated scalar ones ---------
    # Every point of the published 0.05 grid, both statistics, before any fit
    # is read. A disagreement here means the fine grid is measuring something
    # other than what §2.3 and §2.6.2 measured.
    coarse = [i for i, w in enumerate(grid)
              if abs(w * 20 - round(w * 20)) < 1e-9]
    max_sb_err = max_k_err = 0.0
    for i in coarse:
        w = float(grid[i])
        ref = shape_bias(w * L1 + (1 - w) * L2, shapes, textures, cls)
        max_sb_err = max(max_sb_err, abs(ref["shape_bias"] - sb_curve[i]))
        assert ref["n_shape"] + ref["n_texture"] == int(round(dec_count[i])), (
            f"decided count differs at w={w}")
        ch = [None if v == NEITHER else ("S" if v == SHAPE else "T")
              for v in M[i]]
        for j, h in enumerate(names):
            a_ = kappa(ch, humans[h])
            b_ = md.kappa_against(*obs_indicators(H[j]))[i]
            if np.isfinite(a_) or np.isfinite(b_):
                max_k_err = max(max_k_err, abs(float(a_) - float(b_)))
    print(f"matmul vs validated scalar: max |d shape_bias| = {max_sb_err:.2e}, "
          f"max |d kappa| = {max_k_err:.2e}  over {len(coarse)} grid points",
          flush=True)
    assert max_sb_err < 1e-9 and max_k_err < 1e-9, "matmul form disagrees"

    # ---- resampling designs, all drawn from one seeded generator -----------
    rng = np.random.default_rng(args.boot_seed)
    boot_w = rng.multinomial(n_items, np.full(n_items, 1.0 / n_items),
                             size=args.n_boot).T.astype(np.float32)   # (n, B)
    sb_boot, _ = md.rate(boot_w)

    # Control A: R random half-splits, as 0/1 masks over items.
    half = n_items // 2
    SA = np.zeros((n_items, args.n_splits), dtype=np.float32)
    for r in range(args.n_splits):
        SA[rng.choice(n_items, half, replace=False), r] = 1.0
    SB = 1.0 - SA

    print(f"\nfamily rate range over the {len(grid)}-point grid: "
          f"{np.nanmin(sb_curve):.4f} - {np.nanmax(sb_curve):.4f}", flush=True)

    out_obs = {}
    for j, h in enumerate(names):
        hs, ht, hd = obs_indicators(H[j])
        k_curve = md.kappa_against(hs, ht, hd)

        e1 = summarise_fit(grid, sb_curve, h_rate[j], mode="rate")
        e2 = summarise_fit(grid, k_curve, None, mode="trial")
        e2["pct_of_ceiling"] = 100.0 * e2["value_at_w_hat"] / CEIL

        # --- item bootstrap on both fitted weights (§106.4) -----------------
        k_boot = md.kappa_against(*obs_indicators(H[j], boot_w))
        with np.errstate(invalid="ignore"):
            w_e1_b = grid[np.nanargmin(np.abs(sb_boot - h_rate[j]), axis=0)]
            w_e2_b = grid[nanargmax_cols(k_boot)]
            att_b = ((np.nanmin(sb_boot, axis=0) <= h_rate[j])
                     & (h_rate[j] <= np.nanmax(sb_boot, axis=0)))
        e1["boot_ci"] = [float(np.percentile(w_e1_b, 2.5)),
                         float(np.percentile(w_e1_b, 97.5))]
        e1["boot_frac_attainable"] = float(att_b.mean())
        e2["boot_ci"] = [float(np.percentile(w_e2_b, 2.5)),
                         float(np.percentile(w_e2_b, 97.5))]
        e2["boot_frac_at_boundary"] = float(
            ((w_e2_b <= grid[0] + 1e-9) | (w_e2_b >= grid[-1] - 1e-9)).mean())

        # --- Control A: split-half cross-validation (§106.10) ---------------
        kA = md.kappa_against(*obs_indicators(H[j], SA))     # (n_w, R)
        kB = md.kappa_against(*obs_indicators(H[j], SB))
        idx = nanargmax_cols(kA)
        r_ix = np.arange(args.n_splits)
        in_sample = kA[idx, r_ix]                 # fitted and scored on A
        held_out = kB[idx, r_ix]                  # fitted on A, scored on B
        # The no-selection floor: kappa on the held-out half at a w chosen
        # without reference to the data at all, averaged over the grid.
        floor = np.nanmean(kB, axis=0)
        cv = {
            "n_splits": args.n_splits,
            "in_sample_mean": float(np.nanmean(in_sample)),
            "held_out_mean": float(np.nanmean(held_out)),
            "held_out_sd": float(np.nanstd(held_out, ddof=1)),
            "no_fitting_floor_mean": float(np.nanmean(floor)),
            "held_out_pct_of_ceiling": float(100 * np.nanmean(held_out) / CEIL),
            "floor_pct_of_ceiling": float(100 * np.nanmean(floor) / CEIL),
            "generalisation_gap": float(np.nanmean(in_sample)
                                        - np.nanmean(held_out)),
            "held_out_above_floor": float(np.nanmean(held_out)
                                          - np.nanmean(floor)),
        }

        # --- Control B: permutation null for the selected maximum -----------
        perm = np.empty((n_items, args.n_perm), dtype=np.int8)
        for q in range(args.n_perm):
            perm[:, q] = H[j][rng.permutation(n_items)]
        k_perm = md.kappa_against((perm == SHAPE).astype(np.float32),
                                  (perm == TEXTURE).astype(np.float32),
                                  (perm != NEITHER).astype(np.float32))
        null_max = np.nanmax(np.where(np.isfinite(k_perm), k_perm, -np.inf),
                             axis=0)
        obs_max = e2["value_at_w_hat"]
        nullctl = {
            "n_perm": args.n_perm,
            "null_max_mean": float(np.mean(null_max)),
            "null_max_p95": float(np.percentile(null_max, 95)),
            "null_max_max": float(np.max(null_max)),
            "observed_max": float(obs_max),
            # (#null >= observed + 1) / (P + 1): the exact one-sided p, never 0.
            "p_perm": float((np.sum(null_max >= obs_max) + 1) / (args.n_perm + 1)),
            "excess_over_null_mean": float(obs_max - np.mean(null_max)),
        }

        out_obs[h] = {
            "observer_rate": float(h_rate[j]),
            "observer_decided": int(h_decided[j]),
            "rate_fit": e1,
            "trial_fit": e2,
            "cross_validation": cv,
            "permutation_null": nullctl,
            "model_decided_at_w_trial": int(round(dec_count[e2["w_index"]])),
        }
        tag = "ATTAINED" if e1["attainable"] else "UNATTAINABLE"
        print(f"  {h}  rate={h_rate[j]:.4f}  E1 {tag:12s} "
              f"w={e1['w_hat']:.3f} resid={e1['residual']:+.4f}  |  "
              f"E2 w={e2['w_hat']:.3f} k={obs_max:+.4f} "
              f"({e2['pct_of_ceiling']:.1f}% ceil)  "
              f"CV held-out {cv['held_out_mean']:+.4f} "
              f"(floor {cv['no_fitting_floor_mean']:+.4f})  "
              f"perm p={nullctl['p_perm']:.3f} "
              f"null-max {nullctl['null_max_mean']:+.4f}", flush=True)

    n_att = sum(o["rate_fit"]["attainable"] for o in out_obs.values())
    best = max(out_obs.values(), key=lambda o: o["trial_fit"]["pct_of_ceiling"])
    print(f"\nE1 attainable for {n_att}/{len(names)} observers")
    print(f"E2 best over all observers and all w: "
          f"{best['trial_fit']['value_at_w_hat']:+.4f} = "
          f"{best['trial_fit']['pct_of_ceiling']:.1f}% of the human-human ceiling")
    print(f"  its cross-validated held-out kappa: "
          f"{best['cross_validation']['held_out_mean']:+.4f} "
          f"({best['cross_validation']['held_out_pct_of_ceiling']:.1f}% of ceiling), "
          f"no-fitting floor {best['cross_validation']['no_fitting_floor_mean']:+.4f}")
    print(f"  its permutation p: {best['permutation_null']['p_perm']:.4f}")

    out = {
        "model_name": args.model_name,
        "checkpoint": args.checkpoint,
        "w_step": args.w_step,
        "n_boot": args.n_boot,
        "n_splits": args.n_splits,
        "n_perm": args.n_perm,
        "boot_seed": args.boot_seed,
        "human_shape_bias": float(h_rate.mean()),
        "human_human_ceiling": CEIL,
        "human_human_sd": float(np.nanstd(hh, ddof=1)),
        "affine_identity_max_dev": dev,
        "family_rate_min": float(np.nanmin(sb_curve)),
        "family_rate_max": float(np.nanmax(sb_curve)),
        "n_rate_attainable": int(n_att),
        "observers": out_obs,
    }
    path = os.path.join(args.out_dir, f"{args.model_name}_wfit.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    main()
