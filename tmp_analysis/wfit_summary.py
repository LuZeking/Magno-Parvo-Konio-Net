"""Collect the §106 per-observer w-fits across all nine runs into the tables.

Reads results/fit_w_observers/*_wfit.json and reports, per §106.5, per-run
values with the RANGE across seeds rather than a p-value, for three arms:
diet/best, diet/final, no-diet/final.

Order of reporting follows the order the decisions are made:

  1. Provenance -- the fine grid against the PUBLISHED coarse grid in
     results/sweep_error_consistency/, which is the link to §2.6.2.
  2. E1, the rate fit -- how many observers are reachable at all (§106.7's
     prediction 1).
  3. E2, the trial fit -- the fitted weight per observer, under BOTH
     normalisations (§106.11).
  4. Control A, cross-validation -- the full ten-value held-out vector, never
     its maximum, plus the joint form with observer selection inside the fold.
  5. Control B, the joint permutation null bounding both selections, with the
     winner tallies that say whether the per-observer denominator is driving
     the argmax (§106.12).

The verdict rule was fixed in §106.10 BEFORE any of this ran and is not
renegotiated here: held-out kappa near the in-sample value AND the observed
maximum outside the permutation null means real observer-specific structure and
a positive to be led with; held-out kappa collapsing to the no-fitting floor
means the bound stands.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

FIT_DIR = "results/fit_w_observers"
COARSE_DIR = "results/sweep_error_consistency"

ARMS = {
    "diet / best saved": ["dkl_bcs_seed0_best", "dkl_bcs_seed1_best",
                          "dkl_bcs_seed2_best"],
    "diet / final": ["dkl_bcs_seed0_final", "dkl_bcs_seed1_final",
                     "dkl_bcs_seed2_final"],
    "no diet / final": ["dkl_base_seed0_final", "dkl_base_seed1_final",
                        "dkl_base_seed2_final"],
}


def rng_str(v, fmt="{:.4f}"):
    v = np.asarray(v, dtype=float)
    return (f"{fmt.format(np.nanmean(v))} "
            f"[{fmt.format(np.nanmin(v))}, {fmt.format(np.nanmax(v))}]")


def main():
    runs = {}
    for p in sorted(glob.glob(os.path.join(FIT_DIR, "*_wfit.json"))):
        d = json.load(open(p))
        runs[d["model_name"]] = d
    if not runs:
        raise SystemExit(f"no fits in {FIT_DIR}")
    print(f"loaded {len(runs)} runs\n")

    any_run = next(iter(runs.values()))
    names = sorted(any_run["observers"])
    CEIL = any_run["human_human_ceiling"]
    own = {h: any_run["own_ceilings"][h] for h in names}
    print(f"pooled ceiling {CEIL:.4f} (sd {any_run['human_human_sd']:.4f}); "
          f"own ceilings {min(own.values()):.4f}-{max(own.values()):.4f} "
          f"({max(own.values()) / min(own.values()):.2f}x)")
    print(f"human shape bias {any_run['human_shape_bias']:.4f}")
    print(f"grid step {any_run['w_step']}, bootstrap {any_run['n_boot']}, "
          f"splits {any_run['n_splits']}, permutations {any_run['n_perm']}")
    for r in runs.values():
        assert abs(r["human_human_ceiling"] - CEIL) < 1e-9, "ceiling differs"
        for h in names:
            assert abs(r["own_ceilings"][h] - own[h]) < 1e-9, "own ceiling differs"
    print()

    # ---- 1. provenance ----------------------------------------------------
    print("=" * 78)
    print("PROVENANCE  fine grid against the published 0.05 grid (§2.6.2's data)")
    print("=" * 78)
    for name, d in sorted(runs.items()):
        cp = os.path.join(COARSE_DIR, f"{name}_kappa_sweep.json")
        if not os.path.exists(cp):
            print(f"  {name:24s} no coarse file")
            continue
        c = json.load(open(cp))
        c_sb = np.array([r["shape_bias"] for r in c["sweep"]])
        c_w = np.array([r["w"] for r in c["sweep"]])
        # TOLERANCE, and why it is not zero. The coarse sweep scored DIRECT
        # logits clf(w*F1 + (1-w)*F2); the fine grid scores INTERPOLATED ones
        # w*clf(F1) + (1-w)*clf(F2). The affine identity between them holds to
        # 3.6e-05 in the logits (each run asserts this), so the two paths can
        # disagree in principle -- but only if a near-tie flips an argmax, and
        # one flipped image would move shape bias by ~1/500 = 2e-3. Observed
        # disagreement is ~2e-8, i.e. no decision flips anywhere on the grid,
        # which is the check actually worth making. Within each run the
        # comparison is like-for-like and exact (max |d| = 0.00e+00, asserted
        # there).
        TOL = 1e-4
        assert d["family_rate_min"] <= c_sb.min() + TOL, name
        assert d["family_rate_max"] >= c_sb.max() - TOL, name
        c_best = max(max(v for v in r["kappa_per_human"].values()
                         if v is not None and np.isfinite(v))
                     for r in c["sweep"])
        f_best = max(d["observers"][h]["trial_fit"]["value_at_w_hat"]
                     for h in names)
        assert f_best >= c_best - TOL, f"{name}: fine {f_best} < coarse {c_best}"
        # Compare the two paths ONLY at the 21 weights both grids contain.
        # Comparing the ranges instead would conflate real extra resolution
        # (the fine grid legitimately finds more extreme values between the
        # coarse points) with path disagreement, and would report ~4e-3 of
        # honest extra coverage as if it were error.
        npz = os.path.join(FIT_DIR, f"{name}_wfit_curves.npz")
        drift = float("nan")
        if os.path.exists(npz):
            z = np.load(npz, allow_pickle=True)
            g, fsb = z["grid"], z["sb_curve"]
            idx = [int(np.argmin(np.abs(g - w))) for w in c_w]
            drift = float(np.max(np.abs(np.asarray(fsb)[idx] - c_sb)))
        print(f"  {name:24s} rate {d['family_rate_min']:.4f}-{d['family_rate_max']:.4f} "
              f"(coarse {c_sb.min():.4f}-{c_sb.max():.4f})  "
              f"best kappa fine {f_best:+.4f} vs coarse {c_best:+.4f}  "
              f"path drift at shared w {drift:.1e}")
    print()

    # ---- 2. E1 ------------------------------------------------------------
    print("=" * 78)
    print("E1  RATE FIT   is any observer's shape-bias rate reachable at all?")
    print("=" * 78)
    for arm, members in ARMS.items():
        present = [m for m in members if m in runs]
        if not present:
            continue
        att = [runs[m]["n_rate_attainable"] for m in present]
        rmax = [runs[m]["family_rate_max"] for m in present]
        resid = [abs(runs[m]["observers"][h]["rate_fit"]["residual"])
                 for m in present for h in names]
        print(f"  {arm:18s} attainable {sum(att)}/{10 * len(present)} "
              f"observer-runs   family max rate {rng_str(rmax)}   "
              f"|residual| {rng_str(resid)}")
    diet = [m for m in ARMS["diet / best saved"] + ARMS["diet / final"]
            if m in runs]
    print("\n  per observer over the diet arm, |rate - closest reachable|:")
    for h in names:
        rs = [abs(runs[m]["observers"][h]["rate_fit"]["residual"]) for m in diet]
        at = sum(runs[m]["observers"][h]["rate_fit"]["attainable"] for m in diet)
        fa = [runs[m]["observers"][h]["rate_fit"]["boot_frac_attainable"]
              for m in diet]
        print(f"    {h}  rate={runs[diet[0]]['observers'][h]['observer_rate']:.4f}  "
              f"|resid| {rng_str(rs)}  attainable {at}/{len(diet)} runs  "
              f"bootstrap-attainable {np.mean(fa):.3f}")
    print()

    # ---- 3. E2 ------------------------------------------------------------
    print("=" * 78)
    print("E2  TRIAL FIT  the weight that best predicts each observer's choices")
    print("=" * 78)
    for arm, members in ARMS.items():
        present = [m for m in members if m in runs]
        if not present:
            continue
        print(f"\n  {arm}")
        w_all, k_all = [], []
        for h in names:
            ws = [runs[m]["observers"][h]["trial_fit"]["w_hat"] for m in present]
            ks = [runs[m]["observers"][h]["trial_fit"]["value_at_w_hat"]
                  for m in present]
            po = [runs[m]["observers"][h]["trial_fit"]["pct_of_own_ceiling"]
                  for m in present]
            pp = [runs[m]["observers"][h]["trial_fit"]["pct_of_pooled_ceiling"]
                  for m in present]
            bd = sum(runs[m]["observers"][h]["trial_fit"]["at_w_boundary"]
                     for m in present)
            w_all += ws; k_all += ks
            print(f"    {h}  w {rng_str(ws, '{:.3f}')}  "
                  f"kappa {rng_str(ks, '{:+.4f}')}  "
                  f"{np.mean(po):5.1f}% own / {np.mean(pp):5.1f}% pooled  "
                  f"boundary {bd}/{len(present)}")
        print(f"    ---- fitted w spans {np.min(w_all):.3f}-{np.max(w_all):.3f} "
              f"across observers and seeds (sd {np.std(w_all, ddof=1):.3f})")
    print()

    # ---- 4. Control A -----------------------------------------------------
    print("=" * 78)
    print("CONTROL A  cross-validation. FULL held-out vector, never its max")
    print("=" * 78)
    for arm, members in ARMS.items():
        present = [m for m in members if m in runs]
        if not present:
            continue
        print(f"\n  {arm}")
        for h in names:
            ins = [runs[m]["observers"][h]["cross_validation"]["in_sample_mean"]
                   for m in present]
            out = [runs[m]["observers"][h]["cross_validation"]["held_out_mean"]
                   for m in present]
            flo = [runs[m]["observers"][h]["cross_validation"]
                   ["no_fitting_floor_mean"] for m in present]
            print(f"    {h}  in-sample {np.mean(ins):+.4f} -> held-out "
                  f"{np.mean(out):+.4f}  floor {np.mean(flo):+.4f}  "
                  f"above floor {np.mean(out) - np.mean(flo):+.4f}")
        allout = [runs[m]["observers"][h]["cross_validation"]["held_out_mean"]
                  for m in present for h in names]
        allflo = [runs[m]["observers"][h]["cross_validation"]
                  ["no_fitting_floor_mean"] for m in present for h in names]
        npos = sum(a > b for a, b in zip(allout, allflo))
        print(f"    ---- held-out above floor in {npos}/{len(allout)} "
              f"observer-runs")
        for lab in ("raw", "own_ceiling"):
            ins = [runs[m]["joint_cross_validation"][lab]["in_sample_mean"]
                   for m in present]
            out = [runs[m]["joint_cross_validation"][lab]["held_out_mean"]
                   for m in present]
            flo = [runs[m]["joint_cross_validation"][lab]
                   ["no_fitting_floor_mean"] for m in present]
            print(f"    JOINT [{lab:11s}] in-sample {np.mean(ins):+.4f} -> "
                  f"held-out {np.mean(out):+.4f}  floor {np.mean(flo):+.4f}")
    print()

    # ---- 5. Control B -----------------------------------------------------
    print("=" * 78)
    print("CONTROL B  joint permutation null (both selections bounded)")
    print("=" * 78)
    for arm, members in ARMS.items():
        present = [m for m in members if m in runs]
        if not present:
            continue
        print(f"\n  {arm}")
        for lab in ("raw", "own_ceiling"):
            obs = [runs[m]["joint_permutation_null"][lab]["observed_best_of_ten"]
                   for m in present]
            nm = [runs[m]["joint_permutation_null"][lab]["null_mean"]
                  for m in present]
            p95 = [runs[m]["joint_permutation_null"][lab]["null_p95"]
                   for m in present]
            pv = [runs[m]["joint_permutation_null"][lab]["p_perm"]
                  for m in present]
            win = [runs[m]["joint_permutation_null"][lab]["observed_winner"]
                   for m in present]
            rr = [runs[m]["joint_permutation_null"][lab]
                  ["null_wins_vs_inv_ceiling_r"] for m in present]
            print(f"    [{lab:11s}] observed {rng_str(obs, '{:+.4f}')}  "
                  f"null {rng_str(nm, '{:+.4f}')}  p95 {np.mean(p95):+.4f}")
            print(f"    {'':13s} p = {rng_str(pv, '{:.4f}')}   "
                  f"winners {win}   corr(null wins, 1/ceiling) "
                  f"{np.mean(rr):+.3f}")
    # the denominator channel, pooled over runs
    print("\n  null-winner tally over all runs (own_ceiling normalisation):")
    tally = {h: 0 for h in names}
    for m in runs:
        for h, c in runs[m]["joint_permutation_null"]["own_ceiling"][
                "null_winner_counts"].items():
            tally[h] += c
    tot = sum(tally.values())
    for h in sorted(names, key=lambda x: -tally[x]):
        print(f"    {h}  {tally[h]:6d} / {tot}  ({100 * tally[h] / tot:5.1f}%)  "
              f"own ceiling {own[h]:.4f}")
    print()

    # ---- verdict ----------------------------------------------------------
    print("=" * 78)
    print("VERDICT under the rule fixed in §106.10 BEFORE any of this ran")
    print("=" * 78)
    all_att = sum(runs[m]["n_rate_attainable"] for m in runs)
    print(f"  E1 attainable in {all_att}/{10 * len(runs)} observer-runs")
    ho = [runs[m]["observers"][h]["cross_validation"]["held_out_mean"]
          for m in runs for h in names]
    fl = [runs[m]["observers"][h]["cross_validation"]["no_fitting_floor_mean"]
          for m in runs for h in names]
    ins = [runs[m]["observers"][h]["cross_validation"]["in_sample_mean"]
           for m in runs for h in names]
    print(f"  Control A over all {len(ho)} observer-runs: in-sample "
          f"{np.mean(ins):+.4f} -> held-out {np.mean(ho):+.4f}, "
          f"floor {np.mean(fl):+.4f}")
    print(f"    generalisation gap {np.mean(ins) - np.mean(ho):+.4f}; "
          f"held-out above floor by {np.mean(ho) - np.mean(fl):+.4f}")
    print(f"    highest held-out of any observer-run {np.max(ho):+.4f} "
          f"= {100 * np.max(ho) / CEIL:.1f}% of pooled ceiling")
    for lab in ("raw", "own_ceiling"):
        pv = [runs[m]["joint_permutation_null"][lab]["p_perm"] for m in runs]
        nsig = sum(x < 0.05 for x in pv)
        print(f"  Control B [{lab:11s}]: joint p < .05 in {nsig}/{len(pv)} runs, "
              f"p range [{min(pv):.4f}, {max(pv):.4f}]")


if __name__ == "__main__":
    main()
