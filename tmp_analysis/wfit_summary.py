"""Collect the §106 per-observer w-fits across all nine runs into the tables.

Reads results/fit_w_observers/*_wfit.json and reports, per §106.5, per-run
values with the RANGE across seeds rather than a p-value, for three arms:
diet/best, diet/final, no-diet/final.

Reports in the order the decisions are made:

  1. E1, the rate fit -- how many observers are attainable at all, and the
     residual when none are. §106.3/§106.7 prediction 1.
  2. E2, the trial fit -- the fitted weight per observer and its kappa, with the
     spread across observers that says whether one weighting serves them all.
  3. Control A (cross-validation) and Control B (permutation null), which
     between them decide whether E2's numbers survive their own selection.
     §106.10.

The verdict rule was fixed in §106.10 before these ran: held-out kappa near the
in-sample value AND the observed maximum outside the permutation null means
real observer-specific structure and a positive to be led with; held-out kappa
collapsing to the no-fitting floor means the bound stands.

Also cross-checks the fine grid against the PUBLISHED coarse grid in
results/sweep_error_consistency/, which is the provenance link between this
section and §2.6.2.
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
    return f"{fmt.format(np.nanmean(v))} [{fmt.format(np.nanmin(v))}, {fmt.format(np.nanmax(v))}]"


def main():
    runs = {}
    for p in sorted(glob.glob(os.path.join(FIT_DIR, "*_wfit.json"))):
        d = json.load(open(p))
        runs[d["model_name"]] = d
    if not runs:
        raise SystemExit(f"no fits in {FIT_DIR}")
    print(f"loaded {len(runs)} runs: {', '.join(sorted(runs))}\n")

    any_run = next(iter(runs.values()))
    names = sorted(any_run["observers"])
    CEIL = any_run["human_human_ceiling"]
    print(f"human-human ceiling kappa = {CEIL:.4f} "
          f"(sd {any_run['human_human_sd']:.4f}); "
          f"human shape bias {any_run['human_shape_bias']:.4f}")
    print(f"grid step {any_run['w_step']}, bootstrap {any_run['n_boot']}, "
          f"splits {any_run['n_splits']}, permutations {any_run['n_perm']}")
    for r in runs.values():
        assert abs(r["human_human_ceiling"] - CEIL) < 1e-9, "ceiling differs"
    print()

    # ---- provenance: fine grid vs the published coarse grid ---------------
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
        # The fine grid contains every coarse w, so its rate range must
        # CONTAIN the coarse one, and its per-observer maxima can only be
        # >= the coarse maxima. Anything else means the two describe
        # different models.
        assert d["family_rate_min"] <= c_sb.min() + 1e-9, name
        assert d["family_rate_max"] >= c_sb.max() - 1e-9, name
        c_best = max(max(v for v in r["kappa_per_human"].values()
                         if v is not None and np.isfinite(v))
                     for r in c["sweep"])
        f_best = max(d["observers"][h]["trial_fit"]["value_at_w_hat"]
                     for h in names)
        assert f_best >= c_best - 1e-9, f"{name}: fine {f_best} < coarse {c_best}"
        print(f"  {name:24s} rate {d['family_rate_min']:.4f}-{d['family_rate_max']:.4f} "
              f"(coarse {c_sb.min():.4f}-{c_sb.max():.4f})  "
              f"best kappa fine {f_best:+.4f} >= coarse {c_best:+.4f}  OK")
    print()

    # ---- E1: the rate fit --------------------------------------------------
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
    print("\n  per observer, |rate - closest reachable|, over the diet arm:")
    diet = [m for m in ARMS["diet / best saved"] + ARMS["diet / final"]
            if m in runs]
    for h in names:
        rs = [abs(runs[m]["observers"][h]["rate_fit"]["residual"]) for m in diet]
        at = sum(runs[m]["observers"][h]["rate_fit"]["attainable"] for m in diet)
        fa = [runs[m]["observers"][h]["rate_fit"]["boot_frac_attainable"]
              for m in diet]
        print(f"    {h}  rate={runs[diet[0]]['observers'][h]['observer_rate']:.4f}  "
              f"|resid| {rng_str(rs)}  attainable in {at}/{len(diet)} runs  "
              f"bootstrap-attainable {np.mean(fa):.3f}")
    print()

    # ---- E2: the trial fit -------------------------------------------------
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
            pc = [runs[m]["observers"][h]["trial_fit"]["pct_of_ceiling"]
                  for m in present]
            bd = sum(runs[m]["observers"][h]["trial_fit"]["at_w_boundary"]
                     for m in present)
            w_all += ws; k_all += ks
            print(f"    {h}  w {rng_str(ws, '{:.3f}')}  "
                  f"kappa {rng_str(ks, '{:+.4f}')}  "
                  f"{np.mean(pc):5.1f}% ceil  boundary {bd}/{len(present)}")
        print(f"    ---- fitted w across observers and seeds: "
              f"{np.min(w_all):.3f} - {np.max(w_all):.3f} "
              f"(sd across observers {np.std(w_all, ddof=1):.3f})")
        print(f"    ---- best kappa anywhere in this arm: {np.max(k_all):+.4f} "
              f"= {100 * np.max(k_all) / CEIL:.1f}% of ceiling")
    print()

    # ---- Controls ----------------------------------------------------------
    print("=" * 78)
    print("CONTROL A  split-half cross-validation of the fitted weight (§106.10)")
    print("=" * 78)
    for arm, members in ARMS.items():
        present = [m for m in members if m in runs]
        if not present:
            continue
        ins = [runs[m]["observers"][h]["cross_validation"]["in_sample_mean"]
               for m in present for h in names]
        out = [runs[m]["observers"][h]["cross_validation"]["held_out_mean"]
               for m in present for h in names]
        flo = [runs[m]["observers"][h]["cross_validation"]["no_fitting_floor_mean"]
               for m in present for h in names]
        above = [runs[m]["observers"][h]["cross_validation"]["held_out_above_floor"]
                 for m in present for h in names]
        n_pos = sum(a > 0 for a in above)
        print(f"  {arm:18s} in-sample {rng_str(ins, '{:+.4f}')}")
        print(f"  {'':18s} held-out  {rng_str(out, '{:+.4f}')}  "
              f"({100 * np.mean(out) / CEIL:.1f}% of ceiling)")
        print(f"  {'':18s} no-fit floor {rng_str(flo, '{:+.4f}')}  "
              f"({100 * np.mean(flo) / CEIL:.1f}% of ceiling)")
        print(f"  {'':18s} held-out ABOVE floor in {n_pos}/{len(above)} "
              f"observer-runs, by {rng_str(above, '{:+.4f}')}")
    print()

    print("=" * 78)
    print("CONTROL B  permutation null for the selected maximum (§106.10)")
    print("=" * 78)
    for arm, members in ARMS.items():
        present = [m for m in members if m in runs]
        if not present:
            continue
        obs = [runs[m]["observers"][h]["permutation_null"]["observed_max"]
               for m in present for h in names]
        nm = [runs[m]["observers"][h]["permutation_null"]["null_max_mean"]
              for m in present for h in names]
        p = [runs[m]["observers"][h]["permutation_null"]["p_perm"]
             for m in present for h in names]
        sig = sum(x < 0.05 for x in p)
        print(f"  {arm:18s} observed max {rng_str(obs, '{:+.4f}')}  "
              f"null max {rng_str(nm, '{:+.4f}')}")
        print(f"  {'':18s} p < .05 in {sig}/{len(p)} observer-runs   "
              f"p range [{min(p):.4f}, {max(p):.4f}]")
    print()

    # ---- the headline ------------------------------------------------------
    print("=" * 78)
    print("VERDICT under the rule fixed in §106.10 BEFORE these ran")
    print("=" * 78)
    best = None
    for m, d in runs.items():
        for h in names:
            o = d["observers"][h]
            if best is None or (o["trial_fit"]["value_at_w_hat"]
                                > best[2]["trial_fit"]["value_at_w_hat"]):
                best = (m, h, o)
    m, h, o = best
    cv, nl = o["cross_validation"], o["permutation_null"]
    print(f"  best single fit anywhere: {h} in {m}")
    print(f"    w = {o['trial_fit']['w_hat']:.3f}  "
          f"kappa = {o['trial_fit']['value_at_w_hat']:+.4f} "
          f"= {o['trial_fit']['pct_of_ceiling']:.1f}% of the human-human ceiling")
    print(f"    cross-validated held-out kappa {cv['held_out_mean']:+.4f} "
          f"({cv['held_out_pct_of_ceiling']:.1f}% of ceiling), "
          f"no-fitting floor {cv['no_fitting_floor_mean']:+.4f} "
          f"({cv['floor_pct_of_ceiling']:.1f}%)")
    print(f"    generalisation gap {cv['generalisation_gap']:+.4f}; "
          f"held-out above floor by {cv['held_out_above_floor']:+.4f}")
    print(f"    permutation p = {nl['p_perm']:.4f}, "
          f"null max mean {nl['null_max_mean']:+.4f}")

    all_out = [runs[m_]["observers"][h_]["cross_validation"]["held_out_mean"]
               for m_ in runs for h_ in names]
    all_flo = [runs[m_]["observers"][h_]["cross_validation"]["no_fitting_floor_mean"]
               for m_ in runs for h_ in names]
    all_att = sum(runs[m_]["n_rate_attainable"] for m_ in runs)
    print(f"\n  E1 attainable in {all_att}/{10 * len(runs)} observer-runs overall")
    print(f"  Control A over everything: held-out {np.mean(all_out):+.4f} "
          f"vs floor {np.mean(all_flo):+.4f} "
          f"({100 * np.mean(all_out) / CEIL:.1f}% vs "
          f"{100 * np.mean(all_flo) / CEIL:.1f}% of ceiling)")
    print(f"  Highest held-out kappa of any observer-run: "
          f"{np.max(all_out):+.4f} = {100 * np.max(all_out) / CEIL:.1f}% of ceiling")


if __name__ == "__main__":
    main()
