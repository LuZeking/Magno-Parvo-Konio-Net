"""Consistency gate of docs/PLAN_SUPP_MODEL_2026-11-03.md section 3.2: every model-of-record number in
results/supp_v3/supp_tables_model.json that has a counterpart in the main text, Table 1, Extended Data Table 1,
results/figures_v3/{cortex_controls,ext_gradient}.json or the committed cortex summary must equal it.
Literals are the main text's (copied from tmp_analysis/verify_v3_claims.py, which re-derives them from the same
sources), compared at their printed precision; record-to-record comparisons are exact (1e-6).

Used standalone (prints every comparison, exit 1 on any mismatch) and by verify_v3_claims.check_supp_model().

Usage: python tmp_analysis/figs_v3/supp_gate.py [--json results/supp_v3/supp_tables_model.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
MODEL = [f"fs2_dvd300_avg_w050_ls_seed{s}" for s in range(3)]
CONTROL = [f"fs2_dvdfix300_avg_w050_ls_seed{s}" for s in range(3)]
SWAP = [f"fs2_dvd300swap_avg_w050_ls_seed{s}" for s in range(3)]
CORTEX = os.path.join(ROOT, "docs", "slides_2026-10-12_ba_vs_fusion", "figures", "fs2_cortex_summary.json")
CONTROLS = os.path.join(ROOT, "results", "figures_v3", "cortex_controls.json")


def _lit(s: str) -> tuple[float, float]:
    """value and half-unit tolerance of a printed literal ('−0.156' -> (-0.156, 0.0005))."""
    s = s.replace("−", "-").replace("%", "")
    d = len(s.split(".")[1]) if "." in s else 0
    return float(s), 0.5 * 10 ** (-d) + 1e-9


def gate(J: dict) -> list[tuple[str, bool, str]]:
    T = J["tables"]
    out: list[tuple[str, bool, str]] = []

    def lit(name, got, s):
        v, tol = _lit(s)
        out.append((name, abs(got - v) <= tol, f"{got:+.4f} vs {s}"))

    def eq(name, a, b, tol=1e-6):
        a, b = np.asarray(a, float), np.asarray(b, float)
        out.append((name, a.shape == b.shape and np.allclose(a, b, atol=tol, rtol=0), f"max|Δ| = {np.max(np.abs(a - b)) if a.shape == b.shape else 'shape'}"))

    C = json.load(open(CORTEX))["arms"]
    CC = json.load(open(CONTROLS))
    # ---- S1: rho and gradient per seed on the per-voxel index against the committed cortex summary and ED Table 1
    if "S1" in T:
        rows = {(r["run"], r["stat"]): r for r in T["S1"]["rows"]}
        for arm, runs, key in (("model", MODEL, "dvd300"), ("control", CONTROL, "control"), ("channel swap", SWAP, "swap")):
            eq(f"S1 {arm}: rho per seed = cortex summary", [rows[(r, "raw")]["observed_mean_rho"] for r in runs], C[key]["rho_per_seed"])
            eq(f"S1 {arm}: gradient per seed = cortex summary", [rows[(r, "raw")]["gradient_mean"] for r in runs], C[key]["grad_per_seed"])
        for r, s in zip(MODEL, ("−1.259", "−1.291", "−1.262")):
            lit(f"S1 ED T1 model gradient {r[-5:]}", rows[(r, "raw")]["gradient_mean"], s)
        for r, s in zip(CONTROL, ("−0.062", "−0.062", "−0.086")):
            lit(f"S1 ED T1 control gradient {r[-5:]}", rows[(r, "raw")]["gradient_mean"], s)
        for r, s in zip(SWAP, ("−0.015", "−0.016", "−0.025")):
            lit(f"S1 ED T1 swap gradient {r[-5:]}", rows[(r, "raw")]["gradient_mean"], s)
        rho = [rows[(r, "raw")]["observed_mean_rho"] for r in MODEL]
        lit("S1 Results 2.3 model rho, weakest", max(rho), "−0.964"); lit("S1 Results 2.3 model rho, strongest", min(rho), "−0.973")
        out.append(("S1 model: most extreme of 630 in every seed, both statistics", all(rows[(r, s)]["is_most_extreme"] for r in MODEL for s in ("raw", "norm")), ""))
    # ---- S2: in-silico cue conflict against the cortex summary and Results 2.3
    if "S2" in T:
        P = T["S2"]["per_run"]
        for arm, runs, key in (("model", MODEL, "dvd300"), ("control", CONTROL, "control"), ("channel swap", SWAP, "swap")):
            eq(f"S2 {arm}: magno − parvo per seed = cortex summary", [P[r]["diff_mean"] for r in runs], C[key]["cc_per_seed"])
        d = [P[r]["diff_mean"] for r in MODEL]
        lit("S2 Results 2.3 model, lowest seed", min(d), "+1.32"); lit("S2 Results 2.3 model, highest seed", max(d), "+1.34")
        lit("S2 Results 2.3 control", np.mean([P[r]["diff_mean"] for r in CONTROL]), "+0.07")
        lit("S2 Results 2.3 swap", np.mean([P[r]["diff_mean"] for r in SWAP]), "−0.01")
    # ---- S3: the 2x2 and Table 1
    if "S3" in T:
        S = T["S3"]; m = S["means_held_out"]
        for k, s in (("achromatic_transform", "+0.316"), ("achromatic_none", "−0.177"), ("chromatic_none", "−0.069"), ("chromatic_transform", "−0.016")):
            lit(f"S3 2x2 cell {k}", m[k], s)
        lit("S3 interaction, held out", S["interaction_held_out"], "+0.44"); lit("S3 interaction, shared", S["interaction_shared"], "+0.73")
        lit("S3 ramp effect, achromatic", S["ramp_effect_achromatic"], "+0.49"); lit("S3 ramp effect, chromatic", S["ramp_effect_chromatic"], "+0.05")
        c = S["cells"]
        lit("S3 Table 1 model magno shared", np.mean(c["achromatic_transform"]["probe_shared"]), "+0.578")
        lit("S3 Table 1 model magno cue-conflict", np.mean(c["achromatic_transform"]["cue_conflict"]), "0.850")
        lit("S3 Table 1 model parvo held out", np.mean(c["chromatic_none_model"]["probe_held_out"]), "−0.156")
        lit("S3 Table 1 model parvo shared", np.mean(c["chromatic_none_model"]["probe_shared"]), "−0.454")
        lit("S3 Table 1 model parvo cue-conflict", np.mean(c["chromatic_none_model"]["cue_conflict"]), "0.477")
        lit("S3 Table 1 control magno shared", np.mean(c["achromatic_none"]["probe_shared"]), "−0.291")
        lit("S3 Table 1 control parvo shared", np.mean(c["chromatic_none"]["probe_shared"]), "−0.413")
        lit("S3 Table 1 control magno cue-conflict", np.mean(c["achromatic_none"]["cue_conflict"]), "0.384")
        lit("S3 Table 1 control parvo cue-conflict", np.mean(c["chromatic_none"]["cue_conflict"]), "0.426")
        lit("S3 Table 1 swap parvo shared", np.mean(c["chromatic_transform"]["probe_shared"]), "−0.276")
        lit("S3 Table 1 swap parvo cue-conflict", np.mean(c["chromatic_transform"]["cue_conflict"]), "0.390")
    # ---- S4: readout ladder against Table 1, top-1 against Results 2.1
    if "S4" in T:
        A = T["S4"]["arms"]
        for arm, lits in (("model", ("0.850", "0.682", "0.477")), ("control", ("0.384", "0.374", "0.426")), ("channel swap", ("0.397", "0.389", "0.390"))):
            for cond, s in zip(("magno_only", "full", "parvo_only"), lits):
                lit(f"S4 Table 1 {arm} {cond}", np.mean(A[arm]["readouts"][cond]["shape_bias"]), s)
                # not a plan gate item: the stored affine sweep and the stored per-condition evaluation are two pipelines; the
                # table reports the latter (Table 1's); the sweep point is kept as an informational cross-check, bounded at 0.03
                out.append((f"S4 {arm} {cond}: stored per-condition rate within 0.03 of the affine sweep's point (informational)",
                            A[arm]["readouts"][cond]["max_abs_diff_vs_sweep"] < 3e-2, f"max|Δ| = {A[arm]['readouts'][cond]['max_abs_diff_vs_sweep']:.4f}"))
        lit("S4 Results 2.1 model top-1", np.mean(A["model"]["accuracy_final"]), "0.683")
        lit("S4 Results 2.1 control top-1", np.mean(A["control"]["accuracy_final"]), "0.764")
        lit("S4 Results 2.2 swap top-1", np.mean(A["channel swap"]["accuracy_final"]), "0.760")
        out.append(("S4 parameters identical across arms", len({A[a]["parameters"]["total"] for a in A}) == 1, str({a: A[a]["parameters"]["total"] for a in A})))
    # ---- S5: OOD against Results 2.4
    if "S5" in T:
        F = T["S5"]["per_family"]
        for fam, s in (("uniform-noise", "−0.135"), ("high-pass", "−0.136"), ("false-colour", "−0.084"), ("rotation", "−0.040"),
                       ("phase-scrambling", "+0.009"), ("low-pass", "+0.078"), ("contrast", "−0.004")):
            lit(f"S5 Results 2.4 OOD {fam}", F[fam]["model_minus_control"], s)
        out.append(("S5 complete seed separation on the four families", all(F[f]["separated_3v3"] for f in ("uniform-noise", "high-pass", "false-colour", "rotation")), ""))
        out.append(("S5 contrast: seeds interleaved", not F["contrast"]["separated_3v3"], ""))
    # ---- S6: kappa against Results 2.4
    if "S6" in T:
        R = T["S6"]["runs"]
        pct = lambda runs, key: np.mean([R[r][key]["kappa_pct_ceiling"] for r in runs])
        lit("S6 κ model magno % ceiling", pct(MODEL, "w1"), "17.3"); lit("S6 κ model integrated", pct(MODEL, "w05"), "13.2")
        lit("S6 κ control integrated", pct(CONTROL, "w05"), "11.9"); lit("S6 κ control magno", pct(CONTROL, "w1"), "12.6")
        lit("S6 κ best sweep readout, model", max(R[r]["best"]["kappa_pct_ceiling"] for r in MODEL), "29.0")
        for r, k in zip(MODEL, ("5.1", "25.6", "21.3")):
            lit(f"S6 {r[-5:]} magno κ %", R[r]["w1"]["kappa_pct_ceiling"], k)
        if "S4" in T:   # the sweep's w = 1 point and the magno_only evaluation are the same readout up to the sweep's grid (≤ 0.005)
            d = [abs(R[r]["w1"]["shape_bias"] - T["S4"]["arms"]["model"]["readouts"]["magno_only"]["shape_bias"][i]) for i, r in enumerate(MODEL)]
            out.append(("S6 w = 1 rate within 0.005 of the magno_only readout, every seed", max(d) < 5e-3, f"max|Δ| = {max(d):.4f}"))
        lit("S6 human-human ceiling", R[MODEL[0]]["ceiling"], "0.234")
    # ---- S7: against cortex_controls.json's estimator-bias block (same sources, same code path)
    if "S7" in T:
        sb, eb = T["S7"]["sim_block"], CC["estimator_bias"]
        for arm in ("model", "control"):
            for k in ("floor_norm", "floor_raw", "displacement_norm"):
                eq(f"S7 {arm} {k} = cortex_controls.json", sb[arm][k]["per_subject"], eb[arm][k]["per_subject"])
            eq(f"S7 {arm} real contrast seed 0 = cortex_controls.json", sb[arm]["real_norm_seed0"], eb[arm]["real_norm_seed0"])
            eq(f"S7 {arm} recovery ratio = cortex_controls.json", [sb[arm]["recovery_A_est_over_true"]], [eb[arm]["recovery_A_est_over_true"]])
            V = T["S7"]["variants"][arm]
            eq(f"S7 {arm} variants: real per subject = sim block", V["real_per_subject"], sb[arm]["real_norm_seed0"])
            eq(f"S7 {arm} variants: group floor = mean floor", [V["floor_group"]], [sb[arm]["floor_norm"]["mean"]])
        eq("S7 displacement ratio model/control", [sb["displacement_ratio_model_over_control"]], [eb["displacement_ratio_model_over_control"]])
    # ---- S8: partial slopes are the stored record; univariate slopes new
    if "S8" in T:
        R = T["S8"]["runs"]
        for run in MODEL + CONTROL + SWAP + ["untrained_dvd300_seed0"]:
            for stat in R[run]:
                reg = CC["per_seed_regression"][run][stat]
                eq(f"S8 {run[-5:] if run != 'untrained_dvd300_seed0' else 'untrained'} {stat}: ecc partial = cortex_controls.json",
                   R[run][stat]["ecc_partial"]["per_subject"], [reg[str(s)]["ecc_partial"] for s in range(1, 9)])
                eq(f"S8 {run[-5:] if run != 'untrained_dvd300_seed0' else 'untrained'} {stat}: size partial = cortex_controls.json",
                   R[run][stat]["size_partial"]["per_subject"], [reg[str(s)]["size_partial"] for s in range(1, 9)])
        lit("S8 VIF, model seed 0", R[MODEL[0]]["norm"]["vif_mean"], f"{CC['vif_mean_model_seed0']:.3f}")
    # ---- S9: the forced estimator against Table 1 / Results 2.2
    if "S9" in T:
        S = T["S9"]
        for arm, lits in (("model", ("0.730", "0.648", "0.520")), ("control", ("0.393", "0.383", "0.471"))):
            for lab, s in zip(("magno stream alone (w = 1)", "both streams (w = 0.5)", "parvo stream alone (w = 0)"), lits):
                lit(f"S9 forced {arm} {lab}", np.mean(S[f"{arm}, {lab}"]["forced"]), s)
        for arm, lits in (("model", ("0.850", "0.682", "0.477")), ("control", ("0.384", "0.374", "0.426")), ("channel swap", ("0.397", "0.389", "0.390"))):
            for lab, s in zip(("magno stream alone (w = 1)", "both streams (w = 0.5)", "parvo stream alone (w = 0)"), lits):
                lit(f"S9 restricted {arm} {lab}", np.mean(S[f"{arm}, {lab}"]["restricted"]), s)
        dm = [np.mean(S[f"model, {lab}"]["decided"]) for lab, _, _ in ((l, 0, 0) for l in ("magno stream alone (w = 1)", "both streams (w = 0.5)", "parvo stream alone (w = 0)"))]
        lit("S9 decided, model fewest", min(dm), "426"); lit("S9 decided, model most", max(dm), "580")
    # ---- S10: the recomputed fs2 arms reproduce the stored atlas record
    if "S10" in T:
        out.append(("S10 recomputed untrained/control/model = ext_gradient.json", T["S10"]["recompute_max_abs_diff_vs_stored"] < 1e-6,
                    f"max|Δ| = {T['S10']['recompute_max_abs_diff_vs_stored']:.2e} over {T['S10']['n_compared']} cells"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=os.path.join(ROOT, "results", "supp_v3", "supp_tables_model.json"))
    args = ap.parse_args()
    res = gate(json.load(open(args.json)))
    bad = [r for r in res if not r[1]]
    for name, ok, detail in res:
        print(f"{'ok  ' if ok else 'FAIL'} {name}  {detail}")
    print(f"\n{len(res) - len(bad)}/{len(res)} comparisons agree; {len(bad)} mismatches")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
