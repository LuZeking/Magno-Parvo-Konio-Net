"""Tests for tmp_analysis/figs_v3/supp_tables_model.py: synthetic fixtures carrying the real files' dtypes, and a
real-data regression of the S7 reimplementation against PAPER_v2's published BA rows (Supplementary Table S7)
when the BA files are reachable. Run before submitting slurm/supp_tables_model.sh.

Usage: PYTHONPATH=.:tmp_analysis:tmp_analysis/figs_v2 python tmp_analysis/test_supp_tables_model.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE, os.path.join(HERE, "figs_v2"), os.path.join(HERE, "figs_v3")):
    if p not in sys.path:
        sys.path.insert(0, p)
import supp_tables_model as M  # noqa: E402
import hierarchy_permutation_null as hpn  # noqa: E402

BA_ENC = "/home/byrong/projects/TSN/Magno-Parvo-Konio-Net/.claude/worktrees/mpk-rewrite/results/encoding_cc"
BA_VPREC = os.path.join(ROOT, "results", "vp_recovery_realnc")


def test_wilcoxon_floor():
    w = M.wil(-np.arange(1, 9, dtype=float))
    assert w["n_neg"] == 8 and w["n"] == 8 and abs(w["p"] - 0.0078125) < 1e-9, w
    assert M.pfmt(w["p"]) == ".0078" and M.pfmt(1.0) == "1.00"


def test_exact_null():
    perms = hpn.distinct_labelings(hpn.HIER)
    assert len(perms) == 630
    rng = np.random.default_rng(0)
    base = np.array([1.0, 0.9, 0.5, 0.4, 0.1, 0.0, -0.5])
    prof = base[None, :] + rng.normal(scale=0.01, size=(8, 7))
    r = M.s1_row(prof, hpn.HIER, perms)
    assert r["is_most_extreme"] and abs(r["p_exact_one_sided"] - 1 / 630) < 1e-12 and r["observed_mean_rho"] < -0.95, r
    r2 = M.s1_row(-prof, hpn.HIER, perms)          # an increasing profile is the least extreme in the declared direction
    assert r2["p_exact_one_sided"] == 1.0 and not r2["is_most_extreme"], r2
    assert len(r["per_subject_rho"]) == 8 and r["n_permutations"] == 630


def test_cc_streams_synthetic(tmp):
    enc = os.path.join(tmp, "enc"); os.makedirs(enc)
    rng = np.random.default_rng(1)
    exp = {}
    for s in range(1, 9):
        n = 500
        nc = rng.uniform(0, 0.4, n).astype(np.float64)
        m = rng.normal(size=n).astype(np.float64); p = rng.normal(size=n).astype(np.float64)
        m[:5] = np.nan
        np.savez(os.path.join(enc, f"run_x_subj{s:02d}.npz"), subject=np.int64(s), model_name=np.str_("run_x"), noise_ceiling=nc,
                 unique_magno=rng.uniform(0, 1, n), unique_parvo=rng.uniform(0, 1, n), cc_magno_vsb=m, cc_parvo_vsb=p,
                 **{"roi_prf-visualrois": rng.integers(0, 8, n).astype(np.int64)})
        ok = nc > 0.1
        exp[s] = (np.median(m[ok & np.isfinite(m)]), np.median(p[ok & np.isfinite(p)]))
    old = M.ENC; M.ENC = enc
    try:
        mg, pv = M.cc_streams("run_x")
    finally:
        M.ENC = old
    assert np.allclose(mg, [exp[s][0] for s in range(1, 9)]) and np.allclose(pv, [exp[s][1] for s in range(1, 9)])


def test_forced_estimator_synthetic(tmp):
    import forced_choice as FC
    cats = sorted(FC.CI)
    assert len(cats) == 16
    n, k = 1200, 300
    shapes = np.array([cats[i % 16] for i in range(n)], dtype="<U8")
    textures = np.array([cats[(i + 1) % 16] for i in range(n)], dtype="<U8")
    logits = np.zeros((n, 112), np.float32)
    for i in range(n):
        logits[i, FC.CI[shapes[i] if i < k else textures[i]]] = 1.0
    np.savez(os.path.join(tmp, "x_seed0_logits.npz"), shapes=shapes, textures=textures, full=logits)
    rows = FC.arm(os.path.join(tmp, "x_seed?_logits.npz"), "full")
    assert len(rows) == 1 and abs(rows[0]["forced"] - k / n) < 1e-9 and abs(rows[0]["restricted"] - k / n) < 1e-9 and rows[0]["decided"] == n, rows


def test_rel_auc():
    d = {"families": {fam: {"accuracy": [0.8, 0.4, 0.2]} for fam in M.FAM}}
    a = M.rel_auc(d)
    assert all(abs(a[fam] - np.mean([1.0, 0.5, 0.25])) < 1e-12 for fam in M.FAM)


def test_s3_cell_map():
    q1 = {"chromatic_none": [-0.08, -0.07, -0.06], "chromatic_transform": [-0.02, -0.01, -0.01], "achromatic_none": [-0.18, -0.18, -0.17],
          "achromatic_transform": [0.33, 0.30, 0.31]}
    q0 = {k: [v * 2 for v in vs] for k, vs in q1.items()}
    inter = lambda c: (np.mean(c["achromatic_transform"]) - np.mean(c["achromatic_none"])) - (np.mean(c["chromatic_transform"]) - np.mean(c["chromatic_none"]))
    two = {"Q1": {"cells": q1, "interaction": inter(q1)}, "Q0": {"cells": q0, "interaction": inter(q0)}}
    key = {("control", "parvo"): "chromatic_none", ("channel swap", "parvo"): "chromatic_transform", ("control", "magno"): "achromatic_none",
           ("model", "magno"): "achromatic_transform"}
    arm_of = {r: a for a, rs in M.ARMS.items() for r in rs}
    sd = lambda r: int(r[-1])
    shared = lambda r, st: q0[key[(arm_of[r], st)]][sd(r)] if (arm_of[r], st) in key else -0.9 - 0.01 * sd(r)
    conflict = lambda r, cond, k: (600 if k == "decided" else 0.5)
    ex = {r: {"parvo": {"Q1": {"bias": -0.15 - 0.01 * sd(r)}}} for r in M.MODEL}
    out, rows = M.s3_rows(two, ex, shared, conflict)
    assert rows[2].startswith("| chromatic (L−M) | none | control, L−M stream |"), rows[2]
    assert rows[5].startswith("| achromatic (luminance) | yes | model, luminance stream |") and "+0.313 [0.030]" in rows[5], rows[5]
    assert out["cells"]["achromatic_transform"]["runs"] == M.MODEL and out["cells"]["chromatic_none"]["runs"] == M.CONTROL
    assert abs(out["interaction_held_out"] - inter(q1)) < 1e-12 and abs(out["ramp_effect_achromatic"] - (0.3133333 + 0.1766667)) < 1e-6
    assert "same cell read in the model" in rows[6] and "−0.160 [0.020]" in rows[6], rows[6]


def test_s7_ba_regression():
    """PAPER_v2 Supplementary Table S7 rows, recomputed with the generator's reimplementation on the BA files."""
    if not (os.path.isdir(BA_ENC) and len(os.listdir(BA_VPREC)) >= 16):
        print("  (skipped: BA files not reachable)"); return
    d = M.floor_variants("dkl_bcs_seed0_trained", BA_VPREC, BA_ENC)
    n = M.floor_variants("dkl_base_seed0_trained", BA_VPREC, BA_ENC)
    assert abs(d["floor_group"] - 0.0375) < 5e-4 and abs(n["floor_group"] - 0.0511) < 5e-4, (d["floor_group"], n["floor_group"])
    assert abs(d["a_slope_intercept_fixed_at_group_floor"] - 1.210) < 5e-4 and abs(n["a_slope_intercept_fixed_at_group_floor"] - 1.281) < 5e-4
    assert abs(np.mean(d["real_per_subject"]) + 0.8335) < 5e-4 and abs(np.mean(n["real_per_subject"]) + 0.0785) < 5e-4
    exp = {"group floor, minimal": (-0.720, 8, -0.101, 8), "per-subject floor, minimal": (-0.720, 8, -0.101, 8),
           "group floor, + residual before scaling": (-0.626, 8, -0.034, 6), "per-subject floor, + residual before scaling": (-0.626, 8, -0.034, 5),
           "group floor, + residual after scaling": (-0.607, 8, -0.015, 4), "per-subject floor, + residual after scaling": (-0.607, 8, -0.015, 5)}
    for k, (md, nd_, mn, nn) in exp.items():
        assert abs(d["variants"][k]["mean"] - md) < 5e-4 and d["variants"][k]["n_neg"] == nd_, (k, d["variants"][k])
        assert abs(n["variants"][k]["mean"] - mn) < 5e-4 and n["variants"][k]["n_neg"] == nn, (k, n["variants"][k])
    # PAPER_v2's hand-entered "rho vs rank +0.54, p = .22" for the no-diet geometry is reproduced by neither the mean
    # per-subject rho (0.719) nor the rho of the mean profile (0.964); the model's row uses the former and says so.
    assert len(n["per_roi_floor_rho_vs_rank"]["per_subject"]) == 8 and abs(n["per_roi_floor_rho_vs_rank"]["rho"] - 0.719) < 5e-3, n["per_roi_floor_rho_vs_rank"]


def main() -> int:
    tests = [test_wilcoxon_floor, test_exact_null, test_cc_streams_synthetic, test_forced_estimator_synthetic, test_rel_auc, test_s3_cell_map,
             test_s7_ba_regression]
    fails = 0
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR") or None) as tmp:
        for t in tests:
            try:
                (t(tmp) if t.__code__.co_argcount else t())
                print(f"ok   {t.__name__}")
            except Exception as e:  # noqa: BLE001
                fails += 1
                print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    print(f"{len(tests) - fails}/{len(tests)} passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
