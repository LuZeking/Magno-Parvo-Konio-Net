"""Corrected permutation null and exemplar-cluster interval for the shape-target
correspondence r_h of the cortex-to-decision test (results/cortex_decision/).

Defect found 2026-10-28, after the registered analysis had run (job 26719989,
code 376fb05): six of the 1,200 cue-conflict images have no resolved observer
trial, so the observers' shape-rate profile h, and h_resid, are NaN there. The
observed r_h masks non-finite images (`pearson`), but the permutation null
(`pearson_rows`) and the exemplar-cluster bootstrap (`cluster_bootstrap_rs`)
did not, so every permuted r on the shape target was NaN: ctd_analysis.json
reports p_h = 1/2001 = .0005 for every arm and region (vacuous), null_h_97.5 =
NaN and an exemplar CI of (nan, nan) for r_h. The resolution target v has no
NaN image and is unaffected, as are B, r_h itself, the Wilcoxon counts, the
input-floor comparison, P-C1, P-C3, P-C4 and the reading of record (P-C1).

This script recomputes, for every arm x region (joint band; seeds averaged per
subject, as the analysis does), the subject-mean r_h, its within-cell
permutation p and the null's 97.5th percentile, and each subject's exemplar
CI of r_h, masking the non-finite images per permutation exactly as `pearson`
masks them. Same permutation sequence (seed 0, 2,000 draws), same bootstrap
seed (1, 1,000 draws). r_h and p_v are recomputed and checked against the
record (p_v with the masked code must equal the record's, since v has no NaN).

Usage (from the worktree):
  python tmp_analysis/ctd_h_null_fix.py  -> results/cortex_decision/ctd_h_null_fix.json
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from ctd_perimage_margins import CODE, FITS, HUMAN, cd, load_joint  # noqa: E402

RECORD = "results/cortex_decision/ctd_analysis.json"
OUT = "results/cortex_decision/ctd_h_null_fix.json"
EXTRASTRIATE = ("hV4", "ventral", "lateral")


def pearson_rows_masked(M: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pearson r of every row of M with y over the images finite in both (per row).
    Equals cortex_decision.pearson_rows when everything is finite."""
    M = np.asarray(M, float)
    y = np.asarray(y, float)
    W = np.isfinite(M) & np.isfinite(y)[None, :]
    n = W.sum(axis=1)
    Mz = np.where(W, M, 0.0)
    Y = np.where(W, y[None, :], 0.0)
    A = np.where(W, Mz - (Mz.sum(axis=1) / n)[:, None], 0.0)
    B = np.where(W, Y - (Y.sum(axis=1) / n)[:, None], 0.0)
    den = np.sqrt((A * A).sum(axis=1) * (B * B).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (A * B).sum(axis=1) / den, np.nan)


def cluster_bootstrap_r_masked(x, y, clusters, n_boot: int, seed: int = 1):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    return cd.cluster_bootstrap_r(x[m], y[m], np.asarray(clusters)[m], n_boot, seed)


def main() -> None:
    rec = json.load(open(RECORD))
    human = cd.load_human(HUMAN)
    classes = sorted(set(human["shapes"]) | set(human["textures"]))
    cells = cd.cell_ids(human["shapes"], human["textures"])
    clusters = cd.object_exemplar(human["stems"])
    n_perm = int(rec["settings"]["n_perm"])
    perms = cd.within_cell_permutations(cells, n_perm)
    h_perm = human["h_resid"][perms]
    v_perm = human["v_resid"][perms]
    finite_h = np.isfinite(human["h_resid"])
    # self-check of the masked estimator on all-finite data
    probe = np.random.default_rng(0).standard_normal((5, len(finite_h)))
    assert np.allclose(pearson_rows_masked(probe, human["v_resid"]), cd.pearson_rows(probe, human["v_resid"]), atol=1e-12)
    rows = [cd.pearson(probe[k], human["h_resid"]) for k in range(5)]
    assert np.allclose(pearson_rows_masked(probe, human["h_resid"]), rows, atol=1e-12)
    out = {"defect": "permutation null and exemplar CI of r_h computed over NaN images (6 of 1,200 with no resolved observer trial)",
           "record": RECORD, "code": open(os.path.join(CODE, "COMMIT")).read().strip(), "fits_dir": FITS,
           "n_images": int(len(finite_h)), "n_images_finite_h": int(finite_h.sum()), "n_perm": n_perm,
           "n_boot_exemplar": cd.N_BOOT_EXEMPLAR, "ceiling_h": human["ceiling_h"],
           "per_arm_roi": {}, "per_subject": {}, "P-C2_permutation_criterion": {}}
    t0 = time.time()
    max_dr = 0.0
    for arm, runs in rec["settings"]["arms"].items():
        for s in cd.SUBJECTS:
            fits = [load_joint(os.path.join(FITS, f"{run}_subj{s:02d}.npz")) for run in runs]
            for roi in rec["settings"]["rois"]:
                vox = [cd.roi_voxels(f["z"], roi, rec["settings"]["reliable_voxels_only"]) for f in fits]
                if min(len(v) for v in vox) < cd.MIN_VOX:
                    continue
                m = np.mean([cd.cell_margins(f, v, "joint", classes, rec["settings"]["method"])["m"] for f, v in zip(fits, vox)], axis=0)
                c = np.mean([cd.cell_margins(f, v, "joint", classes, rec["settings"]["method"])["c"] for f, v in zip(fits, vox)], axis=0)
                m_res = cd.residualise(m, human["shapes"], human["textures"])
                c_res = cd.residualise(c, human["shapes"], human["textures"])
                r_h = cd.pearson(m_res, human["h_resid"])
                r_rec = rec["per_subject"][arm][str(s)][roi]["joint"]["r_h_resid"]
                max_dr = max(max_dr, abs(r_h - r_rec))
                if abs(r_h - r_rec) > 1e-9:
                    raise RuntimeError(f"{arm} subj{s} {roi}: recomputed r_h {r_h:.6f} != record {r_rec:.6f}")
                cell = out["per_subject"].setdefault(arm, {}).setdefault(str(s), {})
                cell[roi] = {"r_h_resid": r_h, "pct_h": 100 * r_h / human["ceiling_h"],
                             "r_h_resid_ci_exemplar": cluster_bootstrap_r_masked(m_res, human["h_resid"], clusters, cd.N_BOOT_EXEMPLAR),
                             "_rh_null": pearson_rows_masked(h_perm, m_res), "_rv_null": pearson_rows_masked(v_perm, c_res),
                             "r_v_resid": cd.pearson(c_res, human["v_resid"])}
            print(f"[hfix] {arm} subj{s:02d} done (+{time.time() - t0:.0f}s)", flush=True)
    for arm in out["per_subject"]:
        for roi in rec["settings"]["rois"]:
            subs = [out["per_subject"][arm][s][roi] for s in out["per_subject"][arm] if roi in out["per_subject"][arm][s]]
            if not subs:
                continue
            rh = np.mean([d["_rh_null"] for d in subs], axis=0)
            rv = np.mean([d["_rv_null"] for d in subs], axis=0)
            obs_h = float(np.mean([d["r_h_resid"] for d in subs]))
            obs_v = float(np.mean([d["r_v_resid"] for d in subs]))
            p_v = cd.perm_p(obs_v, rv)
            p_v_rec = rec["permutation"][arm][roi]["p_v"]
            if abs(p_v - p_v_rec) > 1e-12 or abs(obs_h - rec["permutation"][arm][roi]["r_h_mean"]) > 1e-9:
                raise RuntimeError(f"{arm} {roi}: v-side check failed ({p_v} vs {p_v_rec})")
            null975 = float(np.percentile(np.abs(rh), 97.5))
            out["per_arm_roi"].setdefault(arm, {})[roi] = {
                "n_subjects": len(subs), "r_h_mean": obs_h, "pct_h_mean": 100 * obs_h / human["ceiling_h"],
                "p_h": cd.perm_p(obs_h, rh), "null_h_97.5": null975, "pct_null_h_97.5": 100 * null975 / human["ceiling_h"],
                "null_h_nan": int(np.isnan(rh).sum()), "p_v_check": p_v, "p_v_record": p_v_rec,
                "p_h_record_vacuous": rec["permutation"][arm][roi]["p_h"]}
    for arm in out["per_subject"]:
        for s in out["per_subject"][arm]:
            for roi in out["per_subject"][arm][s]:
                out["per_subject"][arm][s][roi].pop("_rh_null")
                out["per_subject"][arm][s][roi].pop("_rv_null")
    crit = {roi: out["per_arm_roi"]["model"][roi]["p_h"] for roi in EXTRASTRIATE}
    out["P-C2_permutation_criterion"] = {"model_p_h": crit, "met_p_lt_0.01_in_all_three": all(p < 0.01 for p in crit.values()),
                                         "note": "P-C2 also requires r_h above the input floor in 8/8 subjects per region, "
                                                 "which the record already fails (hV4 7/8, ventral 0/8, lateral 1/8); "
                                                 "the reading of record is set by P-C1 and is unchanged"}
    out["max_abs_diff_r_h_vs_record"] = max_dr
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1, default=cd._json_default)
    print(f"[hfix] model p_h: " + ", ".join(f"{r}={v['p_h']:.4f} (null97.5 {v['pct_null_h_97.5']:.1f}%)" for r, v in out["per_arm_roi"]["model"].items()))
    print(f"[hfix] gabor p_h: " + ", ".join(f"{r}={v['p_h']:.4f}" for r, v in out["per_arm_roi"]["gabor"].items()))
    print(f"[hfix] saved {OUT} (+{time.time() - t0:.0f}s; max |r_h - record| = {max_dr:.2e})", flush=True)


if __name__ == "__main__":
    main()
