"""Per-image shape margins of one region for Fig. 6e — the one quantity the
cortex-to-decision analysis (src/analysis/encoding/cortex_decision.py) does
not store in ctd_analysis.json. Recomputes the joint-band margins of every
subject (seeds averaged, as the analysis does), exactly as the analysis does
(its own functions, the pinned code), and writes the residualised margins
with the per-subject r_h (which must equal the json's) and the exemplar-
cluster bootstrap interval of the subject-mean margin's r.

Usage (from the worktree):
  python tmp_analysis/ctd_perimage_margins.py [ROI=hV4] [arms=model,gabor]
Writes results/cortex_decision/ctd_perimage_<ROI>.json.
Code: CTD_CODE (default: the pinned snapshot 376fb05 the analysis ran from).
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

CODE = os.environ.get("CTD_CODE", "/home/byrong/projects/TSN/fusion_stage1_code/376fb05")
sys.path.insert(0, CODE)
from src.analysis.encoding import cortex_decision as cd  # noqa: E402

FITS = os.environ.get("CTD_FITS", "/scratch/byrong/mpk/encoding/results/2026-10-26_ctd")
HUMAN = os.environ.get("CTD_HUMAN", "/scratch/byrong/mpk/perimage_behavior/results/2026-09-07_geirhos_perimage")
BAND = "joint"


def load_joint(path: str, repro_tol: float = 1e-3) -> dict:
    """load_fit restricted to the joint band (the other bands' 130 MB per file are not needed)."""
    z = np.load(path, allow_pickle=False)
    d = float(z["repro_max_abs_diff"]) if "repro_max_abs_diff" in z.files else float("nan")
    if d > repro_tol:
        raise RuntimeError(f"{path}: repro_max_abs_diff {d:.3e} > {repro_tol:g}")
    fit = {"path": path, "subject": int(z["subject"]), "repro_max_abs_diff": d,
           "stems": np.asarray([str(s) for s in z["cc_stems"]]),
           "shapes": np.asarray([str(s) for s in z["cc_shapes"]]),
           "textures": np.asarray([str(s) for s in z["cc_textures"]]),
           "tpl_labels": np.asarray([str(s) for s in z["tpl_labels"]]),
           "tpl_source": np.asarray([str(s) for s in z["tpl_source"]]),
           f"cc_{BAND}": np.asarray(z[f"cc_pred_{BAND}"], np.float32),
           f"tpl_{BAND}": np.asarray(z[f"tpl_pred_{BAND}"], np.float32),
           "z": z}
    return fit


def main() -> None:
    roi = sys.argv[1] if len(sys.argv) > 1 else "hV4"
    arms = (sys.argv[2] if len(sys.argv) > 2 else "model,gabor").split(",")
    out_path = f"results/cortex_decision/ctd_perimage_{roi}.json"
    human = cd.load_human(HUMAN)
    classes = sorted(set(human["shapes"]) | set(human["textures"]))
    clusters = cd.object_exemplar(human["stems"])
    out = {"roi": roi, "band": BAND, "code": open(os.path.join(CODE, "COMMIT")).read().strip(),
           "fits_dir": FITS, "stems": human["stems"].tolist(), "shapes": human["shapes"].tolist(),
           "textures": human["textures"].tolist(), "h": human["h"].tolist(), "h_resid": human["h_resid"].tolist(),
           "ceiling_h": human["ceiling_h"], "arms": {}}
    t0 = time.time()
    for arm in arms:
        per = {}
        for s in cd.SUBJECTS:
            ms = []
            for run in cd.ARMS[arm]:
                fit = load_joint(os.path.join(FITS, f"{run}_subj{s:02d}.npz"))
                if not np.array_equal(fit["stems"], human["stems"]):
                    raise RuntimeError(f"{fit['path']}: stems differ from the observers' listing")
                vox = cd.roi_voxels(fit["z"], roi, True)
                ms.append(cd.cell_margins(fit, vox, BAND, classes, "pearson")["m"])
                print(f"[perimage] {arm} {run} subj{s:02d} {roi}: {len(vox)} voxels, +{time.time() - t0:.0f}s", flush=True)
            m = np.mean(ms, axis=0)
            m_res = cd.residualise(m, human["shapes"], human["textures"])
            per[str(s)] = {"n_seeds": len(ms), "m": m.tolist(), "m_resid": m_res.tolist(),
                           "B": float(np.mean(m > 0)), "r_h_resid": cd.pearson(m_res, human["h_resid"])}
        M = np.mean([np.asarray(per[s]["m_resid"]) for s in per], axis=0)
        r = cd.pearson(M, human["h_resid"])
        keep = np.isfinite(human["h_resid"])            # 6 images no observer resolved (ledger §3 defect)
        ci = cd.cluster_bootstrap_r(M[keep], human["h_resid"][keep], clusters[keep], cd.N_BOOT_EXEMPLAR)
        out["arms"][arm] = {"per_subject": per, "subject_mean_m_resid": M.tolist(),
                            "r_h_resid_subject_mean": r, "pct_h_subject_mean": 100 * r / human["ceiling_h"],
                            "r_h_resid_subject_mean_ci_exemplar": ci}
        print(f"[perimage] {arm} {roi}: subject-mean margin r_h = {r:+.4f} ({100 * r / human['ceiling_h']:+.1f}% of ceiling), "
              f"exemplar CI {ci}", flush=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(out, fh, default=cd._json_default)
    print(f"[perimage] saved {out_path} (+{time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
