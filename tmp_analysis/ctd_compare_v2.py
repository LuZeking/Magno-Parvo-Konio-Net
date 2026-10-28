"""Compare the v2 cortex-to-decision analysis (corrected module, slurm/ctd_analysis_v2.sh)
with the v1 record and the corrected shape-target null (ledger §3–§4).

Rule: every leaf of the v1 record must be reproduced by v2 (floats to 1e-12, NaN
= NaN), except the quantities the NaN defect touched, which v2 recomputes with the
images masked: the shape-target permutation p and null percentile (per arm x ROI,
also P-C2's `perm_p` under `h`) and the r_h exemplar-cluster CIs. In the main run
those must equal results/cortex_decision/ctd_h_null_fix.json (joint band; the
magno/parvo bands' CIs, which the fix did not recompute, must be finite); in the
sensitivity runs they must be finite where v1 had NaN.

Usage: python tmp_analysis/ctd_compare_v2.py [V2_DIR] -> results/cortex_decision/v2/compare_v2.json
"""
from __future__ import annotations

import json
import math
import os
import sys

V1 = "results/cortex_decision"
V2 = sys.argv[1] if len(sys.argv) > 1 else "/scratch/byrong/mpk/encoding/results/2026-10-26_ctd/analysis_v2"
FIX = os.path.join(V1, "ctd_h_null_fix.json")
RUNS = ["", "sens_spearman", "sens_v1_streams", "sens_all_voxels"]
TOL = 1e-12


def leaves(x, path=()):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from leaves(v, path + (str(k),))
    elif isinstance(x, list) and x and not isinstance(x[0], (dict, list)):
        yield path, x
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from leaves(v, path + (str(i),))
    else:
        yield path, x


def same(a, b):
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(u, v) for u, v in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        try:
            a, b = float(a), float(b)
        except (TypeError, ValueError):
            return a == b
        return (math.isnan(a) and math.isnan(b)) or abs(a - b) <= TOL
    return a == b


def kind(path):
    """Which corrected quantity a leaf is, or None if v2 must reproduce v1."""
    if len(path) >= 4 and path[0] == "permutation" and path[-1] in ("p_h", "null_h_97.5"):
        return ("perm", path[1], path[2], path[-1])
    if path[0] == "predictions" and path[1] == "P-C2" and len(path) == 6 and path[2] == "rois" and path[4] == "h" and path[-1] == "perm_p":
        return ("perm", "model", path[3], "p_h")           # P-C2 is scored on the model arm
    if path[0] == "per_subject" and path[-1] == "r_h_resid_ci_exemplar":
        return ("ci", path[1], path[2], path[3], path[4])      # arm, subject, roi, band
    return None


def main() -> int:
    fix = json.load(open(FIX))
    out = {"v2_dir": V2, "runs": {}}
    bad_total = 0
    for run in RUNS:
        a = json.load(open(os.path.join(V1, run, "ctd_analysis.json")))
        b = json.load(open(os.path.join(V2, run, "ctd_analysis.json")))
        A = dict(leaves(a))
        B = dict(leaves(b))
        n_eq = n_fix = n_fin = 0
        bad = []
        for path, va in A.items():
            if path[0] == "settings" and path[1] == "fits_dir":
                continue
            if path not in B:
                bad.append(("missing in v2", "/".join(path)))
                continue
            vb = B[path]
            k = kind(path)
            if k is None:
                if same(va, vb):
                    n_eq += 1
                else:
                    bad.append(("differs", "/".join(path), va, vb))
            elif k[0] == "perm" and run == "":
                ref = fix["per_arm_roi"][k[1]][k[2]][k[3]]
                if same(vb, ref):
                    n_fix += 1
                else:
                    bad.append(("perm != fix", "/".join(path), vb, ref))
            elif k[0] == "ci" and run == "" and k[4] == "joint":
                ref = fix["per_subject"][k[1]][k[2]][k[3]]["r_h_resid_ci_exemplar"]
                if same(vb, ref):
                    n_fix += 1
                else:
                    bad.append(("ci != fix", "/".join(path), vb, ref))
            else:
                vals = vb if isinstance(vb, list) else [vb]
                if all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals) and (not isinstance(vb, list) or vb[0] < vb[1]):
                    n_fin += 1
                else:
                    bad.append(("not finite", "/".join(path), vb))
        extra = [p for p in B if p not in A]
        rec = {"n_reproduced": n_eq, "n_equal_to_fix": n_fix, "n_finite_corrected": n_fin, "n_bad": len(bad),
               "n_extra_in_v2": len(extra), "reading_v1": a["reading_of_record"], "reading_v2": b["reading_of_record"],
               "met_v1": {k: v.get("met") for k, v in a["predictions"].items()},
               "met_v2": {k: v.get("met") for k, v in b["predictions"].items()},
               "bad": [list(map(str, x)) for x in bad[:20]]}
        out["runs"][run or "main"] = rec
        bad_total += len(bad) + len(extra)
        print(f"[{run or 'main'}] reproduced {n_eq}, equal to fix {n_fix}, finite corrected {n_fin}, bad {len(bad)}, extra {len(extra)}; "
              f"reading v1={a['reading_of_record']} v2={b['reading_of_record']}")
        for x in bad[:10]:
            print("   ", x)
    out["ok"] = bad_total == 0
    os.makedirs(os.path.join(V1, "v2"), exist_ok=True)
    with open(os.path.join(V1, "v2", "compare_v2.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("OK" if out["ok"] else "BAD", "->", os.path.join(V1, "v2", "compare_v2.json"))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
