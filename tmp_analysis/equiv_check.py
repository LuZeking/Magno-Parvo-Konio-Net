"""W2a acceptance check: does the memory-bounded encoding path reproduce a stored fit?

  python tmp_analysis/equiv_check.py --new NEW.npz --ref results/encoding/<model>_subjNN.npz

Rule (docs/PLAN_NMI9_2026-09-15.md, W2a): voxelwise unique variances of the two
bands correlate with the stored fit at r >= 0.99, and the per-ROI index (median
over reliable voxels of the noise-ceiling-normalised U_m - U_p, prf-visualrois)
differs by < 0.02 in every ROI. Exit status 0 on PASS, 1 on FAIL.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

ROIS = {1: "V1v", 2: "V1d", 3: "V2v", 4: "V2d", 5: "V3v", 6: "V3d", 7: "hV4"}


def get(z, *names):
    for n in names:
        if n in z.files:
            return z[n]
    raise KeyError(names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--r-min", type=float, default=0.99)
    ap.add_argument("--roi-tol", type=float, default=0.02)
    a = ap.parse_args()
    new, ref = np.load(a.new, allow_pickle=True), np.load(a.ref, allow_pickle=True)
    ok = True
    print(f"{'quantity':26s} {'r':>8s} {'max|diff|':>10s} {'median|diff|':>13s}")
    for key, names in [("r2_magno", ("r2_magno",)), ("r2_parvo", ("r2_parvo",)), ("r2_joint", ("r2_joint",)),
                       ("unique_magno (pre-nc)", ("unique_magno_pre_nc", "unique_magno_raw")),
                       ("unique_parvo (pre-nc)", ("unique_parvo_pre_nc", "unique_parvo_raw"))]:
        x, y = get(ref, *names).astype(float), get(new, *names).astype(float)
        m = np.isfinite(x) & np.isfinite(y)
        r = float(np.corrcoef(x[m], y[m])[0, 1])
        d = np.abs(x[m] - y[m])
        flag = "" if r >= a.r_min else "   <-- below r_min"
        if key.startswith("unique") and r < a.r_min:
            ok = False
        print(f"{key:26s} {r:8.5f} {d.max():10.2e} {np.median(d):13.2e}{flag}")

    nc = get(ref, "noise_ceiling").astype(float)
    roi = get(ref, "roi_prf-visualrois").astype(int)
    um_r, up_r = get(ref, "unique_magno").astype(float), get(ref, "unique_parvo").astype(float)
    um_n, up_n = get(new, "unique_magno").astype(float), get(new, "unique_parvo").astype(float)
    print(f"\n{'ROI':6s} {'n vox':>6s} {'index ref':>10s} {'index new':>10s} {'diff':>8s}")
    worst = 0.0
    for lab, name in ROIS.items():
        v = (roi == lab) & (nc > 0.1) & np.isfinite(um_r) & np.isfinite(up_r) & np.isfinite(um_n) & np.isfinite(up_n)
        if v.sum() < 20:
            print(f"{name:6s} {int(v.sum()):6d}   (fewer than 20 voxels, skipped)")
            continue
        ir = float(np.median(um_r[v] - up_r[v]))
        inew = float(np.median(um_n[v] - up_n[v]))
        worst = max(worst, abs(ir - inew))
        print(f"{name:6s} {int(v.sum()):6d} {ir:10.4f} {inew:10.4f} {ir - inew:8.4f}")
    if worst >= a.roi_tol:
        ok = False
    print(f"\nlargest per-ROI index difference: {worst:.4f} (tolerance {a.roi_tol})")
    print("EQUIVALENCE PASS" if ok else "EQUIVALENCE FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
