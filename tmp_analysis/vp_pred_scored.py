"""Does the real descending nc profile reverse condition B's artefact?

My §84.3 prediction, recorded BEFORE this run existed:
  SIGN      positive (artefact rises with hierarchical rank)
  MAGNITUDE ~16% variation in normalised bias; contrast order +0.05,
            against the -0.351 my backwards ascending ramp produced.
This script is the falsification test, not a confirmation exercise.
"""
import glob
import json
import numpy as np
from scipy import stats as sps

D = ("/home/byrong/projects/TSN/Magno-Parvo-Konio-Net/.claude/worktrees/"
     "mpk-dev/results/vp_recovery_realnc")
ROI = ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4"]

by_roi, models = {}, set()
for f in sorted(glob.glob(f"{D}/*.json")):
    d = json.load(open(f))
    models.add(d["model"])
    for r in d["conditions"]["B_manufacture"]["rows"]:
        by_roi.setdefault(r["roi_index"], []).append(r)
print(f"arms present: {sorted(models)}   n_subj/ROI: "
      f"{len(by_roi[0])}\n")

# every true_diff must be exactly zero, else this is not a null
alltrue = [r["true_diff"] for rs in by_roi.values() for r in rs]
assert max(abs(t) for t in alltrue) < 1e-9, "condition B is not a zero null"
print("null verified: max |true_diff| = "
      f"{max(abs(t) for t in alltrue):.2e}\n")

print(f"{'ROI':5s} {'nc':>6s} {'norm bias':>10s} {'sd':>7s} {'raw=e*nc':>9s}")
nc, norm, raw = [], [], []
for i in sorted(by_roi):
    rs = by_roi[i]
    c = rs[0]["noise_ceiling"]
    assert all(abs(r["noise_ceiling"] - c) < 1e-12 for r in rs)
    e = np.array([r["est_diff"] for r in rs])
    nc.append(c); norm.append(e.mean()); raw.append((e * c).mean())
    print(f"{ROI[i]:5s} {c:6.3f} {e.mean():10.4f} {e.std(ddof=1):7.4f} "
          f"{(e * c).mean():9.4f}")

nc, norm, raw = map(np.array, (nc, norm, raw))
rank = np.arange(len(nc))
rho_nc = sps.spearmanr(rank, nc)
rho_no = sps.spearmanr(rank, norm)
rho_rw = sps.spearmanr(rank, raw)
print(f"\nnc      : rho vs rank {rho_nc.statistic:+.3f} (p={rho_nc.pvalue:.4f})"
      f"  range {nc.min():.3f}-{nc.max():.3f}  factor {nc.max()/nc.min():.2f}")
print(f"norm    : rho vs rank {rho_no.statistic:+.3f} (p={rho_no.pvalue:.4f})"
      f"  spread {norm.max() - norm.min():.4f}")
print(f"raw     : rho vs rank {rho_rw.statistic:+.3f} (p={rho_rw.pvalue:.4f})"
      f"  spread {raw.max() - raw.min():.4f}")
print(f"corr(norm, 1/nc) = {np.corrcoef(norm, 1.0 / nc)[0, 1]:+.4f}")

contrast = norm[-1] - norm[0]          # hV4 - V1v, as §2.2 reads it
pct = (norm.max() - norm.min()) / abs(norm.mean()) * 100
print("\n--- prediction scored ---")
print(f"SIGN      predicted POSITIVE   observed {contrast:+.4f} "
      f"-> {'PASS' if contrast > 0 else 'FAIL'}")
print(f"MAGNITUDE predicted ~+0.05 and ~16% variation")
print(f"          observed contrast {contrast:+.4f}, variation {pct:.1f}%")
print(f"          |contrast| < 0.15 -> "
      f"{'PASS' if abs(contrast) < 0.15 else 'FAIL (too large)'}")
print(f"\nartefact floor (mean normalised bias) = {norm.mean():+.4f} "
      f"toward {'magno' if norm.mean() > 0 else 'parvo'}")
