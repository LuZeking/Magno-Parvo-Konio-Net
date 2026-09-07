"""Verify the peer's contrast reading, and check WHICH construction reproduces.

Claims under test (6f336f5 / their message):
  (i)   §2.2's statistic is paired(spi[:,hV4], mean(V1v,V1d)) across subjects
  (ii)  ratio contrast    diet -0.834 / no-diet -0.079  vs published -0.862/-0.101
  (iii) difference contrast diet -0.166 / no-diet -0.027 vs published -0.176/-0.034
  (iv)  floor in contrast units: ratio +0.0511 (8/8, p=.0078), diff +0.00090 (4/8)
  (v)   condition A est/true: ratio 1.255 (inflates), difference 0.713 (compresses)

(ii) and (iii) are described as reproductions but are 3% and 6% off. So also test
the two constructions that could explain the residual:
  MEDIAN-OF-RATIOS   median_v[(um-up)/(um+up)]    <- what visualrois.py does
  RATIO-OF-MEDIANS   (median_v um - median_v up)/(median_v um + median_v up)
                     <- what the recovery JSONs can express, since they store
                        only per-ROI medians
"""
import glob
import json
import numpy as np
from scipy import stats as sps

R = "/home/byrong/projects/TSN/Magno-Parvo-Konio-Net/.claude/worktrees/mpk-dev"
NC_MIN = 0.1
MIN_VOX = 20
# prf-visualrois codes, hierarchical order; index 6 is hV4
ORDER = [(1, "V1v"), (2, "V1d"), (3, "V2v"), (4, "V2d"),
         (5, "V3v"), (6, "V3d"), (7, "hV4")]


def real_profiles(model):
    """Per-subject x ROI, in both constructions, exactly as visualrois filters."""
    mor, rom, dif = [], [], []
    for f in sorted(glob.glob(f"{R}/results/encoding/{model}_subj*.npz")):
        z = np.load(f, allow_pickle=True)
        roi, nc = z["roi_prf-visualrois"], z["noise_ceiling"]
        um, up = z["unique_magno"], z["unique_parvo"]
        a, b, c = [], [], []
        for code, _ in ORDER:
            v = (roi == code) & (nc > NC_MIN) & ((um + up) > 1e-3)
            if v.sum() >= MIN_VOX:
                r = (um[v] - up[v]) / (um[v] + up[v])
                a.append(np.median(r[np.isfinite(r)]))
                mm, mp = np.median(um[v]), np.median(up[v])
                b.append((mm - mp) / (mm + mp))
                c.append(mm - mp)
            else:
                a.append(np.nan); b.append(np.nan); c.append(np.nan)
        mor.append(a); rom.append(b); dif.append(c)
    return map(np.array, (mor, rom, dif))


def contrast(P):
    """hV4 - mean(V1v, V1d), per subject."""
    return P[:, 6] - np.nanmean(P[:, [0, 1]], axis=1)


def report(name, d, target=None):
    ok = np.isfinite(d)
    d = d[ok]
    W, p = sps.wilcoxon(d)
    t = f"   published {target:+.3f}  gap {d.mean() - target:+.4f}" if target \
        is not None else ""
    print(f"  {name:<34} {d.mean():+.4f}  {int((d > 0).sum())}/{ok.sum()} pos  "
          f"p={p:.4f}{t}")


print("=" * 74)
print("(i-iii) §2.2 REPRODUCTION -- which construction gives the published number?")
print("=" * 74)
for model, tag, pub_r, pub_d in (
        ("dkl_bcs_seed0_trained", "diet", -0.862, -0.176),
        ("dkl_base_seed0_trained", "no diet", -0.101, -0.034)):
    mor, rom, dif = real_profiles(model)
    print(f"\n{tag}  (n={mor.shape[0]} subjects)")
    report("ratio, MEDIAN-OF-RATIOS", contrast(mor), pub_r)
    report("ratio, RATIO-OF-MEDIANS", contrast(rom), pub_r)
    report("difference of medians", contrast(dif), pub_d)

print("\n" + "=" * 74)
print("(iv) FLOOR in §2.2's contrast statistic, from condition B")
print("=" * 74)
D = f"{R}/results/vp_recovery_realnc"
ds = [json.load(open(f)) for f in sorted(glob.glob(f"{D}/*.json"))]
print(f"arm(s): {sorted({d['model'] for d in ds})}, n={len(ds)}")
ratio, diff = [], []
for d in ds:
    rs = d["conditions"]["B_manufacture"]["rows"]
    um = np.array([r["est_unique_m"] for r in rs])
    up = np.array([r["est_unique_p"] for r in rs])
    ratio.append((um - up) / (um + up))
    diff.append(um - up)
ratio, diff = np.array(ratio), np.array(diff)
report("ratio index contrast", contrast(ratio))
report("difference contrast", contrast(diff))
print(f"  ratio contrast range {contrast(ratio).min():+.4f}.."
      f"{contrast(ratio).max():+.4f}")
print("\n  NOTE: the JSONs store only per-ROI MEDIANS, so this ratio index is")
print("  RATIO-OF-MEDIANS. §2.2 uses MEDIAN-OF-RATIOS. Same-name, different")
print("  estimator -- see the gap measured above.")

print("\n" + "=" * 74)
print("(v) CONDITION A magnitude fidelity, per statistic")
print("=" * 74)
for stat in ("ratio", "diff"):
    T, E = [], []
    for d in ds:
        rs = d["conditions"]["A_recovery"]["rows"]
        for key, acc in (("true", T), ("est", E)):
            um = np.array([r[f"{key}_unique_m"] for r in rs])
            up = np.array([r[f"{key}_unique_p"] for r in rs])
            acc.append((um - up) / (um + up) if stat == "ratio" else um - up)
    T, E = np.array(T).ravel(), np.array(E).ravel()
    slope = np.polyfit(T, E, 1)[0]
    print(f"  {stat:<6} est = {slope:.3f} x true   -> "
          f"{'INFLATES' if slope > 1 else 'compresses'} by "
          f"{max(slope, 1 / slope):.3f}x   r={np.corrcoef(T, E)[0, 1]:+.3f}")
print("\n  If the two statistics are biased in OPPOSITE directions there is no")
print("  bounding direction, and 'the reported value is a floor' cannot stand.")
