"""Does a per-subject floor give 5/8, p=.3125? Closes the last unplaced number.

6684a41 identifies the variant §96.3 could not place as the group-versus-
per-subject FLOOR pairing (same point estimate, different pairing). Checked
here across both floor choices and both residual-removal orders, so the set is
closed rather than assumed closed.
"""
import glob
import json
import numpy as np
from scipy import stats as sps

R = "/home/byrong/projects/TSN/Magno-Parvo-Konio-Net/.claude/worktrees/mpk-dev"
ARMS = {"no-diet": "dkl_base_seed0_trained", "diet": "dkl_bcs_seed0_trained"}


def ratio(m, p):
    return (m - p) / (m + p)


def con(v):
    return v[6] - v[[0, 1]].mean()


for arm, model in ARMS.items():
    ds = [json.load(open(f)) for f in sorted(glob.glob(
        f"{R}/results/vp_recovery_realnc/vp_recovery_{model}_subj*.json"))]
    Bs = np.array([[ratio(r["est_unique_m"], r["est_unique_p"])
                    for r in d["conditions"]["B_manufacture"]["rows"]]
                   for d in ds])
    floor_group = con(Bs.mean(0))
    floor_subj = np.array([con(Bs[i]) for i in range(len(ds))])
    T = np.array([[ratio(r["true_unique_m"], r["true_unique_p"])
                   for r in d["conditions"]["A_recovery"]["rows"]] for d in ds])
    E = np.array([[ratio(r["est_unique_m"], r["est_unique_p"])
                   for r in d["conditions"]["A_recovery"]["rows"]] for d in ds])
    Y = E - Bs.mean(0)
    a = float(np.sum(T * Y) / np.sum(T * T))
    rc = np.mean([con((Y - a * T)[i]) for i in range(len(ds))])

    real = []
    for f in sorted(glob.glob(f"{R}/results/encoding_cc/{model}_subj*.npz")):
        z = np.load(f, allow_pickle=True)
        roi, nc = z["roi_prf-visualrois"], z["noise_ceiling"]
        um, up = z["unique_magno"], z["unique_parvo"]
        p7 = []
        for c in range(1, 8):
            v = (roi == c) & (nc > 0.1) & ((um + up) > 1e-3)
            rr = (um[v] - up[v]) / (um[v] + up[v]) if v.sum() >= 20 \
                else [np.nan]
            p7.append(np.nanmedian(rr))
        real.append(con(np.array(p7)))
    real = np.array(real)

    print(f"\n{arm}   (a = {a:.3f}, group floor {floor_group:+.4f}, "
          f"residual contrast {rc:+.4f})")
    for lab, fl in (("group floor", floor_group),
                    ("per-subject floor", floor_subj)):
        v = (real - fl) / a
        w = sps.wilcoxon(v)
        print(f"  {lab:18s} {'MINIMAL (floor only)':24s} {v.mean():+.4f}  "
              f"{int((v < 0).sum())}/{len(v)} neg  p={w.pvalue:.4f}")
        for lab2, vv in (("+ resid before scaling", (real - fl - rc) / a),
                         ("+ resid after scaling", (real - fl) / a - rc)):
            w = sps.wilcoxon(vv)
            print(f"  {lab:18s} {lab2:24s} {vv.mean():+.4f}  "
                  f"{int((vv < 0).sum())}/{len(vv)} neg  p={w.pvalue:.4f}")

print("\n6684a41 reports: per-subject floor -> 5/8, p=.3125;")
print("group floor -> 6/8, p=.0547; after-scaling -> 5/8, p=.8438")
