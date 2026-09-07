"""Where does the residual come from, and why do our significance figures differ?

Two loose ends from a9ba8d7:
 (1) they report the fullest treatment for no-diet as 5/8 negative, p = .3125;
     I get 6/8, p = .0547 -- which is the difference between "does not hold"
     and "marginal", so it matters;
 (2) the residual is not absorbed by a quadratic in truth (57% of variance but
     0% of the CONTRAST), so it is not simple nonlinearity, and b(ROI) already
     removed the ROI-additive part. What is it?

Condition A generates its sources as a = 0.30 + 0.10i, b = 0.90 - 0.10i, so the
source MAGNITUDES follow a fixed schedule in ROI index while the ratio index
`true` is a monotone function of i. Total source power a^2+b^2 is U-shaped in i.
If the residual tracks that rather than the ratio, it is a property of condition
A's construction and does not transfer to real data with different magnitudes.
"""
import glob
import json
import numpy as np
from scipy import stats as sps

R = "/home/byrong/projects/TSN/Magno-Parvo-Konio-Net/.claude/worktrees/mpk-dev"
ARMS = {"no-diet": "dkl_base_seed0_trained", "diet": "dkl_bcs_seed0_trained"}
REAL = {"no-diet": -0.0785, "diet": -0.8335}


def load(m):
    return [json.load(open(f)) for f in sorted(glob.glob(
        f"{R}/results/vp_recovery_realnc/vp_recovery_{m}_subj*.json"))]


def ratio(m, p):
    return (m - p) / (m + p)


def con(v):
    return v[6] - v[[0, 1]].mean()


i = np.arange(7)
a_sched, b_sched = 0.30 + 0.10 * i, 0.90 - 0.10 * i
power = a_sched ** 2 + b_sched ** 2
print(f"condition A source schedule: a {np.round(a_sched,2).tolist()}")
print(f"                             b {np.round(b_sched,2).tolist()}")
print(f"total power a^2+b^2          {np.round(power,3).tolist()}  (U-shaped)")

prof = {}
for arm, model in ARMS.items():
    ds = load(model)
    b_roi = np.array([[ratio(r["est_unique_m"], r["est_unique_p"])
                       for r in d["conditions"]["B_manufacture"]["rows"]]
                      for d in ds]).mean(0)
    T = np.array([[ratio(r["true_unique_m"], r["true_unique_p"])
                   for r in d["conditions"]["A_recovery"]["rows"]] for d in ds])
    E = np.array([[ratio(r["est_unique_m"], r["est_unique_p"])
                   for r in d["conditions"]["A_recovery"]["rows"]] for d in ds])
    Y = E - b_roi
    a_lin = float(np.sum(T * Y) / np.sum(T * T))
    res = (Y - a_lin * T)
    prof[arm] = res.mean(0)
    print(f"\n{arm}: residual profile {np.round(prof[arm], 3).tolist()}")
    print(f"  corr(residual, total power) = "
          f"{np.corrcoef(prof[arm], power)[0,1]:+.3f}")
    print(f"  corr(residual, truth)       = "
          f"{np.corrcoef(prof[arm], T.mean(0))[0,1]:+.3f}")

r_arms = np.corrcoef(prof["no-diet"], prof["diet"])[0, 1]
print(f"\ncorr(no-diet residual profile, diet residual profile) = "
      f"{r_arms:+.3f}")
print("Effects differ ~10x between arms; if the residual profile is near-")
print("identical, it is a property of condition A, not of either model.")

# (1) resolve the significance discrepancy: per-subject vs group residual
print("\n" + "=" * 68)
print("SIGNIFICANCE, three ways of removing the residual contrast")
print("=" * 68)
for arm, model in ARMS.items():
    ds = load(model)
    b_roi = np.array([[ratio(r["est_unique_m"], r["est_unique_p"])
                       for r in d["conditions"]["B_manufacture"]["rows"]]
                      for d in ds]).mean(0)
    floor = con(b_roi)
    T = np.array([[ratio(r["true_unique_m"], r["true_unique_p"])
                   for r in d["conditions"]["A_recovery"]["rows"]] for d in ds])
    E = np.array([[ratio(r["est_unique_m"], r["est_unique_p"])
                   for r in d["conditions"]["A_recovery"]["rows"]] for d in ds])
    Y = E - b_roi
    a_lin = float(np.sum(T * Y) / np.sum(T * T))
    res = Y - a_lin * T
    rc_group = np.mean([con(res[k]) for k in range(len(ds))])
    rc_subj = np.array([con(res[k]) for k in range(len(ds))])

    real = []
    for f in sorted(glob.glob(f"{R}/results/encoding_cc/{model}_subj*.npz")):
        z = np.load(f, allow_pickle=True)
        roi, nc = z["roi_prf-visualrois"], z["noise_ceiling"]
        um, up = z["unique_magno"], z["unique_parvo"]
        p7 = []
        for code in range(1, 8):
            v = (roi == code) & (nc > 0.1) & ((um + up) > 1e-3)
            rr = (um[v] - up[v]) / (um[v] + up[v]) if v.sum() >= 20 else [np.nan]
            p7.append(np.nanmedian(rr))
        real.append(con(np.array(p7)))
    real = np.array(real)

    print(f"\n{arm}  (floor {floor:+.4f}, a {a_lin:.3f}, "
          f"group residual contrast {rc_group:+.4f})")
    for label, v in (
            ("group residual", (real - floor - rc_group) / a_lin),
            ("per-subject residual", (real - floor - rc_subj) / a_lin),
            ("residual removed pre-scaling", (real - floor) / a_lin - rc_group),
    ):
        w = sps.wilcoxon(v)
        print(f"  {label:30s} {v.mean():+.4f}  "
              f"{int((v < 0).sum())}/{len(v)} neg  p={w.pvalue:.4f}")
