"""Is the condition-A residual an ROI offset (transferable) or nonlinearity (not)?

a9ba8d7 subtracts a residual CONTRAST as well as the condition-B floor, which
takes the no-diet arm to -0.0337, 5/8, p=.31 and the factor to 18.6x. They flag
its legitimacy as open. There is a sharper argument available.

KEY STRUCTURE. b(ROI) from condition B is the ROI-additive term measured at
truth = 0 (truth is zero at every ROI there, so ROI varies and truth does not).
So after subtracting b(ROI) from condition A, any residual that remains CANNOT
be an ROI-additive offset -- that term has already been removed. What is left
must depend on truth. And in condition A truth is a deterministic function of
ROI index (a = 0.30 + 0.10i, b = 0.90 - 0.10i), so ROI and truth are perfectly
confounded and the two explanations are not separable by ROI structure alone.

TEST: if the residual is a nonlinear response to truth, a quadratic term in
truth should absorb it. If it is genuinely ROI-specific structure beyond
b(ROI), a quadratic in truth should not.
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


for arm, model in ARMS.items():
    ds = load(model)
    print(f"\n{'=' * 68}\n{arm}\n{'=' * 68}")

    B = np.array([[ratio(r["est_unique_m"], r["est_unique_p"])
                   for r in d["conditions"]["B_manufacture"]["rows"]]
                  for d in ds])
    b_roi = B.mean(0)
    floor = con(b_roi)

    T = np.array([[ratio(r["true_unique_m"], r["true_unique_p"])
                   for r in d["conditions"]["A_recovery"]["rows"]] for d in ds])
    E = np.array([[ratio(r["est_unique_m"], r["est_unique_p"])
                   for r in d["conditions"]["A_recovery"]["rows"]] for d in ds])

    # is truth a deterministic function of ROI index?
    sd_across_subj = T.std(0)
    print(f"truth per ROI: {np.round(T.mean(0), 4).tolist()}")
    print(f"  across-subject SD per ROI: max {sd_across_subj.max():.2e}"
          f"   -> truth is {'DETERMINISTIC in ROI' if sd_across_subj.max() < 1e-9 else 'subject-varying'}")

    # linear fit with b(ROI) removed
    Y = E - b_roi                     # per (subj, ROI)
    a_lin = float(np.sum(T * Y) / np.sum(T * T))
    res_lin = Y - a_lin * T
    rc_lin = np.mean([con(res_lin[i]) for i in range(len(ds))])
    rec_lin = (REAL[arm] - floor) / a_lin
    print(f"\nLINEAR (b removed): a = {a_lin:.3f}")
    print(f"  residual per-ROI mean {np.round(res_lin.mean(0), 3).tolist()}")
    print(f"  residual range {res_lin.min():+.3f}..{res_lin.max():+.3f}")
    print(f"  residual CONTRAST {rc_lin:+.4f}  "
          f"= {abs(rc_lin / rec_lin):.0%} of recovered {rec_lin:+.4f}")

    # quadratic in truth, b(ROI) still removed
    X = np.column_stack([T.ravel(), T.ravel() ** 2])
    coef, *_ = np.linalg.lstsq(X, Y.ravel(), rcond=None)
    a_q, c_q = coef
    res_q = Y - (a_q * T + c_q * T ** 2)
    rc_q = np.mean([con(res_q[i]) for i in range(len(ds))])
    print(f"\nQUADRATIC in truth: a = {a_q:.3f}, c = {c_q:+.3f}")
    print(f"  residual per-ROI mean {np.round(res_q.mean(0), 3).tolist()}")
    print(f"  residual CONTRAST {rc_q:+.4f}   "
          f"(linear was {rc_lin:+.4f}; absorbed "
          f"{1 - abs(rc_q / rc_lin):.0%})")
    print(f"  residual SS: linear {np.sum(res_lin**2):.3f} -> "
          f"quadratic {np.sum(res_q**2):.3f}  "
          f"({1 - np.sum(res_q**2)/np.sum(res_lin**2):.0%} explained)")

    # a9ba8d7's fullest treatment, per subject, for the significance claim
    real_by_subj = []
    for f in sorted(glob.glob(f"{R}/results/encoding_cc/{model}_subj*.npz")):
        z = np.load(f, allow_pickle=True)
        roi, nc = z["roi_prf-visualrois"], z["noise_ceiling"]
        um, up = z["unique_magno"], z["unique_parvo"]
        prof = []
        for code in range(1, 8):
            v = (roi == code) & (nc > 0.1) & ((um + up) > 1e-3)
            r = (um[v] - up[v]) / (um[v] + up[v]) if v.sum() >= 20 else [np.nan]
            prof.append(np.nanmedian(r))
        real_by_subj.append(con(np.array(prof)))
    real_by_subj = np.array(real_by_subj)
    full = (real_by_subj - floor - rc_lin) / a_lin
    w = sps.wilcoxon(full)
    print(f"\nFULLEST treatment (floor + residual contrast removed):")
    print(f"  {full.mean():+.4f}  {int((full < 0).sum())}/{len(full)} negative"
          f"  p={w.pvalue:.4f}")
    minimal = (real_by_subj - floor) / a_lin
    wm = sps.wilcoxon(minimal)
    print(f"  minimal (floor only): {minimal.mean():+.4f}  "
          f"{int((minimal < 0).sum())}/{len(minimal)} negative  "
          f"p={wm.pvalue:.4f}")
