"""Verify e1ce0d3's estimated-slope correction, which replaces my additivity claim.

My §93.4 justified `real - floor` by the two floors being indistinguishable
across geometries. e1ce0d3 correctly points out that condition B sets truth to
ZERO in both, so that shows geometry-independence, not independence from the
true contrast -- which is the property the subtraction needs. Their fix uses
condition A to ESTIMATE the multiplicative term instead of assuming it away:

    est = a * truth + b,  with b fixed at the measured condition-B floor
    reported: a = 1.217 (no diet), 1.122 (diet)
    recovered true contrasts -0.106 and -0.777, ratio 7.3x

Checked here, including which fitting unit gives those slopes, since the
recurring disagreements in this thread have all been unstated estimators.
"""
import glob
import json
import numpy as np

R = "/home/byrong/projects/TSN/Magno-Parvo-Konio-Net/.claude/worktrees/mpk-dev"
ARMS = {"no-diet": "dkl_base_seed0_trained", "diet": "dkl_bcs_seed0_trained"}
REAL = {"no-diet": -0.0785, "diet": -0.8335}      # §2.2 units, recomputed
FLOOR = {"no-diet": 0.0511, "diet": 0.0375}


def load(model):
    return [json.load(open(f)) for f in sorted(glob.glob(
        f"{R}/results/vp_recovery_realnc/vp_recovery_{model}_subj*.json"))]


def ratio(um, up):
    return (um - up) / (um + up)


for arm, model in ARMS.items():
    ds = load(model)
    # (i) per-ROI ratio values, 56 points
    t_roi, e_roi = [], []
    # (ii) per-subject CONTRAST, 8 points -- the unit the displacement uses
    t_con, e_con = [], []
    for d in ds:
        rs = d["conditions"]["A_recovery"]["rows"]
        tm = np.array([r["true_unique_m"] for r in rs])
        tp = np.array([r["true_unique_p"] for r in rs])
        em = np.array([r["est_unique_m"] for r in rs])
        ep = np.array([r["est_unique_p"] for r in rs])
        tr, er = ratio(tm, tp), ratio(em, ep)
        t_roi += tr.tolist(); e_roi += er.tolist()
        t_con.append(tr[6] - tr[[0, 1]].mean())
        e_con.append(er[6] - er[[0, 1]].mean())
    b = FLOOR[arm]
    print(f"\n=== {arm} (floor b = {b:+.4f}) ===")
    for label, T, E in (("per-ROI ratio (n=56)", np.array(t_roi),
                         np.array(e_roi)),
                        ("per-subject contrast (n=8)", np.array(t_con),
                         np.array(e_con))):
        # slope with intercept FIXED at the measured floor
        a_fix = float(np.sum(T * (E - b)) / np.sum(T * T))
        # free OLS, for comparison
        a_free, b_free = np.polyfit(T, E, 1)
        rec = (REAL[arm] - b) / a_fix
        print(f"  {label:28s} a|b fixed = {a_fix:.3f}   "
              f"free OLS a = {a_free:.3f}, b = {b_free:+.4f}")
        print(f"  {'':28s} -> recovered true contrast "
              f"{rec:+.4f}")

print("\n--- ratio between arms under each fitting unit ---")
for label in ("roi", "con"):
    recs = {}
    for arm, model in ARMS.items():
        ds = load(model)
        T, E = [], []
        for d in ds:
            rs = d["conditions"]["A_recovery"]["rows"]
            tr = ratio(np.array([r["true_unique_m"] for r in rs]),
                       np.array([r["true_unique_p"] for r in rs]))
            er = ratio(np.array([r["est_unique_m"] for r in rs]),
                       np.array([r["est_unique_p"] for r in rs]))
            if label == "roi":
                T += tr.tolist(); E += er.tolist()
            else:
                T.append(tr[6] - tr[[0, 1]].mean())
                E.append(er[6] - er[[0, 1]].mean())
        T, E = np.array(T), np.array(E)
        b = FLOOR[arm]
        a = float(np.sum(T * (E - b)) / np.sum(T * T))
        recs[arm] = ((REAL[arm] - b) / a, a)
    r = recs["diet"][0] / recs["no-diet"][0]
    print(f"  {label}: a = {recs['no-diet'][1]:.3f} / {recs['diet'][1]:.3f}, "
          f"true = {recs['no-diet'][0]:+.4f} / {recs['diet'][0]:+.4f}, "
          f"ratio {r:.1f}x")
print("\ne1ce0d3 reports a = 1.217 / 1.122, true -0.106 / -0.777, 7.3x")
print("\nNote the conclusion does not rest on a: any a > 0 rescales both arms")
print("without changing either sign or the per-subject consistency, which come")
print("from the displacement itself. The slope refines the MAGNITUDE only.")
