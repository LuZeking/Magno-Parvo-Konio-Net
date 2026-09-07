"""Was §94.4's model check mis-specified? And does the identity carry evidence?

26bc5a3 argues:
 (a) condition B's bias is NOT constant across ROIs (+0.099->+0.148 no diet,
     +0.132->+0.164 diet), so a pooled free intercept estimates the MEAN
     per-ROI bias while the contrast floor is a DIFFERENCE of per-ROI biases --
     different quantities, and §94.4 compared them;
 (b) against the mean per-ROI bias the form fits diet nearly exactly (0.96x)
     and no-diet loosely (0.62x) -- the REVERSE of §94.4's direction;
 (c) with est(ROI) = a*true(ROI) + b(ROI), the contrast bias is
     b(hV4) - b(V1) by construction and is numerically exact at the measured
     floors, so `real - floor` needs no structure-independence assumption;
 (d) bias-removed slopes a = 1.281 / 1.210, true -0.1011 / -0.7198, 7.1x,
     residual intercepts -0.0428 / -0.0016.

All four checked here. (c) is the one to look at hardest: a contrast of a
profile IS the difference of that profile's values, so exactness may be
arithmetic rather than evidential.
"""
import glob
import json
import numpy as np

R = "/home/byrong/projects/TSN/Magno-Parvo-Konio-Net/.claude/worktrees/mpk-dev"
ARMS = {"no-diet": "dkl_base_seed0_trained", "diet": "dkl_bcs_seed0_trained"}
REAL = {"no-diet": -0.0785, "diet": -0.8335}


def load(model):
    return [json.load(open(f)) for f in sorted(glob.glob(
        f"{R}/results/vp_recovery_realnc/vp_recovery_{model}_subj*.json"))]


def ratio(m, p):
    return (m - p) / (m + p)


for arm, model in ARMS.items():
    ds = load(model)
    print(f"\n{'=' * 66}\n{arm}\n{'=' * 66}")

    # (a) per-ROI bias profile from condition B (truth is exactly zero there)
    B = np.array([[ratio(r["est_unique_m"], r["est_unique_p"])
                   for r in d["conditions"]["B_manufacture"]["rows"]]
                  for d in ds])
    b_roi = B.mean(0)
    print(f"(a) per-ROI bias {np.round(b_roi, 4).tolist()}")
    print(f"    rises {b_roi[0]:+.3f} -> {b_roi[6]:+.3f}; "
          f"constant? spread {b_roi.max() - b_roi.min():.4f}")

    # (b) identity: contrast of the profile == difference of profile values
    contrast_floor = np.mean([B[i, 6] - B[i, [0, 1]].mean()
                              for i in range(len(B))])
    identity = b_roi[6] - b_roi[[0, 1]].mean()
    print(f"(b) b(hV4) - b(V1) = {identity:+.6f};  measured contrast floor "
          f"= {contrast_floor:+.6f};  diff {identity - contrast_floor:+.2e}")
    print(f"    mean per-ROI bias = {b_roi.mean():+.4f}")

    # (c) pooled free fit vs the two candidate references
    T, E = [], []
    for d in ds:
        rs = d["conditions"]["A_recovery"]["rows"]
        T += ratio(np.array([r["true_unique_m"] for r in rs]),
                   np.array([r["true_unique_p"] for r in rs])).tolist()
        E += ratio(np.array([r["est_unique_m"] for r in rs]),
                   np.array([r["est_unique_p"] for r in rs])).tolist()
    T, E = np.array(T), np.array(E)
    a_free, b_free = np.polyfit(T, E, 1)
    print(f"(c) pooled free fit: a = {a_free:.3f}, intercept "
          f"{b_free:+.4f}")
    print(f"    vs mean per-ROI bias {b_roi.mean():+.4f} -> "
          f"{b_free / b_roi.mean():.2f}x   <- 26bc5a3's reference")
    print(f"    vs contrast floor    {contrast_floor:+.4f} -> "
          f"{b_free / contrast_floor:.2f}x   <- §94.4's reference")

    # (d) bias-removed fit: subtract each ROI's OWN bias, then fit slope
    Tb, Eb = [], []
    for d in ds:
        rs = d["conditions"]["A_recovery"]["rows"]
        tr = ratio(np.array([r["true_unique_m"] for r in rs]),
                   np.array([r["true_unique_p"] for r in rs]))
        er = ratio(np.array([r["est_unique_m"] for r in rs]),
                   np.array([r["est_unique_p"] for r in rs]))
        Tb += tr.tolist()
        Eb += (er - b_roi).tolist()
    Tb, Eb = np.array(Tb), np.array(Eb)
    a_bias = float(np.sum(Tb * Eb) / np.sum(Tb * Tb))
    a_r, b_r = np.polyfit(Tb, Eb, 1)
    rec = (REAL[arm] - contrast_floor) / a_bias
    print(f"(d) bias-removed: a = {a_bias:.3f}   residual free intercept "
          f"{b_r:+.4f}")
    print(f"    recovered true contrast {rec:+.4f}")
    print(f"    residual intercept as share of recovered contrast: "
          f"{abs(b_r / rec):.0%}")

print("\n" + "=" * 66)
print("26bc5a3 reports: mean-bias refs 0.96x diet / 0.62x no-diet;")
print("a = 1.281 / 1.210; true -0.1011 / -0.7198; 7.1x;")
print("residual intercepts -0.0428 / -0.0016")
