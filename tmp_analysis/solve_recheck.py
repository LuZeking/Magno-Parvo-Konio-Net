"""Redo §88's raw solve under the paper's ACTUAL raw estimator.

Identified: the paper's "raw" is MEDIAN-OF-DIFFERENCES -- median over voxels of
(u_m - u_p) -- not the difference of the two medians. §88 used
difference-of-medians and reported the required subj01 as -0.1275, 6.0x the
sibling median, 47% of the effect. 07d4df6's solve gave -0.1241, 6.3x, 46% and
attributed our 0.003 gap to rounding off the 3-dp published -0.034.

If the gap is the estimator rather than rounding, redoing my solve under
median-of-differences should land on THEIR numbers.
"""
import glob
import numpy as np

R = "/home/byrong/projects/TSN/Magno-Parvo-Konio-Net/.claude/worktrees/mpk-dev"
NC_MIN, MIN_VOX, DENOM_MIN = 0.1, 20, 1e-3
PUB_RAW = -0.034


def per_subject(model, src="encoding", how="median_of_diff"):
    out = {}
    for f in sorted(glob.glob(f"{R}/results/{src}/{model}_subj*.npz")):
        s = int(f.split("subj")[-1].split(".")[0])
        z = np.load(f, allow_pickle=True)
        roi, nc = z["roi_prf-visualrois"], z["noise_ceiling"]
        um, up = z["unique_magno"], z["unique_parvo"]
        prof = []
        for code in range(1, 8):
            v = (roi == code) & (nc > NC_MIN) & ((um + up) > DENOM_MIN)
            if v.sum() >= MIN_VOX:
                prof.append(np.median(um[v] - up[v]) if how == "median_of_diff"
                            else np.median(um[v]) - np.median(up[v]))
            else:
                prof.append(np.nan)
        prof = np.array(prof)
        out[s] = prof[6] - np.nanmean(prof[[0, 1]])
    return out


for how in ("median_of_diff", "diff_of_medians"):
    c = per_subject("dkl_base_seed0_trained", how=how)
    subs = sorted(c)
    sib = np.array([c[s] for s in subs if s != 1])
    n = len(subs)
    need = n * PUB_RAW - sib.sum()
    med = np.median(sib)
    print(f"\n=== raw as {how} ===")
    print(f"  n=8 mean {np.mean([c[s] for s in subs]):+.4f}   "
          f"n=7 mean {sib.mean():+.4f}   subj01 now {c[1]:+.4f}")
    print(f"  to reach published {PUB_RAW:+.3f}, subj01 must be {need:+.4f}")
    print(f"  = {need / med:.1f}x the sibling median ({med:+.4f}), "
          f"{abs(need / n / PUB_RAW):.0%} of the effect")

print("\n07d4df6's solve: -0.1241, 6.3x, 46%")
print("§88's solve:     -0.1275, 6.0x, 47%")
print("\nIf median_of_diff reproduces -0.1241/6.3x/46%, the 0.003 gap between")
print("the two sessions was the ESTIMATOR, not rounding -- and the README's")
print("'six-fold' is matched at 6.3x under the paper's own convention.")
