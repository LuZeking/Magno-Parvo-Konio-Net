"""Is subj01's displacement arm-specific, or the same shift in every fit?

The peer's inference was: "a subject-level peculiarity would show in both arms;
a wrong-front-end no-diet fit shows in one." The 4.65x / 1.23x ratios do look
arm-specific. But the arms have very different baselines (-0.054 vs -0.808), so
a RATIO against each baseline is not the right comparison. Measure the ABSOLUTE
displacement instead, and test it across seeds 1 and 2, which the bad job
(26386734_0, 2026-08-30) never touched.
"""
import glob
import numpy as np

R = "/home/byrong/projects/TSN/Magno-Parvo-Konio-Net/.claude/worktrees/mpk-dev"
NC_MIN, MIN_VOX = 0.1, 20
ORDER = [1, 2, 3, 4, 5, 6, 7]


def contrast_by_subject(model):
    out = {}
    for f in sorted(glob.glob(f"{R}/results/encoding_cc/{model}_subj*.npz")):
        s = int(f.split("subj")[-1].split(".")[0])
        z = np.load(f, allow_pickle=True)
        roi, nc = z["roi_prf-visualrois"], z["noise_ceiling"]
        um, up = z["unique_magno"], z["unique_parvo"]
        prof = []
        for code in ORDER:
            v = (roi == code) & (nc > NC_MIN) & ((um + up) > 1e-3)
            if v.sum() >= MIN_VOX:
                r = (um[v] - up[v]) / (um[v] + up[v])
                prof.append(np.median(r[np.isfinite(r)]))
            else:
                prof.append(np.nan)
        prof = np.array(prof)
        out[s] = prof[6] - np.nanmean(prof[[0, 1]])
    return out


MODELS = [f"dkl_{a}_seed{s}_trained" for a in ("base", "bcs")
          for s in (0, 1, 2)]
print(f"{'model':26s} {'subj01':>8s} {'sib med':>8s} {'ratio':>7s} "
      f"{'DISPLACEMENT':>13s}")
disp = {}
for m in MODELS:
    c = contrast_by_subject(m)
    if 1 not in c or len(c) < 4:
        print(f"{m:26s}  -- not available ({len(c)} subjects)")
        continue
    me = c[1]
    sib = np.array([v for s, v in c.items() if s != 1])
    med = np.median(sib)
    disp[m] = me - med
    print(f"{m:26s} {me:+8.4f} {med:+8.4f} {me / med:7.2f}x {me - med:+13.4f}")

print("\nThe ratio is arm-specific because the baselines differ 15-fold.")
print("The DISPLACEMENT is the estimator-free quantity. If it is the same in")
print("both arms and across seeds the bad job never touched, subj01 is a")
print("subject-level effect, not a defective no-diet fit.")
base = [v for k, v in disp.items() if "base" in k]
bcs = [v for k, v in disp.items() if "bcs" in k]
if base and bcs:
    print(f"\n  no-diet displacements: {np.round(base, 4).tolist()}  "
          f"mean {np.mean(base):+.4f}")
    print(f"  diet    displacements: {np.round(bcs, 4).tolist()}  "
          f"mean {np.mean(bcs):+.4f}")
    print(f"  difference of means: {np.mean(base) - np.mean(bcs):+.4f}")
