"""Does §86.5's estimator mismatch warrant a per-voxel re-run of condition B?

§86.5 recorded that the +0.051 floor is RATIO-OF-MEDIANS (all the recovery
JSONs store are per-ROI medians) while §2.2's effect is MEDIAN-OF-RATIOS, and
said the proper fix was re-running condition B recording the per-voxel ratio.
That is ~8 x 1.5 h of CPU. Before spending it, bound the gap from data already
on disk and check whether the bound moves any claim.

Method: on real fits both estimators are computable, so measure the gap across
all six model x arm combinations and see how it scales with effect magnitude.
If the gap is roughly PROPORTIONAL to the effect, the floor (+0.051, small)
inherits a small absolute gap and the re-run is not warranted. If it is roughly
CONSTANT in absolute terms, the floor could be off by as much as the gap seen
on the large-effect fits, and the re-run is warranted.
"""
import glob
import numpy as np

R = "/home/byrong/projects/TSN/Magno-Parvo-Konio-Net/.claude/worktrees/mpk-dev"
NC_MIN, MIN_VOX, DENOM_MIN = 0.1, 20, 1e-3
FLOOR_RATIO = 0.0511      # §86 / §2.2 units, no-diet geometry
DIET, NODIET = 0.8335, 0.0785   # recomputed |contrast|, §86.4


def both_estimators(model, src="encoding_cc"):
    mor, rom = [], []
    for f in sorted(glob.glob(f"{R}/results/{src}/{model}_subj*.npz")):
        z = np.load(f, allow_pickle=True)
        roi, nc = z["roi_prf-visualrois"], z["noise_ceiling"]
        um, up = z["unique_magno"], z["unique_parvo"]
        a, b = [], []
        for code in range(1, 8):
            v = (roi == code) & (nc > NC_MIN) & ((um + up) > DENOM_MIN)
            if v.sum() >= MIN_VOX:
                r = (um[v] - up[v]) / (um[v] + up[v])
                a.append(np.median(r[np.isfinite(r)]))
                mm, mp = np.median(um[v]), np.median(up[v])
                b.append((mm - mp) / (mm + mp))
            else:
                a.append(np.nan); b.append(np.nan)
        a, b = np.array(a), np.array(b)
        mor.append(a[6] - np.nanmean(a[[0, 1]]))
        rom.append(b[6] - np.nanmean(b[[0, 1]]))
    return np.array(mor), np.array(rom)


MODELS = [f"dkl_{a}_seed{s}_trained" for a in ("base", "bcs")
          for s in (0, 1, 2)]
print(f"{'model':26s} {'med-of-rat':>11s} {'rat-of-med':>11s} "
      f"{'gap':>8s} {'gap/|eff|':>10s}")
eff, gap = [], []
for m in MODELS:
    a, b = both_estimators(m)
    if not len(a):
        continue
    A, B = a.mean(), b.mean()
    g = A - B
    eff.append(abs(A)); gap.append(abs(g))
    print(f"{m:26s} {A:+11.4f} {B:+11.4f} {g:+8.4f} {abs(g / A):10.1%}")

eff, gap = np.array(eff), np.array(gap)
r = np.corrcoef(eff, gap)[0, 1]
print(f"\ncorr(|effect|, |gap|) = {r:+.3f}")
print(f"|gap| absolute:     mean {gap.mean():.4f}  range "
      f"{gap.min():.4f}..{gap.max():.4f}")
frac = gap / eff
print(f"|gap| proportional: mean {frac.mean():.1%}  range "
      f"{frac.min():.1%}..{frac.max():.1%}")

# Two competing extrapolations to the floor's magnitude
print(f"\n--- extrapolating to the floor (+{FLOOR_RATIO:.4f}) ---")
b_prop = FLOOR_RATIO * frac.mean()
b_const = gap.mean()
print(f"if PROPORTIONAL: floor gap ~ {b_prop:.4f}  -> floor in "
      f"[{FLOOR_RATIO - b_prop:.4f}, {FLOOR_RATIO + b_prop:.4f}]")
print(f"if CONSTANT    : floor gap ~ {b_const:.4f}  -> floor in "
      f"[{FLOOR_RATIO - b_const:.4f}, {FLOOR_RATIO + b_const:.4f}]")

print("\n--- does either bound move a claim? ---")
for name, lo, hi in (("proportional", FLOOR_RATIO - b_prop,
                      FLOOR_RATIO + b_prop),
                     ("constant", max(FLOOR_RATIO - b_const, 1e-6),
                      FLOOR_RATIO + b_const)):
    print(f"  {name}: diet {DIET / hi:.1f}x..{DIET / lo:.1f}x the floor;  "
          f"no-diet {NODIET / hi:.1f}x..{NODIET / lo:.1f}x")
print("\n§2.2 claims a large gradient with the diet and none resolvable")
print("without it. That survives iff diet stays >> 1 and no-diet stays ~1.")
