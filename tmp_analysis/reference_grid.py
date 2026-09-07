"""Is the unexplained normalised gap a REFERENCE difference, not a denominator?

07d4df6 searched denominator conventions and stopped, correctly, on the grounds
that finding a fourth variant that happens to fit is not evidence. This tests
something outside that space and specified in advance.

visualrois.py prints TWO contrasts side by side:
    SPI hV4 - V1          (reference = mean(V1v, V1d))
    SPI hV4 - V1/V2/V3    (reference = mean of all six V1-V3 subdivisions)

For untrained the second is -0.4868 against a published -0.471, versus -0.5441
for the first. That is a candidate the denominator search could not reach, and
it is the kind of error this thread keeps producing: two statistics printed
adjacently under similar names.

PREDICTION, stated before the grid is computed: if the published normalised
values are V1/V2/V3 contrasts, then EVERY normalised value should match the
V123 column while the raw values keep matching the V1 column. If the V123
column fixes untrained but breaks diet, the hypothesis is dead -- the test is
all twelve simultaneously, not one gap closed.
"""
import glob
import numpy as np

R = "/home/byrong/projects/TSN/Magno-Parvo-Konio-Net/.claude/worktrees/mpk-dev"
NC_MIN, MIN_VOX, DENOM_MIN = 0.1, 20, 1e-3
V1 = [0, 1]                      # V1v, V1d
V123 = [0, 1, 2, 3, 4, 5]        # V1v..V3d

# published (model, src, norm, raw); no-diet seed0 is explained by §88's
# defective subj01 and is shown only for completeness
PUB = [
    ("untrained_seed0", "encoding", -0.471, -0.118),
    ("dkl_bcs_seed0_trained", "encoding", -0.862, -0.176),
    ("dkl_base_seed0_trained", "encoding", -0.101, -0.034),
]


def profiles(model, src):
    norm, raw = [], []
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
                # median-of-DIFFERENCES: the paper's raw statistic (§92.2).
                # This file originally used med(u_m) - med(u_p), which is a
                # different estimator and was identified as the wrong one only
                # after the grid was first run. The verdicts are unchanged --
                # V1 wins every raw row either way -- but the numbers below
                # are now in the paper's own convention.
                b.append(np.median(um[v] - up[v]))
            else:
                a.append(np.nan); b.append(np.nan)
        norm.append(a); raw.append(b)
    return np.array(norm), np.array(raw)


def contrast(P, ref):
    return (P[:, 6] - np.nanmean(P[:, ref], axis=1)).mean()


print(f"{'model':26s} {'stat':5s} {'V1 ref':>9s} {'V123 ref':>9s} "
      f"{'published':>10s} {'|V1-pub|':>9s} {'|V123-pub|':>11s} {'winner':>7s}")
score = {"V1": 0, "V123": 0}
for model, src, pub_n, pub_r in PUB:
    norm, raw = profiles(model, src)
    for stat, P, pub in (("norm", norm, pub_n), ("raw", raw, pub_r)):
        c1, c2 = contrast(P, V1), contrast(P, V123)
        d1, d2 = abs(c1 - pub), abs(c2 - pub)
        win = "V1" if d1 < d2 else "V123"
        score[win] += 1
        print(f"{model:26s} {stat:5s} {c1:+9.4f} {c2:+9.4f} {pub:+10.3f} "
              f"{d1:9.4f} {d2:11.4f} {win:>7s}")

print(f"\ncloser-to-published tally: {score}")
print("\nThe hypothesis needs V123 to win EVERY normalised row while V1 wins")
print("every raw row. Any other pattern means the reference is not the cause.")

# the three-seed normalised values, to see whether the pattern is stable
print("\n--- normalised contrast, both references, all six seed fits ---")
print(f"{'model':26s} {'V1 ref':>9s} {'V123 ref':>9s}")
for a in ("base", "bcs"):
    for s in (0, 1, 2):
        m = f"dkl_{a}_seed{s}_trained"
        norm, _ = profiles(m, "encoding_cc")
        if not len(norm):
            continue
        print(f"{m:26s} {contrast(norm, V1):+9.4f} "
              f"{contrast(norm, V123):+9.4f}")
