"""Verify the diet-arm floor, and test the inference cedadc6 flagged.

Their numbers to check: diet floor +0.0375 normalised / -0.00213 raw, 8/8 and
2/8, rho +0.804, range [+0.0141,+0.0651]; Mann-Whitney between geometries
p=0.328; condition-A fidelity x1.151 (ratio) and x0.686 (difference).

Then the question they raised. They compare |effect| to |floor| as a ratio and
read "no-diet can sit below its own floor" as STRENGTHENING "no gradient
resolvable without the diet". But the floor is a SIGNED bias (+0.051, toward
magno) and the effect is NEGATIVE (-0.054). A ratio of magnitudes treats the
floor as if it were a noise level. For a signed bias pointing the other way the
right comparison is per-subject and paired: is the real contrast displaced from
what the SAME subject's instrument returns under a true zero?
"""
import glob
import json
import numpy as np
from scipy import stats as sps

R = "/home/byrong/projects/TSN/Magno-Parvo-Konio-Net/.claude/worktrees/mpk-dev"
NC_MIN, MIN_VOX, DENOM_MIN = 0.1, 20, 1e-3
ARMS = {"no-diet": "dkl_base_seed0_trained", "diet": "dkl_bcs_seed0_trained"}


def floor_by_subject(model):
    """Condition-B contrast per subject: ratio index and difference."""
    out = {}
    for f in sorted(glob.glob(
            f"{R}/results/vp_recovery_realnc/vp_recovery_{model}_subj*.json")):
        d = json.load(open(f))
        rs = d["conditions"]["B_manufacture"]["rows"]
        um = np.array([r["est_unique_m"] for r in rs])
        up = np.array([r["est_unique_p"] for r in rs])
        assert max(abs(r["true_diff"]) for r in rs) < 1e-9
        ratio = (um - up) / (um + up)
        diff = um - up
        out[d["subject"]] = {
            "ratio": ratio[6] - ratio[[0, 1]].mean(),
            "diff": diff[6] - diff[[0, 1]].mean(),
            "ratio_prof": ratio,
        }
    return out


def real_by_subject(model):
    """Real contrast per subject, §2.2's conventions."""
    out = {}
    for f in sorted(glob.glob(f"{R}/results/encoding_cc/{model}_subj*.npz")):
        s = int(f.split("subj")[-1].split(".")[0])
        z = np.load(f, allow_pickle=True)
        roi, nc = z["roi_prf-visualrois"], z["noise_ceiling"]
        um, up = z["unique_magno"], z["unique_parvo"]
        pr, pd = [], []
        for code in range(1, 8):
            v = (roi == code) & (nc > NC_MIN) & ((um + up) > DENOM_MIN)
            if v.sum() >= MIN_VOX:
                r = (um[v] - up[v]) / (um[v] + up[v])
                pr.append(np.median(r[np.isfinite(r)]))
                pd.append(np.median(um[v] - up[v]))
            else:
                pr.append(np.nan); pd.append(np.nan)
        pr, pd = np.array(pr), np.array(pd)
        out[s] = {"ratio": pr[6] - np.nanmean(pr[[0, 1]]),
                  "diff": pd[6] - np.nanmean(pd[[0, 1]])}
    return out


print("=" * 70)
print("VERIFY cedadc6's floor numbers")
print("=" * 70)
fl = {k: floor_by_subject(v) for k, v in ARMS.items()}
for arm in ("no-diet", "diet"):
    f = fl[arm]
    subs = sorted(f)
    ra = np.array([f[s]["ratio"] for s in subs])
    di = np.array([f[s]["diff"] for s in subs])
    rho = np.mean([sps.spearmanr(np.arange(7), f[s]["ratio_prof"]).statistic
                   for s in subs])
    print(f"\n{arm:8s} n={len(subs)}")
    print(f"  normalised floor {ra.mean():+.4f}  {int((ra > 0).sum())}/{len(ra)}"
          f"  p={sps.wilcoxon(ra).pvalue:.4f}  range "
          f"[{ra.min():+.4f},{ra.max():+.4f}]")
    print(f"  raw floor        {di.mean():+.5f}  "
          f"{int((di > 0).sum())}/{len(di)}  p={sps.wilcoxon(di).pvalue:.4f}")
    print(f"  mean per-subject rho vs rank {rho:+.3f}")

a = np.array([fl["no-diet"][s]["ratio"] for s in sorted(fl["no-diet"])])
b = np.array([fl["diet"][s]["ratio"] for s in sorted(fl["diet"])])
mw = sps.mannwhitneyu(a, b)
print(f"\nMann-Whitney between geometries: U={mw.statistic:.1f} "
      f"p={mw.pvalue:.4f}   (they report p=0.328)")
print(f"point estimates {a.mean():+.4f} vs {b.mean():+.4f}  "
      f"= {abs(a.mean() / b.mean() - 1):.0%} apart")

print("\n" + "=" * 70)
print("CONDITION A fidelity on each geometry")
print("=" * 70)
for arm, model in ARMS.items():
    T = {"ratio": [], "diff": []}
    E = {"ratio": [], "diff": []}
    for f in sorted(glob.glob(
            f"{R}/results/vp_recovery_realnc/vp_recovery_{model}_subj*.json")):
        d = json.load(open(f))
        for r in d["conditions"]["A_recovery"]["rows"]:
            for tag, acc in (("true", T), ("est", E)):
                um, up = r[f"{tag}_unique_m"], r[f"{tag}_unique_p"]
                acc["ratio"].append((um - up) / (um + up))
                acc["diff"].append(um - up)
    for stat in ("ratio", "diff"):
        sl = np.polyfit(T[stat], E[stat], 1)[0]
        print(f"  {arm:8s} {stat:5s} est = {sl:.3f} x true")

print("\n" + "=" * 70)
print("THE INFERENCE: paired real-vs-own-instrument, per subject")
print("=" * 70)
print("The floor is a SIGNED positive bias; the effect is negative. So test")
print("displacement from the instrument's null output, not a magnitude ratio.")
for arm, model in ARMS.items():
    real = real_by_subject(model)
    f = fl[arm]
    subs = sorted(set(real) & set(f))
    for stat in ("ratio",):
        rr = np.array([real[s][stat] for s in subs])
        ff = np.array([f[s][stat] for s in subs])
        d = rr - ff
        w = sps.wilcoxon(d)
        print(f"\n{arm:8s} ({stat})  n={len(subs)}")
        print(f"  real            {rr.mean():+.4f}  "
              f"per-subj {np.round(rr, 3).tolist()}")
        print(f"  own floor       {ff.mean():+.4f}  "
              f"per-subj {np.round(ff, 3).tolist()}")
        print(f"  real - floor    {d.mean():+.4f}  "
              f"{int((d < 0).sum())}/{len(d)} negative  p={w.pvalue:.4f}")
        # leave-one-out on subj01 for the no-diet arm
        if arm == "no-diet" and 1 in subs:
            k = [i for i, s in enumerate(subs) if s != 1]
            d7 = d[k]
            print(f"  n=7 (drop subj01) {d7.mean():+.4f}  "
                  f"{int((d7 < 0).sum())}/{len(d7)} negative  "
                  f"p={sps.wilcoxon(d7).pvalue:.4f}")
