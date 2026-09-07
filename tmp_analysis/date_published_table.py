"""Can the published table be dated by reverse-engineering the missing subject?

Established by git/mtime archaeology (§88):
  visualrois.py   unchanged since 2026-08-28, and hardcodes untrained_seed0
  stats.py        unchanged since 2026-08-29
  diet fits       results/encoding/*bcs_seed0*  all 2026-08-30
  cc fits         results/encoding_cc/*         2026-09-01
  table written   2026-09-06 14:24 (c0002a2, added as prose)

So nothing the table depends on changed AFTER it was written. That rules out
post-hoc code/data drift and leaves the state the numbers were computed FROM.

Hypothesis for the no-diet arm: they were computed BEFORE the 2026-09-01
quarantine, i.e. with the DEFECTIVE subj01 still in the n=8 set. Test it by
solving for the subj01 value that the published mean requires, then checking
that value against the quarantine README's two independent descriptions of the
defective file ("six-fold outlier", "contributed 57% of the effect size").
"""
import glob
import numpy as np

R = "/home/byrong/projects/TSN/Magno-Parvo-Konio-Net/.claude/worktrees/mpk-dev"
NC_MIN, MIN_VOX = 0.1, 20


def profiles(model, src):
    out = {}
    for f in sorted(glob.glob(f"{R}/results/{src}/{model}_subj*.npz")):
        s = int(f.split("subj")[-1].split(".")[0])
        z = np.load(f, allow_pickle=True)
        roi, nc = z["roi_prf-visualrois"], z["noise_ceiling"]
        um, up = z["unique_magno"], z["unique_parvo"]
        pr, pd = [], []
        for code in range(1, 8):
            v = (roi == code) & (nc > NC_MIN) & ((um + up) > 1e-3)
            if v.sum() >= MIN_VOX:
                r = (um[v] - up[v]) / (um[v] + up[v])
                pr.append(np.median(r[np.isfinite(r)]))
                pd.append(np.median(um[v]) - np.median(up[v]))
            else:
                pr.append(np.nan); pd.append(np.nan)
        pr, pd = np.array(pr), np.array(pd)
        out[s] = (pr[6] - np.nanmean(pr[[0, 1]]),
                  pd[6] - np.nanmean(pd[[0, 1]]))
    return out


def solve(model, pub_ratio, pub_diff, label):
    c = profiles(model, "encoding")
    subs = sorted(c)
    print(f"\n=== {label} ({model}) ===")
    for idx, (name, pub) in enumerate((("ratio", pub_ratio),
                                       ("diff", pub_diff))):
        vals = np.array([c[s][idx] for s in subs])
        sib = np.array([c[s][idx] for s in subs if s != 1])
        cur01 = c[1][idx]
        n = len(subs)
        # what value at subj01 makes the n=8 mean equal the published number?
        need = n * pub - sib.sum()
        print(f"  {name:5s} published {pub:+.4f}   current n={n} "
              f"{vals.mean():+.4f}   n=7 {sib.mean():+.4f}")
        print(f"        subj01 now {cur01:+.4f};  to reach the published mean "
              f"it would need {need:+.4f}")
        print(f"        that is {need / np.median(sib):.1f}x the sibling "
              f"median ({np.median(sib):+.4f}), and {abs(need / n / pub):.0%} "
              f"of the published effect")
        print(f"        sibling range {sib.min():+.4f}..{sib.max():+.4f}")


solve("dkl_base_seed0_trained", -0.101, -0.034, "NO DIET")
solve("dkl_bcs_seed0_trained", -0.862, -0.176, "DIET (control: never defective)")
print("\nREADME describes the defective file as a SIX-FOLD outlier that")
print("contributed 57% of the no-diet effect. If the solved values land near")
print("6x and 57% for the no-diet arm, the published no-diet numbers date to")
print("BEFORE the 2026-09-01 quarantine. The diet arm had no defective file,")
print("so an implausible solved value there means its gap has another cause.")
