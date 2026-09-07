"""Is subj01 still an hV4-V1 outlier, and is it arm-asymmetric?

Slurm settles the provenance question: task 16 of 26399760 ran
model=dkl_base_seed0_trained subject=1 frontend='--no-lum-blur --no-lum-cs'
-- the CORRECT front end, features recomputed rather than trusted from cache --
and saved to results/encoding_cc/. The file now in results/encoding/ carries
that task's exact mtime and is byte-identical, so it IS the repair.

So the open question is no longer "which file is this" but "why is a
correct-front-end fit still outlying". Measured here per subject, in §2.2's
own statistic (median-of-ratios, then hV4 - mean(V1v, V1d)).
"""
import glob
import numpy as np

R = "/home/byrong/projects/TSN/Magno-Parvo-Konio-Net/.claude/worktrees/mpk-dev"
NC_MIN, MIN_VOX = 0.1, 20
ORDER = [(1, "V1v"), (2, "V1d"), (3, "V2v"), (4, "V2d"),
         (5, "V3v"), (6, "V3d"), (7, "hV4")]


def contrasts(model, src="encoding_cc"):
    out = {}
    for f in sorted(glob.glob(f"{R}/results/{src}/{model}_subj*.npz")):
        s = int(f.split("subj")[-1].split(".")[0])
        z = np.load(f, allow_pickle=True)
        roi, nc = z["roi_prf-visualrois"], z["noise_ceiling"]
        um, up = z["unique_magno"], z["unique_parvo"]
        prof_r, prof_d = [], []
        for code, _ in ORDER:
            v = (roi == code) & (nc > NC_MIN) & ((um + up) > 1e-3)
            if v.sum() >= MIN_VOX:
                r = (um[v] - up[v]) / (um[v] + up[v])
                prof_r.append(np.median(r[np.isfinite(r)]))
                prof_d.append(np.median(um[v]) - np.median(up[v]))
            else:
                prof_r.append(np.nan); prof_d.append(np.nan)
        pr, pd = np.array(prof_r), np.array(prof_d)
        out[s] = (pr[6] - np.nanmean(pr[[0, 1]]),
                  pd[6] - np.nanmean(pd[[0, 1]]))
    return out


for model, tag in (("dkl_base_seed0_trained", "NO DIET (the arm in question)"),
                   ("dkl_bcs_seed0_trained", "DIET (control for subject)")):
    c = contrasts(model)
    subs = sorted(c)
    ratio = np.array([c[s][0] for s in subs])
    diff = np.array([c[s][1] for s in subs])
    print(f"\n=== {model} -- {tag} ===")
    print(f"{'subj':>4s} {'ratio hV4-V1':>13s} {'diff hV4-V1':>12s}")
    for s, r, d in zip(subs, ratio, diff):
        mark = "  <-- subj01" if s == 1 else ""
        print(f"{s:4d} {r:13.4f} {d:12.4f}{mark}")
    for name, v in (("ratio", ratio), ("diff", diff)):
        sib = np.array([v[i] for i, s in enumerate(subs) if s != 1])
        me = v[subs.index(1)]
        med = np.median(sib)
        print(f"  {name}: subj01 {me:+.4f}  sibling median {med:+.4f}  "
              f"ratio {me / med:.2f}x  sibling range "
              f"{sib.min():+.4f}..{sib.max():+.4f}")
        print(f"       n=8 mean {v.mean():+.4f}   n=7 (drop subj01) "
              f"{sib.mean():+.4f}")
