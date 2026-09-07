"""Which rho estimator gives b5faaee's +0.68 for the rising artefact?

The floor's two printed values are now identified:
  "+0.025 raw"        = the STORED est_diff        (mean +0.0256)
  "+0.126 normalised" = (um-up)/(um+up) ratio index (mean +0.1244, med +0.1250)
Neither of my rank correlations (+0.536 stored, +0.964 ratio) is +0.68, so the
rho must come from a third aggregation. Enumerate.
"""
import glob
import json
import numpy as np
from scipy import stats as sps

D = ("/home/byrong/projects/TSN/Magno-Parvo-Konio-Net/.claude/worktrees/"
     "mpk-dev/results/vp_recovery_realnc")
ds = [json.load(open(f)) for f in sorted(glob.glob(f"{D}/*.json"))]
nroi = len(ds[0]["conditions"]["B_manufacture"]["rows"])
rank = np.arange(nroi)


def series(d, kind):
    rs = d["conditions"]["B_manufacture"]["rows"]
    um = np.array([r["est_unique_m"] for r in rs])
    up = np.array([r["est_unique_p"] for r in rs])
    nc = np.array([r["noise_ceiling"] for r in rs])
    return {"stored": um - up, "ratio": (um - up) / (um + up),
            "raw_nc": (um - up) * nc}[kind]


for kind in ("stored", "ratio", "raw_nc"):
    per_subj_rho = np.array([sps.spearmanr(rank, series(d, kind)).statistic
                             for d in ds])
    per_subj_r = np.array([sps.pearsonr(rank, series(d, kind)).statistic
                           for d in ds])
    pooled = np.concatenate([series(d, kind) for d in ds])
    pooled_rank = np.tile(rank, len(ds))
    grp = np.array([np.mean([series(d, kind)[i] for d in ds])
                    for i in range(nroi)])
    print(f"--- {kind} ---")
    print(f"  rho on ROI means            {sps.spearmanr(rank, grp).statistic:+.3f}"
          f"  (p={sps.spearmanr(rank, grp).pvalue:.4f})")
    print(f"  mean of per-subject rho     {per_subj_rho.mean():+.3f}  "
          f"(median {np.median(per_subj_rho):+.3f}, "
          f"pos {int((per_subj_rho > 0).sum())}/{len(ds)})")
    print(f"  mean of per-subject pearson {per_subj_r.mean():+.3f}")
    print(f"  rho on pooled cells         "
          f"{sps.spearmanr(pooled_rank, pooled).statistic:+.3f}")
    # Wilcoxon on the per-subject rho against zero
    w = sps.wilcoxon(per_subj_rho)
    print(f"  Wilcoxon per-subject rho vs 0: p={w.pvalue:.4f}")
print("\ntarget rho from b5faaee: +0.68")
