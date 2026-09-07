"""Which definition of the artefact floor gives +0.126 / +0.025 and rho=+0.68?

b5faaee's commit message reports the condition-B floor as
"+0.126 normalised / +0.025 raw, 8/8 subjects, RISING with rank (rho = +0.68)".
My own aggregation of the same eight files gives +0.0256 for the STORED
est_diff and +0.0094 once nc is multiplied back out. Those cannot both be
right, so enumerate the candidate definitions and see which reproduces which
number. A definition that reproduces the printed value identifies it; one that
does not, rules itself out.

Key fact about the stored value: vp_recovery_realnc.py calls
variance_partition(..., noise_ceiling=ceil), so est_unique_* -- and therefore
est_diff -- is ALREADY divided by the noise ceiling. "raw" must mean
est_diff * nc, not est_diff.
"""
import glob
import json
import numpy as np
from scipy import stats as sps

D = ("/home/byrong/projects/TSN/Magno-Parvo-Konio-Net/.claude/worktrees/"
     "mpk-dev/results/vp_recovery_realnc")

ds = [json.load(open(f)) for f in sorted(glob.glob(f"{D}/*.json"))]
print(f"{len(ds)} subjects, arm(s) {sorted({d['model'] for d in ds})}\n")

rows = [r for d in ds for r in d["conditions"]["B_manufacture"]["rows"]]
nc = np.array([r["noise_ceiling"] for r in rows])
um = np.array([r["est_unique_m"] for r in rows])
up = np.array([r["est_unique_p"] for r in rows])
dif = np.array([r["est_diff"] for r in rows])
assert np.allclose(dif, um - up), "est_diff is not est_unique_m - est_unique_p"
assert max(abs(r["true_diff"]) for r in rows) < 1e-9, "not a zero null"

cands = {
    "stored est_diff (already /nc)": dif,
    "est_diff * nc  (undo the /nc)": dif * nc,
    "est_diff / nc  (divide again)": dif / nc,
    "(um-up)/(um+up)  ratio index": (um - up) / (um + up),
    "(um-up)/mean(um,up)": (um - up) / ((um + up) / 2),
}
print(f"{'definition':34s} {'mean':>9s} {'median':>9s}")
for k, v in cands.items():
    print(f"{k:34s} {v.mean():+9.4f} {np.median(v):+9.4f}")

print("\ntargets from b5faaee: +0.126 'normalised', +0.025 'raw'")

# per-ROI aggregation and the rank correlation, for each definition
print(f"\n{'definition':34s} {'rho':>7s} {'p':>7s}  per-ROI means")
nroi = len(ds[0]["conditions"]["B_manufacture"]["rows"])
rank = np.arange(nroi)
for k, v in cands.items():
    per = np.array([v[np.arange(len(rows)) % nroi == i].mean()
                    for i in range(nroi)])
    rr = sps.spearmanr(rank, per)
    print(f"{k:34s} {rr.statistic:+7.3f} {rr.pvalue:7.3f}  "
          f"{np.round(per, 4).tolist()}")

# 8/8 sign consistency, on the stored value
print("\nsign consistency across subjects (stored est_diff, ROI mean):")
per_subj = np.array([np.mean([r["est_diff"] for r in
                              d["conditions"]["B_manufacture"]["rows"]])
                     for d in ds])
print(f"  {np.round(per_subj, 4).tolist()}")
print(f"  positive in {int((per_subj > 0).sum())}/{len(per_subj)} subjects")
w = sps.wilcoxon(per_subj)
print(f"  Wilcoxon vs 0: p = {w.pvalue:.4f} (exact floor at n=8 is 0.0078)")

# and the per-subject rank correlation, averaged -- a different rho estimator
rhos = []
for d in ds:
    v = [r["est_diff"] for r in d["conditions"]["B_manufacture"]["rows"]]
    rhos.append(sps.spearmanr(np.arange(len(v)), v).statistic)
rhos = np.array(rhos)
print(f"\nper-subject rho(est_diff, rank): mean {rhos.mean():+.3f}, "
      f"median {np.median(rhos):+.3f}, positive in "
      f"{int((rhos > 0).sum())}/{len(rhos)}")
