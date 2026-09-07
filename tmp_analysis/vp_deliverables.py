"""The three §82/§84 deliverables at 1000 components, plus the mechanism audit.

Two runs are compared:
  ASC   results/vp_recovery         nc = 0.15..0.45 ASCENDING (the §84 bug)
  REAL  results/vp_recovery_realnc  nc = measured, DESCENDING
Both at n_comp=1000. The 60-component pilot is not on disk here; where the
pilot is quoted the number comes from the §82 record and is labelled as such.
"""
import glob
import json
import numpy as np
from scipy import stats as sps

BASE = ("/home/byrong/projects/TSN/Magno-Parvo-Konio-Net/.claude/worktrees/"
        "mpk-dev/results")
RUNS = {"ASC": f"{BASE}/vp_recovery", "REAL": f"{BASE}/vp_recovery_realnc"}


def load(d):
    return [json.load(open(f)) for f in sorted(glob.glob(f"{d}/*.json"))]


runs = {k: load(v) for k, v in RUNS.items()}

print("=" * 66)
print("(3) IDENTIFIABILITY SPECTRUM at n_comp=1000")
print("=" * 66)
print(f"{'run':5s} {'n':>2s} {'sig_shared':>11s} {'sig_min':>10s} "
      f"{'sig_mean':>9s} {'sig_med':>8s}")
for k, ds in runs.items():
    sh = np.array([d["cca"]["sigma_shared"] for d in ds])
    mn = np.array([d["cca"]["sigma_unique_m"] for d in ds])
    me = np.array([d["cca"]["sigma_mean"] for d in ds])
    md = np.array([d["cca"]["sigma_median"] for d in ds])
    print(f"{k:5s} {len(ds):2d} {sh.mean():11.4f} {mn.mean():10.2e} "
          f"{me.mean():9.4f} {md.mean():8.4f}")
print("\npilot record (S82, n_comp=60): sig_min 0.0029, sig_max 0.813")

print("\n" + "=" * 66)
print("(1) BIAS MAGNITUDE, condition B (true difference is exactly 0)")
print("=" * 66)
for k, ds in runs.items():
    by = {}
    for d in ds:
        for r in d["conditions"]["B_manufacture"]["rows"]:
            by.setdefault(r["roi_index"], []).append(r)
    nc = np.array([by[i][0]["noise_ceiling"] for i in sorted(by)])
    norm = np.array([np.mean([r["est_diff"] for r in by[i]])
                     for i in sorted(by)])
    raw = norm * nc
    rank = np.arange(len(nc))
    p = np.polyfit(np.log(nc), np.log(np.abs(raw)), 1)[0]
    rn = sps.spearmanr(rank, norm)
    print(f"\n{k}: nc {nc.min():.3f}-{nc.max():.3f} "
          f"(factor {nc.max() / nc.min():.2f}), "
          f"rho(nc,rank)={sps.spearmanr(rank, nc).statistic:+.3f}")
    print(f"   normalised bias  mean {norm.mean():+.4f}  "
          f"spread {norm.max() - norm.min():.4f}  "
          f"rho vs rank {rn.statistic:+.3f} (p={rn.pvalue:.3f})")
    print(f"   raw bias         mean {raw.mean():+.4f}  "
          f"spread {raw.max() - raw.min():.4f}  "
          f"rho vs rank {sps.spearmanr(rank, raw).statistic:+.3f}")
    print(f"   contrast hV4-V1v (normalised) = {norm[-1] - norm[0]:+.4f}")
    print(f"   raw ~ nc^{p:+.2f}   corr(norm, 1/nc) = "
          f"{np.corrcoef(norm, 1.0 / nc)[0, 1]:+.3f}")

print("\n" + "=" * 66)
print("(2) DISCRETISATION CHECK -- are est_diff values quantised?")
print("=" * 66)
for k, ds in runs.items():
    e = np.array([r["est_diff"] for d in ds
                  for c in ("A_recovery", "B_manufacture")
                  for r in d["conditions"][c]["rows"]])
    u6 = np.unique(np.round(e, 6))
    u3 = np.unique(np.round(e, 3))
    gaps = np.diff(np.sort(e))
    print(f"{k}: n={e.size}  unique@6dp={u6.size}  unique@3dp={u3.size}  "
          f"min gap {gaps[gaps > 0].min():.2e}")

print("\n" + "=" * 66)
print("CONDITION A at 1000 components -- attenuation, not just rank")
print("=" * 66)
for k, ds in runs.items():
    t = np.array([[r["true_diff"]
                   for r in d["conditions"]["A_recovery"]["rows"]] for d in ds])
    e = np.array([[r["est_diff"]
                   for r in d["conditions"]["A_recovery"]["rows"]] for d in ds])
    rho = np.array([d["conditions"]["A_recovery"]["rho_est_vs_rank"]
                    for d in ds])
    slope = np.polyfit(t.ravel(), e.ravel(), 1)[0]
    print(f"{k}: rho(est,rank) mean {rho.mean():+.3f} "
          f"(min {rho.min():+.3f})")
    print(f"   true range {t.mean(0).min():+.4f}..{t.mean(0).max():+.4f}  "
          f"est range {e.mean(0).min():+.4f}..{e.mean(0).max():+.4f}")
    print(f"   est = {slope:.3f} x true  -> attenuation {1 / slope:.2f}x")
