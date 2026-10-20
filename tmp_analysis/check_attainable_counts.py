import glob, json
import numpy as np

diet = ["dkl_bcs_seed0_best", "dkl_bcs_seed1_best", "dkl_bcs_seed2_best",
        "dkl_bcs_seed0_final", "dkl_bcs_seed1_final", "dkl_bcs_seed2_final"]
runs = {}
for p in glob.glob("results/fit_w_observers/dkl_*_wfit.json"):   # the BA runs only; fs2_* fits live beside them
    d = json.load(open(p)); runs[d["model_name"]] = d
names = sorted(next(iter(runs.values()))["observers"])

print("bootstrap-attainable fraction, DIET arm (6 runs), full precision:")
zero, nonzero = [], []
for h in names:
    v = [runs[m]["observers"][h]["rate_fit"]["boot_frac_attainable"] for m in diet]
    mean = float(np.mean(v))
    tag = "ZERO" if all(x == 0.0 for x in v) else "nonzero"
    print(f"  {h}  mean {mean:.6f}  per-run {['%.4f' % x for x in v]}  {tag}")
    (zero if all(x == 0.0 for x in v) else nonzero).append(h)

print(f"\nexactly zero in EVERY diet run: {len(zero)}  {zero}")
print(f"nonzero somewhere:              {len(nonzero)}  {nonzero}")

# and over all nine runs, for completeness
allr = sorted(runs)
z2 = [h for h in names
      if all(runs[m]["observers"][h]["rate_fit"]["boot_frac_attainable"] == 0.0
             for m in allr)]
print(f"exactly zero in all NINE runs:  {len(z2)}  {z2}")
