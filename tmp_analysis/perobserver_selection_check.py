"""How much of a per-observer error-consistency maximum is selection?

§2.6.2 reports kappa averaged over the ten observers, maximised over `w`
(7.6-20.7% of the pooled human-human ceiling). A per-observer statistic --
argmax over `w` for each observer, then the best of the ten -- is selected
twice and is not comparable to it.

This reads the ALREADY-STORED kappa_per_human at 21 weights for 9 runs and
recomputes the doubly-selected statistic on that coarse grid, so the
contribution of grid resolution can be separated from the contribution of
selection at zero compute cost. Read-only; writes nothing.

Context: PLAN §106.7 predicted a fitted-w kappa under 25% of ceiling and
observed 38.2% for subject-03 on dkl_bcs_seed0_best over a 1001-point grid.
That value reproduces exactly here on 21 points, and is the SMALLEST of the
nine runs (family spans 38.2-61.1%), so the finer grid contributes nothing and
the prediction was comparing two different statistics.
"""
import glob
import json

import numpy as np

SRC = "results/sweep_error_consistency/*.json"


def per_run(path):
    d = json.load(open(path))
    ceil = d["human_human_ceiling"]
    kph = [p["kappa_per_human"] for p in d["sweep"]]
    names = list(kph[0]) if isinstance(kph[0], dict) else None
    A = np.array([[(v if v is not None else np.nan)
                   for v in (k.values() if isinstance(k, dict) else k)]
                  for k in kph], dtype=float)                  # 21 x n_obs
    obs_max = np.nanmax(A, axis=0)          # argmax over w, per observer
    return dict(
        run=path.rsplit("/", 1)[-1].replace("_kappa_sweep.json", ""),
        published=100 * np.nanmax(np.nanmean(A, axis=1)) / ceil,
        best_of_n=100 * np.nanmax(obs_max) / ceil,
        median_obs=100 * np.nanmedian(obs_max) / ceil,
        who=(names[int(np.nanargmax(obs_max))] if names else
             f"obs{int(np.nanargmax(obs_max))}"),
    )


def main():
    rows = [per_run(f) for f in sorted(glob.glob(SRC))]
    if not rows:
        raise SystemExit(f"no results at {SRC}")
    print(f"{'run':40s}{'published':>11s}{'best-of-n':>11s}"
          f"{'median-obs':>12s}  argmax observer")
    for r in rows:
        print(f"{r['run']:40s}{r['published']:10.1f}%{r['best_of_n']:10.1f}%"
              f"{r['median_obs']:11.1f}%  {r['who']}")
    p = [r["published"] for r in rows]
    b = [r["best_of_n"] for r in rows]
    print(f"\nacross {len(rows)} runs on the 21-point grid, pooled ceiling:")
    print(f"  mean over observers, max over w : {min(p):.1f}-{max(p):.1f}%")
    print(f"  per-observer argmax, best of n  : {min(b):.1f}-{max(b):.1f}%")
    print("\nA permutation null for the second statistic must reproduce BOTH\n"
          "selection steps -- recompute every w, take the per-observer argmax,\n"
          "then the best of n -- or it bounds neither.")


if __name__ == "__main__":
    main()
