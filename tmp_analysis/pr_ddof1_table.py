"""The §2.4/§2.5 PR and SPI tables, restated at ddof=1.

§79.5 established that these tables print `r.std()` -- numpy's default ddof=0 --
over 8 NSD subjects, while every behavioural and human `±` in the manuscript is
ddof=1. §4.7 now discloses that split and commits to unifying at ddof=1; this
module produces the replacement numbers.

Two things here are load-bearing.

**The mapping.** A printed cell has to be tied to a feature-cache entry, and the
"after training" column mixes final checkpoints with best-saved ones (§71.8), so
`dkl_bcs_seed0` in that column is the ep125 entry, not `_trained`. Rather than
trust the mapping, every cell's currently-printed dispersion is re-derived and
checked at the printed precision; a mismatch fails the run.

**Which cells are already ddof=1.** That check immediately found two: the two
best-saved cells re-scored under §71.8 (`dkl_bcs_seed0` ep125 and
`dkl_bcs_swap_seed0` ep100) print ddof=1 already, because the re-scoring went
through a different script from the original table -- `pr_seeds.py:118` uses
`std(ddof=1)` where `pr_table.py:41` uses the numpy default. Their means
reproduce to four significant figures while their dispersions do not, which is
what identifies the cause as the estimator rather than the mapping.

So the manuscript carries THREE groups, not two, and one table column is
internally mixed. Applying a blanket "recompute every PR cell at ddof=1" would
double-correct those two cells (0.059 -> 0.063, 0.085 -> 0.091) and quietly
inflate two numbers that were already right. Each cell therefore declares the
estimator it is currently printed at, and that declaration is verified.

    PYTHONPATH=. python tmp_analysis/pr_ddof1_table.py
"""
from __future__ import annotations

import glob

import numpy as np

CACHE = "/scratch/byrong/VST_data/mpk_feature_cache"
NIMG = 1000

# (section, row, column, cache entry, printed mean, printed sd, current
#  estimator, decimal places)
CELLS = [
    # -- §2.4, draft ~lines 557-561 -------------------------------------------
    ("§2.4", "dkl_bcs_seed0", "front end",
     "untrained_seed0", 1.402, 0.089, "sd0", 3),
    # best saved (ep125) under §71.8 -- re-scored, already ddof=1
    ("§2.4", "dkl_bcs_seed0", "after training",
     "dkl_bcs_seed0_ep125_trained", 1.703, 0.059, "sd1", 3),
    ("§2.4", "dkl_bcs_parvo_highsf_seed0", "front end",
     "untrained_parvo_highsf", 0.586, 0.013, "sd0", 3),
    ("§2.4", "dkl_bcs_parvo_highsf_seed0", "after training",
     "dkl_bcs_parvo_highsf_seed0_trained", 1.640, 0.030, "sd0", 3),
    ("§2.4", "dkl_bcs_swap_seed0", "front end",
     "untrained_swap", 1.909, 0.125, "sd0", 3),
    # best saved (ep100) under §71.8 -- re-scored, already ddof=1
    ("§2.4", "dkl_bcs_swap_seed0", "after training",
     "dkl_bcs_swap_seed0_ep100_trained", 2.327, 0.085, "sd1", 3),
    ("§2.4", "dkl_base_seed0", "front end",
     "untrained_seed0_noblur", 1.634, 0.118, "sd0", 3),
    ("§2.4", "dkl_base_seed0", "after training",
     "dkl_base_seed0_trained", 0.947, 0.035, "sd0", 3),
    ("§2.4", "dkl_base_seed1", "after training",
     "dkl_base_seed1_trained", 0.988, 0.040, "sd0", 3),
    # -- §2.5, draft ~lines 685-688 (printed to 2 dp) -------------------------
    ("§2.5", "parvo_highsf", "Gabor bank",
     "gabor_parvo_highsf", 0.56, 0.02, "sd0", 2),
    ("§2.5", "parvo_highsf", "untrained net",
     "untrained_parvo_highsf", 0.59, 0.01, "sd0", 2),
    ("§2.5", "blur_cs (diet)", "Gabor bank",
     "gabor_blur_cs", 1.38, 0.09, "sd0", 2),
    ("§2.5", "blur_cs (diet)", "untrained net",
     "untrained_seed0", 1.40, 0.09, "sd0", 2),
    ("§2.5", "noblur (no diet)", "Gabor bank",
     "gabor_noblur", 2.23, 0.13, "sd0", 2),
    ("§2.5", "noblur (no diet)", "untrained net",
     "untrained_seed0_noblur", 1.63, 0.12, "sd0", 2),
    ("§2.5", "swap", "Gabor bank", "gabor_swap", 3.37, 0.16, "sd0", 2),
    ("§2.5", "swap", "untrained net", "untrained_swap", 1.91, 0.12, "sd0", 2),
]


def pr(X: np.ndarray) -> float:
    X = np.asarray(X[:NIMG], dtype=np.float64)
    X -= X.mean(0, keepdims=True)
    lam = np.linalg.eigvalsh(X @ X.T)
    lam = lam[lam > 0]
    return float((lam.sum() ** 2) / (lam ** 2).sum())


def ratios(entry: str) -> np.ndarray:
    fs = sorted(glob.glob(f"{CACHE}/{entry}_subj*.npz"))
    if not fs:
        raise SystemExit(f"no cache for {entry}")
    out = []
    for f in fs:
        z = np.load(f)
        out.append(pr(z["magno"]) / pr(z["parvo"]))
    return np.array(out)


def main() -> int:
    print("=" * 94)
    print("§2.4 / §2.5 PR AND SPI TABLES RESTATED AT ddof=1")
    print("=" * 94)
    print("\nEach cell declares the estimator it is CURRENTLY printed at. Both")
    print("the mean and that dispersion are re-derived and checked; a mismatch")
    print("fails the run rather than emitting a replacement for a wrong model.\n")

    cache: dict[str, np.ndarray] = {}
    rows, bad = [], []
    for sec, label, col, entry, p_mean, p_sd, cur, dp in CELLS:
        if entry not in cache:
            cache[entry] = ratios(entry)
        r = cache[entry]
        mean, sd0, sd1 = r.mean(), r.std(ddof=0), r.std(ddof=1)
        have = sd0 if cur == "sd0" else sd1

        ok = (f"{mean:.{dp}f}" == f"{p_mean:.{dp}f}"
              and f"{have:.{dp}f}" == f"{p_sd:.{dp}f}")
        if not ok:
            bad.append((sec, label, col, entry, p_mean, p_sd, cur, mean,
                        sd0, sd1))
        rows.append((sec, label, col, len(r), mean, sd0, sd1, cur, dp))
        print(f"  {sec} {label:34s} {col:14s} n={len(r)}  "
              f"mean {mean:.{dp}f}/{p_mean:.{dp}f}  "
              f"{cur} {have:.{dp}f}/{p_sd:.{dp}f}  "
              f"{'ok' if ok else 'MISMATCH'}")

    if bad:
        print(f"\n  {len(bad)} cell(s) did not reproduce.")
        for b in bad:
            print(f"    {b}")
        return 1

    n_sd1 = sum(1 for r in rows if r[7] == "sd1")
    print(f"\n  All {len(rows)} cells reproduce. {len(rows) - n_sd1} are printed"
          f" at ddof=0 and {n_sd1} already at ddof=1.")
    print("  The §2.4 'after training' column is therefore INTERNALLY MIXED:")
    print("  its two best-saved (§71.8 re-scored) cells are already ddof=1,")
    print("  its three final-checkpoint cells are ddof=0.")

    print("\n" + "=" * 94)
    print("REPLACEMENT VALUES — unify at ddof=1")
    print("=" * 94)
    print(f"  {'sec':6s} {'row':34s} {'column':14s} {'mean':>7s} "
          f"{'now':>8s} {'ddof=1':>8s}  change")
    for sec, label, col, n, mean, sd0, sd1, cur, dp in rows:
        now = sd0 if cur == "sd0" else sd1
        note = "unchanged (already ddof=1)" if cur == "sd1" else \
            f"{sd1 - sd0:+.{dp}f}"
        print(f"  {sec:6s} {label:34s} {col:14s} {mean:7.{dp}f} "
              f"{now:8.{dp}f} {sd1:8.{dp}f}  {note}")

    print("\n" + "=" * 94)
    print("MARKDOWN, ready to paste")
    print("=" * 94)
    for sec in ("§2.4", "§2.5"):
        print(f"\n{sec}:")
        for s, label, col, n, mean, sd0, sd1, cur, dp in rows:
            if s != sec:
                continue
            tag = "  [unchanged]" if cur == "sd1" else ""
            print(f"  {label} / {col}: {mean:.{dp}f} ±{sd1:.{dp}f}{tag}")

    changed = [r for r in rows if r[7] == "sd0"]
    print(f"\n  {len(changed)} cells change; {n_sd1} are already correct and")
    print("  MUST NOT be re-scaled (doing so would print 0.063 and 0.091).")
    print("  No mean moves. Dispersions rise by sqrt(8/7) = "
          f"{np.sqrt(8 / 7):.4f}, so no ordering, separation or claim in")
    print("  either table is affected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
