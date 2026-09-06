"""What does `±` actually denote in PAPER_DRAFT.md? Resolved by recomputation.

The NHB v2 referee flagged undefined `±` as a *fatal* statistical item, and it
is one of the two overrides holding the paper at Major revision. Closing it
means writing one sentence in §4.7 -- but the sentence has to be true, and the
only way to know which estimator produced a printed value is to recompute every
candidate and see which one lands.

A first attempt at that sentence defined `±` as "the range across the three
training seeds (max - min)". This module exists because that is false: the
range is 0.0458 where the draft prints 0.025. Shipping a *definition* that
misdescribes the numbers is strictly worse than shipping none, because it
converts a presentational gap into a factual error -- the §50.6 failure mode.

So: for each printed value, every plausible estimator is computed and matched
against what the manuscript prints, at the precision the manuscript prints it.
A cell that matches no candidate, or more than one, is reported as UNRESOLVED
rather than assigned to the nearest guess.

    PYTHONPATH=. python tmp_analysis/uncertainty_notation.py
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re

import numpy as np

CC = "results/cue_conflict"
HUMAN_DIR = "/scratch/byrong/VST_data/geirhos_human_data/cue_conflict"
PAIR = re.compile(r"_([a-z]+)\d+-([a-z]+)\d+\.png$")


def candidates(v: np.ndarray) -> dict[str, float]:
    """Every estimator someone could plausibly have written as `±`."""
    n = len(v)
    return {
        "SD (ddof=1)": float(v.std(ddof=1)),
        "SD (ddof=0)": float(v.std(ddof=0)),
        "SEM": float(v.std(ddof=1) / np.sqrt(n)),
        "range (max-min)": float(v.max() - v.min()),
        "half-range": float((v.max() - v.min()) / 2),
    }


def resolve(label: str, v: np.ndarray, printed: float, unit: str) -> set[str]:
    """Which estimators are consistent with this printed cell? Never guess.

    Returns the SET that matches at the printed precision. A cell printed to
    two decimals cannot discriminate estimators that differ in the third, so
    it constrains rather than identifies -- the identification comes from
    intersecting the sets across cells.
    """
    nd = len(str(printed).split(".")[1])
    cand = candidates(v)
    hits = {k for k, x in cand.items() if round(x, nd) == printed}

    print(f"\n{label}")
    print(f"  n={len(v)} over {unit}; values "
          f"{[round(float(x), 4) for x in v]}")
    print(f"  mean {v.mean():.4f}   manuscript prints +-{printed} "
          f"({nd} dp)")
    for k, x in cand.items():
        mark = "  <== consistent" if k in hits else ""
        print(f"    {k:18s} {x:.4f}{mark}")

    if len(hits) == 1:
        print(f"  -> IDENTIFIED: {next(iter(hits))}")
    else:
        print(f"  -> {len(hits)} consistent at this precision; constrains only")
    return hits


def seed_shape_bias(cond: str) -> np.ndarray:
    return np.array([
        json.load(open(f"{CC}/{cond}_seed{s}.json"))
        ["conditions"]["full"]["shape_bias"] for s in (0, 1, 2)
    ])


def human_shape_bias() -> np.ndarray:
    """Per-observer shape bias, re-derived from Geirhos's raw CSVs.

    geirhos_humans.py runs its whole report at import time (no __main__
    guard), so its loader cannot be imported -- the logic is repeated here
    and checked against its published output instead.
    """
    out = []
    for f in sorted(glob.glob(os.path.join(HUMAN_DIR, "*subject-*.csv"))):
        sh = tx = 0
        for row in csv.DictReader(open(f, newline="")):
            m = PAIR.search(row["imagename"].strip())
            if not m or m.group(1) == m.group(2):
                continue
            resp = row["object_response"].strip()
            if resp == m.group(1):
                sh += 1
            elif resp == m.group(2):
                tx += 1
        out.append(sh / (sh + tx))
    return np.array(out)


def main() -> int:
    print("=" * 78)
    print("WHAT `±` DENOTES IN PAPER_DRAFT.md -- resolved by recomputation")
    print("=" * 78)

    verdicts = {}

    # -- the §2.3 behavioural headline, the value the referee singled out ----
    verdicts["diet integrated shape bias"] = resolve(
        "§2.3  diet, integrated readout: `0.675 ±0.025`",
        seed_shape_bias("dkl_bcs"), 0.025, "3 training seeds")
    verdicts["no-diet integrated shape bias"] = resolve(
        "§2.3  no diet, integrated readout: `0.416 ±0.020`",
        seed_shape_bias("dkl_base"), 0.020, "3 training seeds")

    # -- the human reference on the same measure -----------------------------
    h = human_shape_bias()
    # Validate the re-derivation before trusting it (geirhos_humans.py prints
    # 0.9576 +- 0.0191, n=10, range 0.919-0.987).
    assert len(h) == 10, f"expected 10 observers, got {len(h)}"
    assert abs(h.mean() - 0.9576) < 5e-4, h.mean()
    assert abs(h.std(ddof=1) - 0.0191) < 5e-4, h.std(ddof=1)
    print("\n[check] re-derived human observers reproduce geirhos_humans.py: "
          f"mean {h.mean():.4f}, SD {h.std(ddof=1):.4f}, n={len(h)}")
    verdicts["human observers"] = resolve(
        "§2.3  ten human observers: `0.958 ±0.019`", h, 0.019, "10 observers")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    for k, v in verdicts.items():
        shown = next(iter(v)) if len(v) == 1 else f"{len(v)} consistent"
        print(f"  {k:34s} {shown}")

    # The manuscript uses ONE notation, so the estimator must be consistent
    # with every cell at once. Intersecting is what identifies it: the
    # low-precision cell alone cannot, and neither can any cell in isolation
    # settle a manuscript-wide definition.
    surviving = set.intersection(*verdicts.values())
    print(f"\n  Consistent with ALL {len(verdicts)} cells simultaneously: "
          f"{sorted(surviving) if surviving else 'NOTHING'}")

    if surviving == {"SD (ddof=1)"}:
        print("\n  IDENTIFIED: `±` is the sample SD (ddof=1) about the mean,\n"
              "  over the unit named in each case. It is the only estimator\n"
              "  consistent with every printed cell.")
        print("\n  `range (max-min)` is EXCLUDED by every cell independently:\n"
              "    diet     range 0.0458, half-range 0.0229  vs printed 0.025\n"
              "    no diet  range 0.0384, half-range 0.0192  vs printed 0.020\n"
              "    humans   range 0.0675, half-range 0.0338  vs printed 0.019\n"
              "  The humans cell is decisive on its own: the range is 3.6x\n"
              "  the printed value, and the draft prints that range separately\n"
              "  in the same sentence as 0.919-0.987.")
        rc = 0
    else:
        print(f"\n  NOT IDENTIFIED -- do not write a definition from this.")
        rc = 1

    print("\n  NOT covered here: the §2.4/§2.5 PR and SPI tables carry `±`\n"
          "  across 8 NSD subjects and come from tmp_analysis/pr_table.py:41,\n"
          "  which uses numpy's default `r.std()` -- ddof=0. That is a\n"
          "  DIFFERENT estimator from the cells above; see PLAN §79.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
