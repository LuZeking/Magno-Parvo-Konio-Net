"""The accuracy-matched control on the exemplar-held-out probe (docs/PLAN_EXEMPLAR_2026-09-21.md, addendum of
2026-09-22, array B'; descriptive). Reads the six per-checkpoint files of slurm/exemplar_streams_matched.sh and,
for the no-diet final checkpoints, results/figures_v2/exemplar_streams.json; writes
results/figures_v2/exemplar_matched.json with every probe per stream and the share of the magno-stream gap
between the diet (best saved checkpoint) and the no-diet network (final checkpoint) that the matched network
closes, as for the exemplar-shared probe of Fig. 2e: (matched - no diet) / (diet - no diet), seed means, and per
matched seed against the two arm means.

  python tmp_analysis/figs_v2/exemplar_matched.py
"""
from __future__ import annotations

import json
import os

import numpy as np

SRC = "/scratch/byrong/mpk/exemplar/results/2026-09-22_streams_matched"
OUT = "results/figures_v2/exemplar_matched.json"
DIET = ["dkl_bcs_seed0_ep125", "dkl_bcs_seed1_ep075", "dkl_bcs_seed2_ep050"]
MATCHED = ["dkl_base_matched_seed0_ep010", "dkl_base_matched_seed1_ep010", "dkl_base_matched_seed2_ep011"]
NODIET = ["dkl_base_seed0_trained", "dkl_base_seed1_trained", "dkl_base_seed2_trained"]


def main():
    streams = {n: json.load(open(os.path.join(SRC, f"{n}_stream_exemplar.json")))["streams"] for n in DIET + MATCHED}
    ex = json.load(open("results/figures_v2/exemplar_streams.json"))["streams"]
    streams.update({n: ex[n] for n in NODIET})
    out = dict(plan="docs/PLAN_EXEMPLAR_2026-09-21.md (addendum 2026-09-22, array B')", source=SRC,
               arms=dict(diet=DIET, no_diet=NODIET, matched=MATCHED), streams=streams, gap={})
    for est in ("Q0", "Q1", "Q2", "S64", "S256n"):
        v = {arm: np.array([streams[n]["magno"][est]["bias"] for n in names]) for arm, names in out["arms"].items()}
        d, nd, m = v["diet"].mean(), v["no_diet"].mean(), v["matched"].mean()
        out["gap"][est] = dict(diet=v["diet"].tolist(), no_diet=v["no_diet"].tolist(), matched=v["matched"].tolist(),
                               gap=float(d - nd), closed=float((m - nd) / (d - nd)),
                               closed_per_seed=((v["matched"] - nd) / (d - nd)).tolist())
        g = out["gap"][est]
        print(f"  {est:5s} diet {d:+.3f}  no diet {nd:+.3f}  matched {m:+.3f}  gap {g['gap']:+.3f}  "
              f"closes {100 * g['closed']:.1f}% (seeds {100 * min(g['closed_per_seed']):.1f}-{100 * max(g['closed_per_seed']):.1f}%)")
    json.dump(out, open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
