"""Mean gain curves of the twelve trunk runs under E0 (pair-stratified halves) and E1 (exemplar-disjoint folds),
for Fig. 6a (docs/PLAN_EXEMPLAR_2026-09-21.md). exemplar_aggregate.py keeps only the trunks' spans; the figure
needs the curves. Reads the per-run <tag>_exemplar.json; writes results/figures_v2/exemplar_trunk_curves.json.

  python tmp_analysis/figs_v2/exemplar_trunk_curves.py
"""
from __future__ import annotations

import json
import os

SRC = "/scratch/byrong/mpk/exemplar/results/2026-09-21_exemplar"
OUT = "results/figures_v2/exemplar_trunk_curves.json"
TAGS = {"concat": "concat_dkl_bcs_seed{}", "single34": "run_dkl_bcs_single34_seed{}",
        "single18": "run_dkl_bcs_single_seed{}", "rgb_tsn": "run_rgb_tsn_seed{}"}
KEEP = ("mean_curve", "span_mean", "gated_span", "shape_end", "texture_end", "rho_mean_curve")


def main():
    out = dict(plan="docs/PLAN_EXEMPLAR_2026-09-21.md", source=SRC, families={})
    for key, pat in TAGS.items():
        seeds = []
        for s in range(3):
            d = json.load(open(os.path.join(SRC, pat.format(s) + "_exemplar.json")))
            seeds.append(dict(tag=d["tag"], gate=d["gate"], trained=d["trained"],
                              **{e: {k: d["spans"][e]["primary"][k] for k in KEEP} for e in ("E0", "E1")}))
        out["families"][key] = seeds
        print(key, " ".join(f"E1 gated {x['E1']['gated_span']:.3f} (E0 {x['E0']['gated_span']:.3f})" for x in seeds))
    json.dump(out, open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
