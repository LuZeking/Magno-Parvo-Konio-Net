"""Render results/finzi_1to1/finzi_1to1_stats.json as a markdown table set.

    python -m src.analysis.finzi_1to1.report --dir results/finzi_1to1
"""

from __future__ import annotations

import argparse
import json
import os


def fmt(d: dict) -> str:
    if d.get("label") == "MISSING":
        return "| missing | | | | |"
    lo, hi = d["ci90"]
    return (f"| {d['mean']:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {d['wilcoxon_p']:.4f} | "
            f"{d['n_positive']}/{d['n']} | {d.get('gate', d['label'])} |")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="results/finzi_1to1")
    args = p.parse_args()
    R = json.load(open(os.path.join(args.dir, "finzi_1to1_stats.json")))
    L = ["# Finzi 1-to-1 stream-preference test: results", "",
         "Plan: `docs/PLAN_FINZI_1TO1_2026-09-24.md` (with amendment 1). "
         f"δ = {R['delta']}. Values are per-subject means (n = 8); 90 % t-interval; exact "
         "two-sided Wilcoxon.", "", f"**Verdict: {R['verdict']}**", "",
         "## Gates", "", "| gate | mean Δ | 90 % CI | p | n>0 | outcome |", "|---|---|---|---|---|---|"]
    for k, v in R["gates"].items():
        L.append(f"| {k} " + fmt(v))
    L += ["", "## Primary", "", "| test | mean Δ | 90 % CI | p | n>0 | label |",
          "|---|---|---|---|---|---|"]
    for k, v in R["primary"].items():
        L.append(f"| {k} " + fmt(v))
    L += ["", "Δ_DV = share of dorsal voxels won by magno units − share of ventral voxels won by "
          "magno units (NC-quintile stratified). POSITIVE = magno → dorsal.", "",
          "## Secondary (labels do not change the primary)", "",
          "| test | mean Δ | 90 % CI | p | n>0 | label |", "|---|---|---|---|---|---|"]
    for k, v in R["secondary"].items():
        L.append(f"| {k} " + fmt(v))
    L += ["", "## Descriptive", "",
          "Magno share per stream and NC-normalised test r (r/√NC) of assigned pairs, "
          "fixed-quota draws (unforced variant in brackets).", "",
          "| arm | share ventral | share lateral | share dorsal | r/√NC ventral | lateral | dorsal |",
          "|---|---|---|---|---|---|---|"]
    for arm in ("diet", "nodiet", "swap", "untrained_diet", "untrained_nodiet"):
        a, u = R["descriptive"][arm], R["descriptive"][arm + " (unforced)"]
        L.append(
            f"| {arm} | {a['share_ventral']:.3f} ({u['share_ventral']:.3f}) | "
            f"{a['share_lateral']:.3f} ({u['share_lateral']:.3f}) | "
            f"{a['share_dorsal']:.3f} ({u['share_dorsal']:.3f}) | {a['fr_ventral']:.3f} | "
            f"{a['fr_lateral']:.3f} | {a['fr_dorsal']:.3f} |")
    L += ["", "TDANN with Finzi's own design (all units, rectangular assignment), r/√NC; "
          "published (fsaverage, α = 0.25): dorsal 0.22, lateral 0.25, ventral 0.34.", "",
          "| network | ventral | lateral | dorsal |", "|---|---|---|---|"]
    for k, v in R["descriptive"].items():
        if "Finzi" in k:
            L.append(f"| {k} | {v['fr_ventral']:.3f} | {v['fr_lateral']:.3f} | {v['fr_dorsal']:.3f} |")
    out = os.path.join(args.dir, "REPORT.md")
    with open(out, "w") as f:
        f.write("\n".join(L) + "\n")
    print(out)


if __name__ == "__main__":
    main()
