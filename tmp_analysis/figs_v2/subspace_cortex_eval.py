"""Evaluate the pre-registered cortex predictions P-C1-P-C3 of PLAN_SUBSPACE D4.

  PYTHONPATH=. python tmp_analysis/figs_v2/generality_cortex.py --run-id 2026-09-15_sub --out results/figures_v2/subspace_cortex.json
  PYTHONPATH=. python tmp_analysis/figs_v2/subspace_cortex_eval.py

Reads results/figures_v2/subspace_cortex.json (three-band fits: index (U_S - U_T)/(U_S + U_T)
per region, w* of the 15-d lossy sum) and prints the per-family contrasts with the thresholds.
"""
from __future__ import annotations

import json

import numpy as np

REGIONS = ["V1", "V2", "V3", "hV4", "ventral", "lateral", "dorsal"]
FAMS = ["rgb_tsn", "single34", "resnet50", "vit_b_16", "convnext_base"]


def main():
    ctx = json.load(open("results/figures_v2/subspace_cortex.json"))["families"]
    print(f"{'family':14s} " + " ".join(f"{r:>8s}" for r in REGIONS) + "   extrastriate-V1 (n neg/8, p)   hV4-V1 (n neg, p)   w*: V1  extrastr.  (n lower/8, p)")
    ev = {}
    for fam in FAMS:
        if fam not in ctx:
            print(f"{fam:14s} (not fitted yet)"); continue
        fo = ctx[fam]; st = fo["stats"]
        idx = " ".join(f"{np.nanmean(fo['index'][r]):+8.2f}" for r in REGIONS)
        e = st["extrastriate_minus_V1"]; h = st["hV4_minus_V1"]
        w = st.get("wstar_extrastriate_minus_V1"); wm = st.get("wstar_median", {})
        wtxt = f"{wm.get('V1', float('nan')):.2f}  {np.nanmean([wm.get(r, np.nan) for r in ('ventral', 'lateral', 'dorsal')]):.2f}  ({w['n_neg']}/{w['n']}, {w['p']:.4f})" if w else "—"
        print(f"{fam:14s} {idx}   {e['mean']:+.3f} ({e['n_neg']}/{e['n']}, {e['p']:.4f})   {h['mean']:+.3f} ({h['n_neg']}/{h['n']}, {h['p']:.4f})   {wtxt}")
        ev[fam] = dict(ext_neg=e["n_neg"], ext_p=e["p"], wstar_neg=(w["n_neg"] if w else None))
    print("\nPre-registered predictions:")
    if all(f in ev for f in ("rgb_tsn", "single34", "resnet50")):
        c1 = {f: ev[f]["ext_neg"] for f in ("rgb_tsn", "single34", "resnet50")}
        print(f"  P-C1 extrastriate-V1 negative in >= 7/8 for rgb_tsn, single34 AND resnet50: {c1}  met={all(v >= 7 for v in c1.values())}")
    c2 = {f: ev[f]["ext_neg"] for f in ("vit_b_16", "convnext_base") if f in ev}
    if c2:
        print(f"  P-C2 >= 7/8 in at least one of ViT-B/16, ConvNeXt-B: {c2}  met={any(v >= 7 for v in c2.values())}  (n families fitted {len(c2)}/2)")
    c3 = {f: ev[f]["wstar_neg"] for f in ev if ev[f]["wstar_neg"] is not None}
    if c3:
        n_ok = sum(v >= 7 for v in c3.values())
        print(f"  P-C3 w* lower in extrastriate than V1 in >= 7/8, in >= 3 of 5 families: {c3}  -> {n_ok} families  met={n_ok >= 3}  (fitted {len(c3)}/5)")


if __name__ == "__main__":
    main()
