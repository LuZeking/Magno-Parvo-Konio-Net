"""Extended Data Table 11 — stream preference beyond hV4 on every NSD atlas.
Reads results/figures_v2/ext_gradient.json (tmp_analysis/figs_v2/ext_gradient.py) and
appends the table to results/figures_v2/ed_tables.md (idempotent).
Usage: python tmp_analysis/figs_v2/ed_table11.py
"""
import json

import numpy as np

D = json.load(open("results/figures_v2/ext_gradient.json"))
ARMS = ["untrained", "no diet", "diet", "swap", "parvo_highsf", "thesis"]
HEAD = {"untrained": "untrained (diet front end)", "no diet": "no diet (3 seeds)", "diet": "diet (3 seeds)",
        "swap": "swap (seed 0)", "parvo_highsf": "parvo + high-SF (seed 0)", "thesis": "thesis ckpt (n = 1)"}
ORDER = [("Kastner2015", ["VO1", "VO2", "PHC1", "PHC2", "LO1", "LO2", "TO1", "TO2", "V3A", "V3B", "IPS0", "IPS1", "IPS2"]),
         ("streams", ["early", "midventral", "midlateral", "midparietal", "ventral", "lateral", "parietal"]),
         ("floc-faces", ["OFA", "FFA-1", "FFA-2"]), ("floc-places", ["OPA", "PPA"]),
         ("floc-bodies", ["EBA", "FBA-1", "FBA-2"]), ("floc-words", ["OWFA", "VWFA-1", "VWFA-2", "mfs-words"])]


def f(x):
    return "—" if x is None or not np.isfinite(x) else f"{x:+.2f}".replace("-", "−")


lines = ["**Extended Data Table 11 | Stream preference beyond hV4, on every NSD atlas.** Median over reliable voxels of the per-voxel index (Um − Up)/(Um + Up) (noise ceiling > 0.1, Um + Up > 10⁻³, ≥ 20 voxels per ROI per subject), mean over the eight subjects (n = subjects with enough voxels); the no-diet and diet arms are the mean over three training seeds per subject, whereas the swap, parvo + high-SF and thesis columns are single networks (seed 0, or the thesis's one checkpoint; Methods 4.5). Positive = magno band, negative = parvo band. Atlases: Wang et al. (2015) Kastner2015 ROIs beyond hV4 (the V1–hV4 rows are Extended Data Table 1 and Fig. 3), NSD's `streams` atlas, and the fLoc category-selective ROIs. Regional contrasts and the exact tests are in Extended Data Fig. 4c and `results/figures_v2/ext_gradient.md`. Source: the stored voxelwise fits in `results/encoding_cc/` and `results/encoding/`, relabelled; no refitting.", "",
         "| atlas | ROI | " + " | ".join(HEAD[a] for a in ARMS) + " | voxels (median, diet) |", "|---|---|" + "---|" * (len(ARMS) + 1)]
for atlas, rois in ORDER:
    for r in rois:
        cells = []
        for a in ARMS:
            t = D[atlas][a][r]
            cells.append("—" if t["mean"] is None else f"{f(t['mean'])} (n={t['n_subjects']})")
        lines.append(f"| {atlas} | {r} | " + " | ".join(cells) + f" | {D[atlas]['diet'][r]['median_vox']:.0f} |")
lines.append("")
md = open("results/figures_v2/ed_tables.md", encoding="utf8").read()
if "Extended Data Table 11" in md:
    a = md.index("**Extended Data Table 11")
    b = md.find("**Extended Data Table 12", a)
    md = md[:a] + (md[b:] if b >= 0 else "")
# insert before Table 12 if present, else append
if "**Extended Data Table 12" in md:
    b = md.index("**Extended Data Table 12")
    md = md[:b].rstrip() + "\n\n" + "\n".join(lines) + "\n\n" + md[b:]
else:
    md = md.rstrip() + "\n\n" + "\n".join(lines) + "\n"
open("results/figures_v2/ed_tables.md", "w", encoding="utf8").write(md)
print("\n".join(lines[:8]), "...")
