# Figure 3 — caption

**In human visual cortex the magno stream carries V1 and the texture-biased stream the rest of the hierarchy, and only with a maturing input.**

Voxel-wise banded-ridge encoding models of eight NSD subjects, one band per stream; each voxel's cross-validated explained variance is partitioned into the part only the magno band accounts for (U_m) and the part only the parvo band does (U_p) (Methods 4.5). Stream preference per ROI: median over reliable voxels (noise ceiling > 0.1, U_m + U_p > 10⁻³, ≥ 20 voxels) of (U_m − U_p)/(U_m + U_p).

**(a)** Stream-preference profile from V1v to hV4 (Wang et al. 2015 atlas); line: mean over the eight subjects of each subject's seed mean; shading: ± s.e.m. The model's profile crosses zero, from +0.55 in V1v (magno-explained) to −0.76 in hV4 (parvo-explained). The control, on the identical test input, is magno-explained everywhere and nearly flat (+0.83 in V1v, +0.76 in hV4), as is the BA no-diet network (+0.91, +0.83); the BA network is parvo-explained from V1 on (−0.00 to −0.84). Dashed: the untrained network on the model's test input.

**(b)** The gradient hV4 − mean(V1v, V1d) per subject. One untrained network on one test input already descends (−0.48); training on the maturing input ends at −1.27 (seeds −1.259 to −1.291), training on the adult input from the first step at −0.06 (control, one seed); model steeper than control in 8/8 subjects (exact Wilcoxon p = .0078). Right: the BA network (seeds −0.745 to −0.834).

**(c)** Spearman ρ between hierarchical rank and stream preference per subject (seed-mean profile), against the exact null over all 630 labellings of the seven ROIs (shaded; dashed: its 2.5th percentile). Per-seed means: −0.964 to −0.973 for the model, −0.29 for the control.

**(d)** The cue-conflict images pushed through the encoding models fitted on natural scenes give a predicted magno − parvo shape-bias difference per subject: +1.32 for the model (seeds +1.32 to +1.34), +0.08 for the control, +1.55 for the BA network (seeds +1.53 to +1.57).

**(e)** Noise-ceiling-normalised median r² over reliable nsdgeneral voxels: 0.471 for the model against 0.496 for its control (8/8 subjects) and 0.395 for the BA network (8/8).

Sources: `docs/slides_2026-10-12_ba_vs_fusion/figures/fs2_cortex_summary.json` (`tmp_analysis/fs2_cortex_summary.py`), the untrained network's fits (`/scratch/byrong/mpk/encoding/results/2026-10-16_fs2`); `tmp_analysis/figs_v2/fig3_cortex.py`.
