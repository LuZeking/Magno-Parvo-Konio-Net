# Extended Data Figure 4 — caption

**The gradient continues past hV4 and does not turn.** The stored banded-ridge fits of Fig. 3, relabelled with NSD's Kastner2015 atlas (the 22 ROIs that pass the ≥ 20-voxel rule), grouped as early (V1–hV4), ventral (VO1/2, PHC1/2), lateral (LO1/2, TO1/2) and dorsal (V3A/B, IPS0–2).

**(a)** Stream preference (U_m − U_p)/(U_m + U_p) per ROI: the untrained network on the model's test input, the control and the model (thick: mean over subjects of each subject's seed mean; thin: subjects), and the BA network's mean.

**(b)** The two components, noise-ceiling-normalised unique variance of the parvo band (U_p, solid) and the magno band (U_m, dashed), median over reliable voxels, for the model and its control. In the model U_p carries every region past V1 and U_m is near zero; in the control U_m carries every region and U_p is near zero.

**(c)** Regional contrasts per subject: each region's mean index minus the V1 mean, and region minus region. Relative to V1 the model's hV4 sits at −1.12 and every extrastriate group lower — ventral −1.26, lateral −1.27, dorsal −1.22 — each in 8/8 subjects (exact Wilcoxon p = .0078); dorsal − ventral and lateral − ventral do not depart from zero. ✱: exact Wilcoxon p < .05, n = 8.

Sources: `results/figures_v3/ext_gradient.json` (`tmp_analysis/figs_v2/ext_gradient.py --set v3`); `tmp_analysis/figs_v2/figE4_ext_gradient.py`.
