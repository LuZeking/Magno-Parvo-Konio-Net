# Figure 6 — caption (skeleton; ⟦PENDING⟧ the runs of `slurm/ctd_encode.sh`)

**Predicted cortical representations of the cue-conflict images: texture-biased, yet carrying the evidence the observers' decisions follow.** The 1,200 cue-conflict images were pushed through each subject's fitted encoding model (Methods 4.8) and each region's predicted pattern was read against sixteen category templates (predicted patterns of natural MiniEcoset photographs of the same categories). Registered design: `docs/PLAN_CORTEX_DECISION_2026-10-26.md`.

**(a)** Cortical shape-bias rate per region — the fraction of images whose predicted pattern correlates more with the shape category's template than with the texture category's — for the model, its control and the channel swap (seeds averaged per subject; points subjects, bars medians), with the untrained network and the input-level (oriented-energy) floor beside them. Reference lines: the model's behavioural rate (0.68) and the ten observers (0.96). ⟦values⟧

**(b)** Correspondence with the observers per region, as a percentage of each target's split-half ceiling: the shape margin against the per-image shape rate (.624 residual ceiling) and the cue margin against the per-image resolution rate (.728), both sides residualised on shape- and texture-category main effects. Grey: the 97.5th percentile of the within-cell permutation null; open symbols: the input floor. ⟦values⟧

**(c)** The same voxels under the magno-only, parvo-only and joint estimators of the same fit: the rate of (a) per band and region in the model. ⟦values⟧

**(d)** One region (⟦which⟧): residualised shape margin against the observers' residualised shape rate, one point per image, with the exemplar-cluster bootstrap interval of r. ⟦values⟧

Sources (pending): `results/cortex_decision/ctd_analysis.json` (from `/scratch/byrong/mpk/encoding/results/2026-10-26_ctd/analysis/`); figure script to be written as `tmp_analysis/figs_v2/fig6_cortex_decision.py`. The former Fig. 6 (frozen single-trunk networks, the model zoo) is no longer part of the manuscript; its files `fig6_generality.*` stay for provenance.
