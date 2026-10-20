# Figure 5 — caption

**Three properties the field treats as travelling with shape bias do not follow from it.**

**(a)** Out-of-distribution robustness: relative-accuracy area under the curve (accuracy over clean accuracy, mean over severities) per corruption family, one marker per seed; the model against its control (final checkpoints, identical test input), the BA network's two arms as small squares (best saved checkpoints). The model is less robust than its control on uniform noise (−0.138), high-pass (−0.143), false colour (−0.084) and rotation (−0.041), within 0.01 on phase scrambling (+0.010), level on contrast (−0.002) and more robust only to low-pass (+0.078); shaded: families the diets train in. ✱: every model seed beyond the control's value; ≈: within it (one control seed).

**(b)** Error consistency with the ten observers (Cohen's κ as a percentage of the human–human ceiling, κ = 0.234) against the shape-bias rate, for every readout of the w sweep: the model (63 readouts; the best reaches 29.0% of the ceiling), its control and the BA network (best checkpoints), and the CNNs of Geirhos et al. (2020).

**(c, d)** The positive control, measured on the BA network (the model's per-image run is pending): the same machinery recovers 24.1% of the split-half ceiling on whether an image resolves to either cue at all, against 8.7% on which cue is picked (c), and the crossed front-end 2 × 2 withdraws the BA diet reading of (c) (d; solid: train/test matched, pale: crossed).

Sources: `results/ood/`, `results/ood_bestep/`, `results/sweep_error_consistency/`, `perimage_correspondence.json` (`/scratch/byrong/mpk/perimage_behavior/results/2026-09-07_geirhos_perimage/`); `tmp_analysis/figs_v2/fig5_dissociations.py`.
