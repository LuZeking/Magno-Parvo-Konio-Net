# Figure 5 — caption

**Three properties the field treats as travelling with shape bias do not follow from it.**

**(a)** Out-of-distribution robustness: relative-accuracy area under the curve (accuracy over clean accuracy, mean over severities) per corruption family, one marker per seed; the model against its control (final checkpoints, three seeds each, identical test input), the BA network's two arms as small squares (best saved checkpoints). The model is less robust than its control on uniform noise (−0.135), high-pass (−0.136), false colour (−0.084) and rotation (−0.040), every model seed below every control seed (exact Mann–Whitney p = 0.10, the floor for three against three); within 0.01 on phase scrambling (+0.009), level on contrast (−0.004, seeds interleaved) and more robust only to low-pass (+0.078); shaded: families the diets train in. ✱: every model seed beyond every control seed; ≈: seeds overlap.

**(b)** Error consistency with the ten observers (Cohen's κ as a percentage of the human–human ceiling, κ = 0.234) against the shape-bias rate, for every readout of the w sweep: the model (63 readouts; the best reaches 29.0% of the ceiling), its control and the BA network (best checkpoints), and the CNNs of Geirhos et al. (2020).

**(c)** Per-image correspondence with the ten observers on held-out images, as a percentage of each target's own split-half ceiling, after removing shape- and texture-category main effects: whether an image resolves to either cue at all (difficulty) and which cue is picked; bars are seed means, dots seeds. On difficulty the model reaches 22.3% (seeds 19.6–24.3%) against 9.0% for the control and 8.2% for the channel swap, every model seed above every seed of both (exact Mann–Whitney p = 0.10); on the cue picked the three are interleaved (13.3%, 10.4%, 9.5%). All three are tested on the identical input.

**(d)** The BA network's crossed front-end 2 × 2 on the same difficulty measure: its diet arm's 24.1% against 9.2% without the diet vanishes when weights and test input are crossed (−0.6% and −3.7%), so there it measured whether a network is tested on the input it was trained on, not the diet (solid: train/test matched, pale: crossed).

Sources: `results/ood/`, `results/ood_bestep/`, `results/sweep_error_consistency/`, `results/perimage_behavior/perimage_correspondence_v3.json` (`tmp_analysis/perimage_correspondence.py`); `tmp_analysis/figs_v2/fig5_dissociations.py`.
