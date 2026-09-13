**Extended Data Table 1 | The cortical gradient hV4 − mean(V1v, V1d), per seed and statistic.** “normalised” is the per-voxel index (U_m − U_p)/(U_m + U_p), “raw” the per-voxel difference U_m − U_p, each summarised by its median over reliable voxels per ROI. Mean over 8 subjects of the per-subject contrast; “sep.” is the number of subjects in which the diet arm is more negative than the no-diet arm; p is the exact paired Wilcoxon (floor .0078). The n = 7 column drops subject 01 (Results 2.3). Source: `results/encoding_cc/`.

| statistic | seed | diet | no diet (n = 8) | no diet (n = 7) | sep. | p |
|---|---|---|---|---|---|---|
| normalised | 0 | −0.833 | −0.078 | −0.054 | 8/8 | 0.0078 |
|  | 1 | −0.785 | −0.096 | −0.084 | 8/8 | 0.0078 |
|  | 2 | −0.745 | −0.049 | −0.031 | 8/8 | 0.0078 |
| raw | 0 | −0.166 | −0.027 | −0.021 | 8/8 | 0.0078 |
|  | 1 | −0.145 | −0.031 | −0.030 | 8/8 | 0.0078 |
|  | 2 | −0.153 | −0.023 | −0.021 | 8/8 | 0.0078 |
| untrained, diet front end | — | −0.544 | | | 8/8 < 0 | 0.0078 |
| untrained, no-diet front end | — | | −0.479 | | 8/8 < 0 | 0.0078 |
| swap, seed 0 (degenerate: U_p ≈ 0) | 0 | −0.006 | | | | |
| parvo_highsf, seed 0 (degenerate: U_m ≈ 0) | 0 | −0.143 | | | | |
| thesis checkpoint (progressive schedule) | — | −0.793 | | | | |

**Extended Data Table 2 | Monotonicity of the profile and the exact permutation null.** Spearman ρ between hierarchical rank [1,1,2,2,3,3,4] and stream preference, mean over subjects; the null enumerates all 630 distinct labellings of the seven ROIs (one-sided in the pre-stated direction, floor 1/630 = .0016). Source: `results/hierarchy_permutation_null.json`.

| arm | seed | statistic | observed mean ρ | exact p | most extreme of 630 | null 2.5th pct |
|---|---|---|---|---|---|---|
| diet | 0 | norm | -0.964 | 0.0016 | yes | -0.754 |
| diet | 0 | raw | -0.964 | 0.0016 | yes | -0.743 |
| diet | 1 | norm | -0.973 | 0.0016 | yes | -0.750 |
| diet | 1 | raw | -0.973 | 0.0016 | yes | -0.743 |
| diet | 2 | norm | -0.966 | 0.0016 | yes | -0.748 |
| diet | 2 | raw | -0.966 | 0.0016 | yes | -0.756 |
| no diet | 0 | norm | -0.502 | 0.0286 | no | -0.502 |
| no diet | 0 | raw | -0.571 | 0.0206 | no | -0.559 |
| no diet | 1 | norm | -0.606 | 0.0190 | no | -0.546 |
| no diet | 1 | raw | -0.661 | 0.0190 | no | -0.594 |
| no diet | 2 | norm | -0.580 | 0.0365 | no | -0.616 |
| no diet | 2 | raw | -0.463 | 0.0921 | no | -0.594 |

**Extended Data Table 3 | In-silico cue conflict through the fitted encoding models, per seed.** Median over reliable voxels of the per-voxel predicted shape-bias index for each stream, mean over 8 subjects; the contrast is magno − parvo, paired within subject against the no-diet arm of the same seed. Source: `results/encoding_cc/`.

| seed | arm | magno stream | parvo stream | magno − parvo | diet − no diet | sep. | p |
|---|---|---|---|---|---|---|---|
| 0 | diet | +0.791 | −0.743 | +1.534 | +1.424 | 8/8 | 0.0078 |
| 0 | no diet | −0.638 | −0.748 | +0.110 | | | |
| 1 | diet | +0.796 | −0.770 | +1.566 | +1.468 | 8/8 | 0.0078 |
| 1 | no diet | −0.601 | −0.699 | +0.098 | | | |
| 2 | diet | +0.799 | −0.759 | +1.558 | +1.474 | 8/8 | 0.0078 |
| 2 | no diet | −0.613 | −0.696 | +0.083 | | | |

**Extended Data Table 4 | The carrier × transform 2×2 on the probe and on behaviour.** Mean over training seeds with the seed range in brackets; “decided” is the mean number of the 1,200 cue-conflict images the readout classifies as the shape or the texture category (the shape-bias denominator). Cells are read out on stream 2 in three cases and on the diet's magno stream in the fourth. Source: `results/figures/fig1_data.npz` (from `results/shape_texture/`, `results/cue_conflict/`).

| carrier | transform | condition, stream | probe bias | cue-conflict shape bias | decided / 1,200 | seeds |
|---|---|---|---|---|---|---|
| chromatic (L−M) | none | no diet, parvo | −0.504 [0.035] | 0.364 [0.080] | 270 | 3 |
| chromatic (L−M) | yes | swap, parvo | −0.145 [0.043] | 0.495 [0.038] | 165 | 3 |
| achromatic (luminance) | none | parvo_highsf, stream 2 | −0.266 [0.017] | 0.396 [0.012] | 584 | 2 |
| achromatic (luminance) | yes | diet, magno | +0.819 [0.014] | 0.878 [0.022] | 488 | 3 |

**Extended Data Table 5 | Single-stream controls trained with the identical diet, and the readout ladder.** Best saved checkpoint per seed; mean [range] over three seeds. Accuracy is validation top-1 at the scored checkpoint. Sources: `results/cue_conflict_bestep/`, `results/cue_conflict_single34_bestep/`, `results/cue_conflict_single_bestep/`, run histories, `results/param_matched/`.

| model / readout | parameters | shape bias | decided / 1,200 | accuracy |
|---|---|---|---|---|
| diet, magno stream alone | 22,397,936 | 0.903 [0.888, 0.916] | 504 | 0.556 [0.552, 0.564] |
| diet, both streams (w = 0.5) | 22,397,936 | 0.655 [0.594, 0.718] | 502 | 0.556 [0.552, 0.564] |
| diet, parvo stream alone | 22,397,936 | 0.400 [0.372, 0.429] | 388 | 0.556 [0.552, 0.564] |
| single stream, ResNet-34 (matched) | 21,338,992 | 0.625 [0.604, 0.652] | 537 | 0.623 [0.616, 0.634] |
| single stream, ResNet-18 (half) | 11,230,832 | 0.567 [0.541, 0.590] | 501 | 0.599 [0.589, 0.608] |
| no diet, both streams | 22,397,936 | 0.416 [0.394, 0.433] | 593 | 0.701 [0.698, 0.706] |

**Extended Data Table 6 | Out-of-distribution robustness: relative-accuracy AUC per corruption family and condition.** Mean over three seeds [range]; the thesis's schedule-trained checkpoint is a single run. “sep.” marks complete seed separation between the diet and no-diet arms (the exact 3 v 3 floor, p = 0.100). Families marked ‡ were declared trained-in before the run. Source: `results/ood_bestep/`.

| family | diet | no diet | swap | parvo_highsf | thesis schedule | diet − no diet | sep. |
|---|---|---|---|---|---|---|---|
| uniform-noise | 0.634 [0.619, 0.647] | 0.729 [0.720, 0.735] | 0.749 [0.744, 0.758] | 0.767 [0.758, 0.773] | 0.640 | −0.095 | yes |
| high-pass | 0.678 [0.647, 0.696] | 0.827 [0.820, 0.831] | 0.831 [0.817, 0.841] | 0.783 [0.775, 0.791] | 0.680 | −0.149 | yes |
| false-colour | 0.854 [0.845, 0.860] | 0.938 [0.933, 0.943] | 0.966 [0.960, 0.971] | 0.874 [0.868, 0.880] | 0.839 | −0.084 | yes |
| rotation | 0.574 [0.567, 0.582] | 0.615 [0.613, 0.616] | 0.605 [0.597, 0.612] | 0.611 [0.602, 0.616] | 0.572 | −0.041 | yes |
| phase-scrambling | 0.382 [0.365, 0.391] | 0.373 [0.369, 0.377] | 0.371 [0.365, 0.375] | 0.382 [0.380, 0.383] | 0.362 | +0.009 | no |
| contrast ‡ | 0.481 [0.460, 0.493] | 0.549 [0.530, 0.578] | 0.543 [0.539, 0.550] | 0.511 [0.498, 0.526] | 0.489 | −0.068 | yes |
| low-pass ‡ | 0.444 [0.434, 0.464] | 0.356 [0.353, 0.359] | 0.359 [0.353, 0.362] | 0.385 [0.375, 0.392] | 0.436 | +0.089 | yes |

**Extended Data Table 7 | Error consistency (Cohen's κ) with the ten human observers.** The trained readout (w = 0.5) and the best readout over the 21 values of w, per run; κ as a percentage of the human–human ceiling (κ = 0.234, 45 pairs). Published CNN values are from Geirhos et al. (2020). Source: `results/sweep_error_consistency/`.

| run | shape bias at w = 0.5 | κ at w = 0.5 | % ceiling | best κ over w | % ceiling | w at best | rate at best |
|---|---|---|---|---|---|---|---|
| dkl_base_seed0_final | 0.421 | +0.0309 | 13.2 | +0.0404 | 17.3 | 1.00 | 0.438 |
| dkl_base_seed1_final | 0.433 | +0.0333 | 14.3 | +0.0421 | 18.0 | 0.75 | 0.429 |
| dkl_base_seed2_final | 0.394 | +0.0372 | 15.9 | +0.0412 | 17.6 | 0.35 | 0.410 |
| dkl_bcs_seed0_best | 0.594 | +0.0096 | 4.1 | +0.0178 | 7.6 | 1.00 | 0.890 |
| dkl_bcs_seed0_final | 0.692 | +0.0234 | 10.0 | +0.0405 | 17.3 | 0.90 | 0.857 |
| dkl_bcs_seed1_best | 0.718 | +0.0269 | 11.5 | +0.0425 | 18.2 | 0.35 | 0.614 |
| dkl_bcs_seed1_final | 0.686 | +0.0319 | 13.7 | +0.0437 | 18.7 | 0.45 | 0.651 |
| dkl_bcs_seed2_best | 0.652 | +0.0372 | 15.9 | +0.0483 | 20.7 | 0.85 | 0.847 |
| dkl_bcs_seed2_final | 0.646 | +0.0280 | 12.0 | +0.0405 | 17.3 | 0.75 | 0.784 |
| AlexNet (published) | 0.429 | +0.0320 | 13.7 | | | | |
| GoogLeNet (published) | 0.312 | +0.0250 | 10.7 | | | | |
| VGG-16 (published) | 0.172 | +0.0170 | 7.3 | | | | |

**Extended Data Table 8 | The estimator's own bias, and the residual-misfit variants.** Synthetic voxels were generated from each arm's fitted 1,000-component feature spaces with the measured per-ROI noise ceilings imposed and the true magno − parvo difference set to zero everywhere; the recovered contrast is the instrument's signed floor. “Displacement” is the real contrast minus the floor, paired per subject. The six variants differ in how the floor is paired (group or per-subject) and whether the residual misfit of the recovery simulation is also removed (before or after rescaling by the recovered slope). Sources: `results/vp_recovery_realnc/diet_floor_paired_2026-09-07.txt`, `floor_pairing_2026-09-07.txt`, `prediction_scored_2026-09-07.txt`.

| quantity | diet | no diet |
|---|---|---|
| floor, normalised contrast (true = 0) | +0.0375 (8/8 > 0, p = .0078) | +0.0511 (8/8 > 0, p = .0078) |
| floor, raw contrast | −0.0021 (p = .15) | +0.0009 (p = .84) |
| real contrast, normalised (seed 0) | −0.8335 | −0.0785 |
| displacement real − floor | −0.871 (8/8, p = .0078) | −0.130 (8/8, p = .0078); n = 7: −0.103 (7/7, p = .0156) |
| recovered slope of estimate vs truth | 1.210 | 1.281 |
| variant: group floor, minimal | −0.720 (8/8, p = .0078) | −0.101 (8/8, p = .0078) |
| variant: per-subject floor, minimal | −0.720 (8/8, p = .0078) | −0.101 (8/8, p = .0078) |
| variant: group floor + residual before scaling | −0.626 (8/8, p = .0078) | −0.034 (6/8, p = .055) |
| variant: per-subject floor + residual before scaling | −0.626 (8/8, p = .0078) | −0.034 (5/8, p = .31) |
| variant: group floor + residual after scaling | −0.607 (8/8, p = .0078) | −0.015 (4/8, p = 1.00) |
| variant: per-subject floor + residual after scaling | −0.607 (8/8, p = .0078) | −0.015 (5/8, p = .84) |
| per-ROI normalised floor, V1v … hV4 (no-diet geometry) | | +0.025, +0.025, +0.026, +0.025, +0.027, +0.026, +0.026 (ρ vs rank +0.54, p = .22) |

**Extended Data Table 9 | Stream preference regressed on pRF eccentricity and pRF size within subject.** Per-voxel stream-preference index regressed on log pRF eccentricity and log pRF size (NSD pRF estimates, voxels with pRF R² > 5%, noise ceiling > 0.1); “ecc | size” is the partial slope for eccentricity controlling for size and vice versa; mean over 8 subjects, with the number of subjects on the stated side and the exact Wilcoxon p. Collinearity between the two predictors is modest (mean VIF 1.66). The arm interaction is the paired difference of partial slopes, seed-averaged within arm. Sources: `results/prf_continuous/dissociation_2026-09-08.txt`, `interaction_2026-09-08.txt`, `input_null_2026-09-08.txt`.

| model | statistic | ecc alone | size alone | ecc \| size | size \| ecc |
|---|---|---|---|---|---|
| diet, seed 0 | normalised | +0.066 (0/8 neg, .0078) | −0.013 (5/8 neg, .31) | +0.119 (0/8 neg, .0078) | −0.086 (8/8 neg, .0078) |
| diet, seed 1 | normalised | +0.065 (0/8, .0078) | −0.029 (7/8, .039) | +0.132 (0/8, .0078) | −0.110 (8/8, .0078) |
| diet, seed 2 | normalised | +0.064 (0/8, .0078) | −0.012 (5/8, .25) | +0.115 (0/8, .0078) | −0.083 (8/8, .0078) |
| no diet, seed 0 | normalised | +0.018 (4/8, .64) | +0.033 (1/8, .039) | −0.004 (4/8, 1.00) | +0.035 (0/8, .0078) |
| no diet, seed 1 | normalised | +0.004 (4/8, .95) | +0.015 (4/8, .38) | −0.009 (6/8, .46) | +0.021 (2/8, .11) |
| no diet, seed 2 | normalised | −0.007 (5/8, .38) | +0.012 (3/8, .31) | −0.022 (6/8, .25) | +0.024 (1/8, .016) |
| diet, seed 0 | raw | +0.103 (1/8, .039) | −0.158 (8/8, .0078) | +0.325 (0/8, .0078) | −0.356 (8/8, .0078) |
| no diet, seed 0 | raw | +0.130 (1/8, .023) | +0.181 (0/8, .0078) | +0.028 (3/8, .84) | +0.163 (0/8, .0078) |
| Gabor bank, diet front end (input null) | normalised | | | +0.035 (6/8 pos, .15) | +0.002 (4/8 neg, .95) |
| untrained network, diet front end | normalised | | | +0.083 (7/8 pos, .016) | −0.038 (6/8 neg, .039) |
| arm interaction, diet − no diet (seed-averaged) | normalised | | | +0.134 (8/8, .0078) | −0.119 (0/8 pos, .0078) |
| arm interaction, diet − no diet (seed-averaged) | raw | | | +0.305 (8/8, .0078) | −0.502 (0/8 pos, .0078) |

**Extended Data Table 10 | Cue-conflict shape bias under the forced two-way estimator.** The primary estimator (Methods 4.3) restricts the logits to the 16 categories, takes the argmax, and scores shape against texture over the images decided as one of the two, so images decided as a third category drop out of the denominator (“decided”). The forced estimator keeps all 1,200 images and scores, per image, whether the logit of the shape category exceeds the logit of the texture category. Mean over seeds [range]. Source: the stored per-image logits in `results/cue_conflict*/`.

| model / readout | restricted (primary) | decided / 1,200 | forced two-way | seeds |
|---|---|---|---|---|
| diet, magno stream alone (best ckpt) | 0.903 [0.888, 0.916] | 504 | 0.775 [0.767, 0.788] | 3 |
| diet, both streams w = 0.5 (best ckpt) | 0.655 [0.594, 0.718] | 502 | 0.588 [0.574, 0.609] | 3 |
| diet, parvo stream alone (best ckpt) | 0.400 [0.372, 0.429] | 388 | 0.414 [0.394, 0.428] | 3 |
| diet, magno stream alone (final ckpt) | 0.878 [0.870, 0.892] | 488 | 0.761 [0.757, 0.764] | 3 |
| diet, both streams w = 0.5 (final ckpt) | 0.675 [0.646, 0.692] | 475 | 0.608 [0.596, 0.618] | 3 |
| diet, parvo stream alone (final ckpt) | 0.418 [0.406, 0.438] | 353 | 0.449 [0.411, 0.472] | 3 |
| no diet, both streams | 0.416 [0.394, 0.433] | 593 | 0.375 [0.357, 0.388] | 3 |
| no diet, magno stream alone | 0.421 [0.398, 0.434] | 602 | 0.378 [0.352, 0.398] | 3 |
| no diet, parvo stream alone | 0.364 [0.311, 0.391] | 270 | 0.442 [0.410, 0.473] | 3 |
| swap, both streams | 0.408 [0.403, 0.412] | 586 | 0.365 [0.362, 0.367] | 3 |
| swap, parvo stream alone (L−M + transform) | 0.495 [0.478, 0.516] | 165 | 0.511 [0.491, 0.531] | 3 |
| parvo_highsf, both streams | 0.486 [0.450, 0.516] | 618 | 0.450 [0.438, 0.457] | 3 |
| parvo_highsf, stream 2 alone (L−M + luminance) | 0.389 [0.374, 0.402] | 598 | 0.346 [0.324, 0.358] | 3 |
| single stream, ResNet-34 (matched) | 0.625 [0.604, 0.652] | 537 | 0.561 [0.552, 0.573] | 3 |
| single stream, ResNet-18 (half) | 0.567 [0.541, 0.590] | 501 | 0.523 [0.506, 0.538] | 3 |
| no diet, accuracy-matched (epoch 10), magno alone | 0.455 | 532 | 0.370 | 1 |
| thesis checkpoint (progressive schedule), both streams | 0.697 | 620 | 0.636 | 1 |
| untrained, diet front end, magno alone | 0.513 | 154 | 0.510 | 1 |
| untrained, no-diet front end, magno alone | 0.568 | 169 | 0.508 | 1 |

**Extended Data Table 11 | Stream preference beyond hV4, on every NSD atlas.** Median over reliable voxels of the per-voxel index (Um − Up)/(Um + Up) (noise ceiling > 0.1, Um + Up > 10⁻³, ≥ 20 voxels per ROI per subject), mean over the eight subjects (n = subjects with enough voxels); trained arms are the mean over training seeds per subject. Positive = magno band, negative = parvo band. Atlases: Wang et al. (2015) Kastner2015 ROIs beyond hV4 (the V1–hV4 rows are Extended Data Table 1 and Fig. 3), NSD's `streams` atlas, and the fLoc category-selective ROIs. Regional contrasts and the exact tests are in Extended Data Fig. 4c and `results/figures_v2/ext_gradient.md`. Source: the stored voxelwise fits in `results/encoding_cc/` and `results/encoding/`, relabelled; no refitting.

| atlas | ROI | untrained (diet front end) | no diet (3 seeds) | diet (3 seeds) | swap | parvo + high-SF | thesis ckpt | voxels (median, diet) |
|---|---|---|---|---|---|---|---|---|
| Kastner2015 | VO1 | +0.00 (n=8) | +0.92 (n=8) | −0.87 (n=8) | +0.98 (n=8) | −0.99 (n=8) | −0.94 (n=8) | 196 |
| Kastner2015 | VO2 | +0.12 (n=8) | +0.97 (n=8) | −0.90 (n=8) | +1.00 (n=8) | −1.00 (n=8) | −0.94 (n=8) | 220 |
| Kastner2015 | PHC1 | +0.41 (n=8) | +0.98 (n=8) | −0.89 (n=8) | +1.00 (n=8) | −1.00 (n=8) | −0.94 (n=8) | 148 |
| Kastner2015 | PHC2 | +0.46 (n=8) | +0.99 (n=8) | −0.87 (n=8) | +1.00 (n=8) | −0.99 (n=8) | −0.93 (n=8) | 150 |
| Kastner2015 | LO1 | +0.13 (n=8) | +0.92 (n=8) | −0.83 (n=8) | +0.99 (n=8) | −0.98 (n=8) | −0.90 (n=8) | 230 |
| Kastner2015 | LO2 | −0.14 (n=8) | +0.92 (n=8) | −0.90 (n=8) | +0.99 (n=8) | −0.99 (n=8) | −0.94 (n=8) | 166 |
| Kastner2015 | TO1 | +0.25 (n=8) | +0.96 (n=8) | −0.86 (n=8) | +0.99 (n=8) | −1.00 (n=8) | −0.91 (n=8) | 280 |
| Kastner2015 | TO2 | +0.20 (n=8) | +0.96 (n=8) | −0.89 (n=8) | +0.99 (n=8) | −1.00 (n=8) | −0.90 (n=8) | 36 |
| Kastner2015 | V3A | +0.39 (n=8) | +0.91 (n=8) | −0.81 (n=8) | +0.99 (n=8) | −0.98 (n=8) | −0.84 (n=8) | 278 |
| Kastner2015 | V3B | +0.24 (n=8) | +0.96 (n=8) | −0.87 (n=8) | +1.00 (n=8) | −0.99 (n=8) | −0.94 (n=8) | 254 |
| Kastner2015 | IPS0 | +0.15 (n=8) | +0.98 (n=8) | −0.87 (n=8) | +1.01 (n=8) | −1.00 (n=8) | −0.92 (n=8) | 317 |
| Kastner2015 | IPS1 | −0.06 (n=8) | +0.98 (n=8) | −0.86 (n=8) | +1.00 (n=8) | −1.00 (n=8) | −0.92 (n=8) | 239 |
| Kastner2015 | IPS2 | +0.02 (n=8) | +1.00 (n=8) | −0.92 (n=8) | +1.01 (n=8) | −1.01 (n=8) | −0.94 (n=8) | 47 |
| streams | early | +0.27 (n=8) | +0.86 (n=8) | −0.42 (n=8) | +0.96 (n=8) | −0.92 (n=8) | −0.45 (n=8) | 3133 |
| streams | midventral | −0.15 (n=8) | +0.87 (n=8) | −0.84 (n=8) | +0.97 (n=8) | −0.98 (n=8) | −0.86 (n=8) | 708 |
| streams | midlateral | −0.04 (n=8) | +0.91 (n=8) | −0.86 (n=8) | +0.98 (n=8) | −0.99 (n=8) | −0.90 (n=8) | 571 |
| streams | midparietal | +0.32 (n=8) | +0.94 (n=8) | −0.84 (n=8) | +1.00 (n=8) | −0.99 (n=8) | −0.91 (n=8) | 512 |
| streams | ventral | +0.08 (n=8) | +0.96 (n=8) | −0.91 (n=8) | +1.00 (n=8) | −1.00 (n=8) | −0.95 (n=8) | 2724 |
| streams | lateral | +0.25 (n=8) | +0.97 (n=8) | −0.87 (n=8) | +1.00 (n=8) | −1.00 (n=8) | −0.91 (n=8) | 1842 |
| streams | parietal | +0.12 (n=8) | +0.99 (n=8) | −0.87 (n=8) | +1.01 (n=8) | −1.00 (n=8) | −0.93 (n=8) | 1082 |
| floc-faces | OFA | −0.07 (n=8) | +0.90 (n=8) | −0.89 (n=8) | +0.98 (n=8) | −1.00 (n=8) | −0.92 (n=8) | 367 |
| floc-faces | FFA-1 | +0.01 (n=8) | +0.95 (n=8) | −0.91 (n=8) | +0.99 (n=8) | −1.00 (n=8) | −0.94 (n=8) | 399 |
| floc-faces | FFA-2 | +0.22 (n=7) | +0.98 (n=7) | −0.92 (n=7) | +0.99 (n=7) | −1.00 (n=7) | −0.95 (n=7) | 260 |
| floc-places | OPA | +0.34 (n=8) | +0.98 (n=8) | −0.86 (n=8) | +1.01 (n=8) | −1.00 (n=8) | −0.93 (n=8) | 1070 |
| floc-places | PPA | +0.44 (n=8) | +0.99 (n=8) | −0.88 (n=8) | +1.00 (n=8) | −0.99 (n=8) | −0.94 (n=8) | 620 |
| floc-bodies | EBA | +0.17 (n=8) | +0.97 (n=8) | −0.88 (n=8) | +1.00 (n=8) | −1.00 (n=8) | −0.92 (n=8) | 2174 |
| floc-bodies | FBA-1 | −0.07 (n=6) | +0.96 (n=6) | −0.91 (n=6) | +1.00 (n=6) | −1.00 (n=6) | −0.95 (n=6) | 314 |
| floc-bodies | FBA-2 | +0.13 (n=8) | +0.98 (n=8) | −0.92 (n=8) | +0.99 (n=8) | −1.00 (n=8) | −0.94 (n=8) | 252 |
| floc-words | OWFA | −0.29 (n=8) | +0.88 (n=8) | −0.85 (n=8) | +0.98 (n=8) | −0.99 (n=8) | −0.88 (n=8) | 392 |
| floc-words | VWFA-1 | −0.16 (n=8) | +0.96 (n=8) | −0.91 (n=8) | +1.00 (n=8) | −1.00 (n=8) | −0.96 (n=8) | 454 |
| floc-words | VWFA-2 | −0.02 (n=5) | +0.95 (n=6) | −0.94 (n=6) | +0.99 (n=6) | −1.00 (n=6) | −0.96 (n=6) | 150 |
| floc-words | mfs-words | −0.08 (n=8) | +0.97 (n=8) | −0.93 (n=8) | +0.99 (n=8) | −1.00 (n=8) | −0.96 (n=8) | 121 |

