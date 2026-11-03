**Supplementary Table S1 | Monotonicity of the profile and the exact permutation null — model of record.** Spearman ρ between hierarchical rank [1,1,2,2,3,3,4] and stream preference (per-voxel index: median over reliable voxels of (U_m − U_p)/(U_m + U_p); ratio of medians: median(U_m − U_p)/median(U_m + U_p)), mean over 8 subjects; the null enumerates all 630 distinct labellings of the seven ROIs (one-sided in the pre-stated direction, floor 1/630 = .0016). Source: the stored voxelwise fits of the model, its control, the channel swap and the untrained network.

| arm | seed | statistic | observed mean ρ | exact p | most extreme of 630 | null 2.5th pct |
|---|---|---|---|---|---|---|
| model | 0 | per-voxel index | −0.973 | 0.0016 | yes | −0.748 |
| model | 0 | ratio of medians | −0.973 | 0.0016 | yes | −0.753 |
| model | 1 | per-voxel index | −0.973 | 0.0016 | yes | −0.753 |
| model | 1 | ratio of medians | −0.973 | 0.0016 | yes | −0.748 |
| model | 2 | per-voxel index | −0.964 | 0.0016 | yes | −0.743 |
| model | 2 | ratio of medians | −0.973 | 0.0016 | yes | −0.762 |
| control | 0 | per-voxel index | −0.294 | 0.1159 | no | −0.419 |
| control | 0 | ratio of medians | −0.411 | 0.0524 | no | −0.460 |
| control | 1 | per-voxel index | −0.346 | 0.1286 | no | −0.533 |
| control | 1 | ratio of medians | −0.397 | 0.1095 | no | −0.555 |
| control | 2 | per-voxel index | −0.413 | 0.1175 | no | −0.585 |
| control | 2 | ratio of medians | −0.459 | 0.1016 | no | −0.629 |
| channel swap | 0 | per-voxel index | −0.204 | 0.2222 | no | −0.466 |
| channel swap | 0 | ratio of medians | −0.337 | 0.0825 | no | −0.427 |
| channel swap | 1 | per-voxel index | −0.218 | 0.1952 | no | −0.420 |
| channel swap | 1 | ratio of medians | −0.317 | 0.0714 | no | −0.380 |
| channel swap | 2 | per-voxel index | −0.241 | 0.1968 | no | −0.430 |
| channel swap | 2 | ratio of medians | −0.312 | 0.1413 | no | −0.487 |
| untrained | 0 | per-voxel index | −0.826 | 0.0032 | no | −0.686 |
| untrained | 0 | ratio of medians | −0.817 | 0.0032 | no | −0.677 |

**Supplementary Table S2 | In-silico cue conflict through the fitted encoding models, per seed — model of record.** Median over reliable voxels (noise ceiling > 0.1) of the per-voxel predicted shape-bias index per stream, mean over 8 subjects; pairings are per subject against the seed of the same index (exact Wilcoxon, floor p = .0078). This is PAPER_v2's in-silico index on the NSD-image fits; the cortex-to-decision rate of Results 2.5 refits the encoders on cue-conflict features and is a different instrument.

| seed | arm | magno stream | parvo stream | magno − parvo | model − control | sep. | p | model − swap | sep. | p |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | model | +0.569 | −0.750 | +1.319 | +1.237 | 8/8 | .0078 | +1.333 | 8/8 | .0078 |
| 0 | control | −0.683 | −0.764 | +0.081 |  |  |  |  |  |  |
| 0 | channel swap | −0.714 | −0.699 | −0.015 |  |  |  |  |  |  |
| 1 | model | +0.559 | −0.759 | +1.317 | +1.266 | 8/8 | .0078 | +1.322 | 8/8 | .0078 |
| 1 | control | −0.716 | −0.767 | +0.051 |  |  |  |  |  |  |
| 1 | channel swap | −0.710 | −0.705 | −0.005 |  |  |  |  |  |  |
| 2 | model | +0.570 | −0.769 | +1.339 | +1.270 | 8/8 | .0078 | +1.358 | 8/8 | .0078 |
| 2 | control | −0.681 | −0.749 | +0.069 |  |  |  |  |  |  |
| 2 | channel swap | −0.693 | −0.674 | −0.019 |  |  |  |  |  |  |
| — | untrained | +0.098 | −0.894 | +0.991 | | | | | | |

**Supplementary Table S3 | The carrier × ramp 2×2 on the probe and on behaviour — model of record.** Mean over training seeds with the seed range in brackets. “probe bias” is the stratified-fold probe (Methods 4.3), “exemplars held out” the exemplar-balanced probe (results/figures_v2/exemplar_streams.json, two_by_two_v3, Q1), “cue-conflict shape bias” the restricted estimator of the named stream read through the shared head, “decided” its denominator. Cells are the four of Results 2.2.

| carrier | ramp | condition, stream | probe bias | probe bias, exemplars held out | cue-conflict shape bias | decided / 1,200 | seeds |
|---|---|---|---|---|---|---|---|
| chromatic (L−M) | none | control, L−M stream | −0.413 [0.009] | −0.069 [0.020] | 0.426 [0.079] | 209 | 3 |
| chromatic (L−M) | yes | channel swap, L−M stream | −0.276 [0.024] | −0.016 [0.012] | 0.390 [0.015] | 206 | 3 |
| achromatic (luminance) | none | control, luminance stream | −0.291 [0.004] | −0.177 [0.010] | 0.384 [0.020] | 673 | 3 |
| achromatic (luminance) | yes | model, luminance stream | +0.578 [0.023] | +0.316 [0.031] | 0.850 [0.033] | 501 | 3 |
| chromatic (L−M) | none | model, L−M stream (same cell read in the model; not in the interaction) | −0.454 [0.015] | −0.156 [0.015] | 0.477 [0.032] | 426 | 3 |

Carrier × ramp interaction (achromatic ramp effect − chromatic ramp effect): +0.44 with exemplars held out, +0.73 with exemplars shared; the ramp moves the achromatic stream by +0.49 and the chromatic stream by +0.05 (held out).

**Supplementary Table S4 | The readout ladder — model of record.** Final checkpoint of every run; mean [range] over three seeds; accuracy is validation top-1 at that checkpoint; parameters are counted from the stored state dict with BatchNorm running statistics excluded. No single-trunk network was trained on the dvd300 schedule, so the single-stream rows of the BA table have no counterpart (Supplementary Note S1).

| model / readout | parameters | shape bias | decided / 1,200 | accuracy |
|---|---|---|---|---|
| model, magno stream alone (w = 1) | 22,512,848 | 0.850 [0.838, 0.871] | 501 | 0.683 [0.682, 0.685] |
| model, both streams (w = 0.5) | 22,512,848 | 0.682 [0.659, 0.698] | 580 | 0.683 [0.682, 0.685] |
| model, parvo stream alone (w = 0) | 22,512,848 | 0.477 [0.463, 0.495] | 426 | 0.683 [0.682, 0.685] |
| control, magno stream alone (w = 1) | 22,512,848 | 0.384 [0.376, 0.397] | 673 | 0.764 [0.761, 0.767] |
| control, both streams (w = 0.5) | 22,512,848 | 0.374 [0.359, 0.393] | 672 | 0.764 [0.761, 0.767] |
| control, parvo stream alone (w = 0) | 22,512,848 | 0.426 [0.388, 0.467] | 209 | 0.764 [0.761, 0.767] |
| channel swap, magno stream alone (w = 1) | 22,512,848 | 0.397 [0.380, 0.417] | 649 | 0.760 [0.757, 0.762] |
| channel swap, both streams (w = 0.5) | 22,512,848 | 0.389 [0.378, 0.405] | 654 | 0.760 [0.757, 0.762] |
| channel swap, parvo stream alone (w = 0) | 22,512,848 | 0.390 [0.384, 0.399] | 206 | 0.760 [0.757, 0.762] |
| single-stream controls on the dvd300 schedule | — | not trained | — | — |

**Supplementary Table S5 | Out-of-distribution robustness: relative-accuracy AUC per corruption family — model of record.** Mean over three seeds [range]; “sep.” marks complete seed separation between the model and the control (exact 3 v 3 floor, p = .100); families marked ‡ were declared trained-in before the BA run and keep the mark. Source: results/ood/fs2_*.json.

| family | model | control | channel swap | model − control | sep. |
|---|---|---|---|---|---|
| uniform-noise | 0.646 [0.642, 0.653] | 0.781 [0.779, 0.784] | 0.772 [0.764, 0.777] | −0.135 | yes |
| high-pass | 0.754 [0.751, 0.760] | 0.890 [0.885, 0.897] | 0.891 [0.884, 0.895] | −0.136 | yes |
| false-colour | 0.893 [0.891, 0.896] | 0.977 [0.975, 0.977] | 0.985 [0.983, 0.987] | −0.084 | yes |
| rotation | 0.605 [0.604, 0.606] | 0.644 [0.643, 0.645] | 0.640 [0.637, 0.642] | −0.040 | yes |
| phase-scrambling | 0.395 [0.393, 0.396] | 0.386 [0.385, 0.388] | 0.384 [0.380, 0.386] | +0.009 | yes |
| contrast ‡ | 0.654 [0.651, 0.661] | 0.659 [0.654, 0.666] | 0.653 [0.649, 0.658] | −0.004 | no |
| low-pass ‡ | 0.440 [0.437, 0.443] | 0.362 [0.360, 0.364] | 0.360 [0.357, 0.364] | +0.078 | yes |

**Supplementary Table S6 | Error consistency (Cohen's κ) with the ten human observers — model of record.** Final checkpoints; the trained readout (w = 0.5), the magno stream alone (w = 1) and the best readout over the 21 values of w, per run; κ as a percentage of the human–human ceiling (κ = 0.234, 45 pairs). Source: results/sweep_error_consistency/fs2_*_final_kappa_sweep.json.

| run | shape bias at w = 0.5 | κ at w = 0.5 | % ceiling | κ at w = 1 (magno alone) | % ceiling | rate at w = 1 | best κ over w | % ceiling | w at best | rate at best |
|---|---|---|---|---|---|---|---|---|---|---|
| fs2_dvd300_avg_w050_ls_seed0_final | 0.698 | +0.0220 | 9.4 | +0.0119 | 5.1 | 0.869 | +0.0272 | 11.7 | 0.00 | 0.464 |
| fs2_dvd300_avg_w050_ls_seed1_final | 0.659 | +0.0292 | 12.5 | +0.0599 | 25.6 | 0.845 | +0.0605 | 25.9 | 0.90 | 0.823 |
| fs2_dvd300_avg_w050_ls_seed2_final | 0.690 | +0.0412 | 17.7 | +0.0496 | 21.3 | 0.837 | +0.0677 | 29.0 | 0.75 | 0.806 |
| fs2_dvdfix300_avg_w050_ls_seed0_final | 0.372 | +0.0240 | 10.3 | +0.0251 | 10.8 | 0.379 | +0.0272 | 11.7 | 0.70 | 0.380 |
| fs2_dvdfix300_avg_w050_ls_seed1_final | 0.359 | +0.0286 | 12.2 | +0.0328 | 14.1 | 0.375 | +0.0329 | 14.1 | 0.90 | 0.375 |
| fs2_dvdfix300_avg_w050_ls_seed2_final | 0.393 | +0.0308 | 13.2 | +0.0305 | 13.1 | 0.396 | +0.0325 | 13.9 | 0.40 | 0.384 |
| fs2_dvd300swap_avg_w050_ls_seed0_final | 0.378 | +0.0335 | 14.4 | +0.0306 | 13.1 | 0.378 | +0.0335 | 14.4 | 0.50 | 0.378 |
| fs2_dvd300swap_avg_w050_ls_seed1_final | 0.383 | +0.0262 | 11.2 | +0.0280 | 12.0 | 0.395 | +0.0301 | 12.9 | 0.75 | 0.389 |
| fs2_dvd300swap_avg_w050_ls_seed2_final | 0.405 | +0.0315 | 13.5 | +0.0333 | 14.3 | 0.415 | +0.0346 | 14.8 | 0.80 | 0.412 |

**Supplementary Table S7 | The estimator's own bias, and the residual-misfit variants — model of record.** Seed 0 of the model and the control, as the BA table (seeds 1–2 and the channel swap were not simulated). Definitions as the BA rows: the floor is the recovered hV4 − mean(V1v, V1d) contrast under a true difference of zero; the six variants differ in how the floor is paired and whether the residual misfit of the recovery simulation is removed before or after rescaling by the recovered slope. Source: results/vp_recovery_realnc_v3/.

| quantity | model | control |
|---|---|---|
| floor, normalised contrast (true = 0) | +0.0180 (5/8 > 0, p = .1094) | +0.0655 (8/8 > 0, p = .0078) |
| floor, raw contrast | −0.0009 (p = .3125) | +0.0036 (p = .0078) |
| real contrast, normalised (seed 0) | −1.2594 | −0.0616 |
| displacement real − floor | −1.2773 (8/8, p = .0078) | −0.1271 (8/8, p = .0078) |
| recovered slope of estimate vs truth (intercept fixed at the group floor) | 1.301 | 1.267 |
| variant: group floor, minimal | −0.9819 (8/8, p = .0078) | −0.1003 (8/8, p = .0078) |
| variant: group floor, + residual before scaling | −0.8918 (8/8, p = .0078) | −0.0291 (4/8, p = .6406) |
| variant: group floor, + residual after scaling | −0.8646 (8/8, p = .0078) | −0.0100 (3/8, p = .9453) |
| variant: per-subject floor, minimal | −0.9819 (8/8, p = .0078) | −0.1003 (8/8, p = .0078) |
| variant: per-subject floor, + residual before scaling | −0.8918 (8/8, p = .0078) | −0.0291 (4/8, p = .6406) |
| variant: per-subject floor, + residual after scaling | −0.8646 (8/8, p = .0078) | −0.0100 (4/8, p = .8438) |
| per-ROI normalised floor, V1v … hV4 (model geometry) | +0.083, +0.080, +0.076, +0.064, +0.072, +0.113, +0.100 (ρ vs rank +0.29, p = .0781) |  |
| per-ROI normalised floor, V1v … hV4 (control geometry) |  | +0.113, +0.109, +0.127, +0.165, +0.172, +0.162, +0.176 (ρ vs rank +0.73, p = .0078) |

**Supplementary Table S8 | Stream preference regressed on pRF eccentricity and pRF size within subject — model of record.** Per-voxel stream-preference index regressed on log pRF eccentricity and log pRF size (NSD pRF estimates; voxels with pRF R² > 5%, noise ceiling > 0.1); “ecc | size” is the partial slope for eccentricity controlling for size and vice versa; mean over 8 subjects, with the number of subjects on the stated side and the exact Wilcoxon p. Partial slopes are the stored record of Extended Data Fig. 3 (results/figures_v3/cortex_controls.json); the univariate slopes are computed here with the same voxel selection.

| model | statistic | ecc alone | size alone | ecc \| size | size \| ecc |
|---|---|---|---|---|---|
| model, seed 0 | normalised | +0.085 (0/8 neg, .0078) | −0.013 (5/8 neg, .4609) | +0.154 (0/8 neg, .0078) | −0.110 (8/8 neg, .0078) |
| model, seed 0 | raw | +0.214 (0/8 neg, .0078) | −0.111 (8/8 neg, .0078) | +0.450 (0/8 neg, .0078) | −0.383 (8/8 neg, .0078) |
| model, seed 1 | normalised | +0.082 (0/8 neg, .0078) | −0.018 (5/8 neg, .2500) | +0.142 (0/8 neg, .0078) | −0.102 (8/8 neg, .0078) |
| model, seed 2 | normalised | +0.106 (0/8 neg, .0078) | −0.023 (5/8 neg, .1953) | +0.194 (0/8 neg, .0078) | −0.142 (8/8 neg, .0078) |
| control, seed 0 | normalised | +0.010 (5/8 neg, 1.00) | +0.033 (1/8 neg, .0156) | −0.019 (6/8 neg, .4609) | +0.046 (1/8 neg, .0156) |
| control, seed 0 | raw | +0.029 (2/8 neg, .4609) | +0.179 (0/8 neg, .0078) | −0.134 (7/8 neg, .0547) | +0.260 (0/8 neg, .0078) |
| control, seed 1 | normalised | −0.005 (5/8 neg, .6406) | +0.026 (1/8 neg, .0391) | −0.039 (6/8 neg, .1953) | +0.052 (0/8 neg, .0078) |
| control, seed 2 | normalised | −0.004 (5/8 neg, .8438) | +0.025 (2/8 neg, .0547) | −0.032 (6/8 neg, .1953) | +0.045 (0/8 neg, .0078) |
| channel swap, seed 0 | normalised | −0.001 (5/8 neg, .9453) | +0.034 (1/8 neg, .0234) | −0.034 (6/8 neg, .1953) | +0.055 (1/8 neg, .0156) |
| channel swap, seed 0 | raw | +0.055 (2/8 neg, .3828) | +0.200 (0/8 neg, .0078) | −0.115 (6/8 neg, .1094) | +0.270 (0/8 neg, .0078) |
| channel swap, seed 1 | normalised | +0.011 (4/8 neg, .8438) | +0.041 (0/8 neg, .0078) | −0.023 (7/8 neg, .1953) | +0.055 (1/8 neg, .0156) |
| channel swap, seed 2 | normalised | +0.003 (4/8 neg, 1.00) | +0.023 (2/8 neg, .0391) | −0.017 (6/8 neg, .2500) | +0.034 (1/8 neg, .0234) |
| untrained network, shared front end | normalised | +0.058 (1/8 neg, .0781) | +0.007 (4/8 neg, .6406) | +0.083 (1/8 neg, .0234) | −0.040 (8/8 neg, .0078) |
| untrained network, shared front end | raw | +0.295 (0/8 neg, .0078) | −0.010 (4/8 neg, 1.00) | +0.471 (0/8 neg, .0078) | −0.289 (8/8 neg, .0078) |
| trained model (seed 0) − untrained, paired | normalised | | | +0.071 (7/8 pos, .0781) | −0.070 (0/8 pos, .0078) |
| trained model (seed 0) − untrained, paired | raw | | | −0.021 (2/8 pos, .4609) | −0.094 (0/8 pos, .0078) |
| trained control (seed 0) − untrained, paired | normalised | | | −0.102 (1/8 pos, .0156) | +0.085 (8/8 pos, .0078) |
| trained control (seed 0) − untrained, paired | raw | | | −0.605 (0/8 pos, .0078) | +0.548 (8/8 pos, .0078) |
| arm interaction, model − control (seed-averaged) | normalised | | | +0.193 (8/8 pos, .0078) | −0.165 (0/8 pos, .0078) |
| arm interaction, model − control (seed-averaged) | raw | | | +0.592 (8/8 pos, .0078) | −0.656 (0/8 pos, .0078) |

**Supplementary Table S9 | Cue-conflict shape bias under the forced two-way estimator — model of record.** The restricted (primary) estimator and the forced two-way estimator over all 1,200 images, from the stored per-image logits of the final checkpoints (results/cue_conflict/fs2_*_logits.npz); mean over seeds [range].

| model / readout | restricted (primary) | decided / 1,200 | forced two-way | seeds |
|---|---|---|---|---|
| model, magno stream alone (w = 1) (final ckpt) | 0.850 [0.838, 0.871] | 501 | 0.730 [0.717, 0.740] | 3 |
| model, both streams (w = 0.5) (final ckpt) | 0.682 [0.659, 0.698] | 580 | 0.648 [0.632, 0.663] | 3 |
| model, parvo stream alone (w = 0) (final ckpt) | 0.477 [0.463, 0.495] | 426 | 0.520 [0.501, 0.537] | 3 |
| control, magno stream alone (w = 1) (final ckpt) | 0.384 [0.376, 0.397] | 673 | 0.393 [0.391, 0.394] | 3 |
| control, both streams (w = 0.5) (final ckpt) | 0.374 [0.359, 0.393] | 672 | 0.383 [0.378, 0.393] | 3 |
| control, parvo stream alone (w = 0) (final ckpt) | 0.426 [0.388, 0.467] | 209 | 0.471 [0.453, 0.482] | 3 |
| channel swap, magno stream alone (w = 1) (final ckpt) | 0.397 [0.380, 0.417] | 649 | 0.399 [0.378, 0.415] | 3 |
| channel swap, both streams (w = 0.5) (final ckpt) | 0.389 [0.378, 0.405] | 654 | 0.393 [0.372, 0.407] | 3 |
| channel swap, parvo stream alone (w = 0) (final ckpt) | 0.390 [0.384, 0.399] | 206 | 0.447 [0.439, 0.454] | 3 |

**Supplementary Table S10 | Stream preference beyond hV4, on every NSD atlas — model of record.** Median over reliable voxels of the per-voxel index (noise ceiling > 0.1, U_m + U_p > 10⁻³, ≥ 20 voxels per ROI per subject), mean over the eight subjects, each trained arm the mean over three seeds per subject. Positive = magno band, negative = parvo band. The BA columns are PAPER_v2's stored values; the fs2 arms are recomputed from the stored voxelwise fits (no refitting) and reproduce results/figures_v3/ext_gradient.json.

| atlas | ROI | untrained (shared front end) | control (3 seeds) | model (3 seeds) | channel swap (3 seeds) | BA no diet (3 seeds) | BA (3 seeds) | voxels (median, model) |
|---|---|---|---|---|---|---|---|---|
| Kastner2015 | VO1 | +0.46 (n=8) | +0.92 (n=8) | −0.81 (n=8) | +0.95 (n=8) | +0.92 (n=8) | −0.87 (n=8) | 195 |
| Kastner2015 | VO2 | +0.52 (n=8) | +0.98 (n=8) | −0.87 (n=8) | +0.99 (n=8) | +0.97 (n=8) | −0.90 (n=8) | 218 |
| Kastner2015 | PHC1 | +0.68 (n=8) | +0.99 (n=8) | −0.85 (n=8) | +1.00 (n=8) | +0.98 (n=8) | −0.89 (n=8) | 148 |
| Kastner2015 | PHC2 | +0.73 (n=8) | +1.00 (n=8) | −0.83 (n=8) | +1.00 (n=8) | +0.99 (n=8) | −0.87 (n=8) | 150 |
| Kastner2015 | LO1 | +0.58 (n=8) | +0.90 (n=8) | −0.81 (n=8) | +0.93 (n=8) | +0.92 (n=8) | −0.83 (n=8) | 230 |
| Kastner2015 | LO2 | +0.26 (n=8) | +0.94 (n=8) | −0.86 (n=8) | +0.96 (n=8) | +0.92 (n=8) | −0.90 (n=8) | 167 |
| Kastner2015 | TO1 | +0.53 (n=8) | +0.97 (n=8) | −0.86 (n=8) | +0.99 (n=8) | +0.96 (n=8) | −0.86 (n=8) | 280 |
| Kastner2015 | TO2 | +0.43 (n=8) | +0.98 (n=8) | −0.89 (n=8) | +0.99 (n=8) | +0.96 (n=8) | −0.89 (n=8) | 36 |
| Kastner2015 | V3A | +0.66 (n=8) | +0.87 (n=8) | −0.68 (n=8) | +0.94 (n=8) | +0.91 (n=8) | −0.81 (n=8) | 278 |
| Kastner2015 | V3B | +0.60 (n=8) | +0.96 (n=8) | −0.80 (n=8) | +0.98 (n=8) | +0.96 (n=8) | −0.87 (n=8) | 255 |
| Kastner2015 | IPS0 | +0.43 (n=8) | +0.98 (n=8) | −0.81 (n=8) | +0.99 (n=8) | +0.98 (n=8) | −0.87 (n=8) | 316 |
| Kastner2015 | IPS1 | +0.26 (n=8) | +0.98 (n=8) | −0.82 (n=8) | +0.99 (n=8) | +0.98 (n=8) | −0.86 (n=8) | 237 |
| Kastner2015 | IPS2 | +0.28 (n=8) | +0.99 (n=8) | −0.89 (n=8) | +1.01 (n=8) | +1.00 (n=8) | −0.92 (n=8) | 47 |
| streams | early | +0.76 (n=8) | +0.74 (n=8) | −0.06 (n=8) | +0.83 (n=8) | +0.86 (n=8) | −0.42 (n=8) | 3106 |
| streams | midventral | +0.38 (n=8) | +0.81 (n=8) | −0.77 (n=8) | +0.88 (n=8) | +0.87 (n=8) | −0.84 (n=8) | 704 |
| streams | midlateral | +0.46 (n=8) | +0.88 (n=8) | −0.85 (n=8) | +0.92 (n=8) | +0.91 (n=8) | −0.86 (n=8) | 570 |
| streams | midparietal | +0.62 (n=8) | +0.93 (n=8) | −0.75 (n=8) | +0.97 (n=8) | +0.94 (n=8) | −0.84 (n=8) | 511 |
| streams | ventral | +0.44 (n=8) | +0.98 (n=8) | −0.90 (n=8) | +0.99 (n=8) | +0.96 (n=8) | −0.91 (n=8) | 2724 |
| streams | lateral | +0.51 (n=8) | +0.99 (n=8) | −0.89 (n=8) | +0.99 (n=8) | +0.97 (n=8) | −0.87 (n=8) | 1838 |
| streams | parietal | +0.42 (n=8) | +0.99 (n=8) | −0.83 (n=8) | +1.00 (n=8) | +0.99 (n=8) | −0.87 (n=8) | 1076 |
| floc-faces | OFA | +0.36 (n=8) | +0.86 (n=8) | −0.90 (n=8) | +0.91 (n=8) | +0.90 (n=8) | −0.89 (n=8) | 364 |
| floc-faces | FFA-1 | +0.36 (n=8) | +0.97 (n=8) | −0.93 (n=8) | +0.98 (n=8) | +0.95 (n=8) | −0.91 (n=8) | 398 |
| floc-faces | FFA-2 | +0.55 (n=7) | +0.99 (n=7) | −0.95 (n=7) | +1.00 (n=7) | +0.98 (n=7) | −0.92 (n=7) | 261 |
| floc-places | OPA | +0.61 (n=8) | +0.98 (n=8) | −0.79 (n=8) | +0.99 (n=8) | +0.98 (n=8) | −0.86 (n=8) | 1067 |
| floc-places | PPA | +0.71 (n=8) | +1.00 (n=8) | −0.84 (n=8) | +1.00 (n=8) | +0.99 (n=8) | −0.88 (n=8) | 618 |
| floc-bodies | EBA | +0.46 (n=8) | +0.98 (n=8) | −0.90 (n=8) | +0.99 (n=8) | +0.97 (n=8) | −0.88 (n=8) | 2178 |
| floc-bodies | FBA-1 | +0.28 (n=6) | +0.98 (n=6) | −0.91 (n=6) | +0.99 (n=6) | +0.96 (n=6) | −0.91 (n=6) | 314 |
| floc-bodies | FBA-2 | +0.45 (n=8) | +1.00 (n=8) | −0.94 (n=8) | +0.99 (n=8) | +0.98 (n=8) | −0.92 (n=8) | 252 |
| floc-words | OWFA | +0.43 (n=8) | +0.76 (n=8) | −0.84 (n=8) | +0.83 (n=8) | +0.88 (n=8) | −0.85 (n=8) | 394 |
| floc-words | VWFA-1 | +0.18 (n=8) | +0.99 (n=8) | −0.91 (n=8) | +1.00 (n=8) | +0.96 (n=8) | −0.91 (n=8) | 460 |
| floc-words | VWFA-2 | +0.19 (n=5) | +0.98 (n=6) | −0.91 (n=5) | +0.99 (n=6) | +0.95 (n=6) | −0.94 (n=6) | 150 |
| floc-words | mfs-words | +0.21 (n=8) | +0.99 (n=8) | −0.93 (n=8) | +1.00 (n=8) | +0.97 (n=8) | −0.93 (n=8) | 122 |
