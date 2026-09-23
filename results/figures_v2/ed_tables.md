**Extended Data Table 1 | The cortical gradient hV4 − mean(V1v, V1d), per seed and statistic.** “normalised” is the per-voxel index (U_m − U_p)/(U_m + U_p), “raw” the per-voxel difference U_m − U_p, each summarised by its median over reliable voxels per ROI. Mean over 8 subjects of the per-subject contrast; “sep.” is the number of subjects in which the diet arm is more negative than the no-diet arm, and for the rows with no paired comparison the number in which the contrast is negative; p is the exact Wilcoxon, paired for the diet against no-diet rows and one-sample otherwise (floor .0078). The n = 7 column drops subject 01 (Results 2.3). The swap and parvo_highsf arms are degenerate by construction, in every seed: the swapped network leaves the parvo band a median unique variance 0.005–0.010 of the magno band's, and the parvo + high-SF network leaves the magno band 0.003–0.005 of the parvo band's, so their gradients describe one surviving band and are not comparable with the two-band arms. Sources: `results/encoding_cc/` (diet, no diet, thesis) and `results/encoding/` (untrained, swap, parvo_highsf).

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
| swap (degenerate: U_p ≈ 0) | 0 | −0.006 | | | 5/8 < 0 | 0.4609 |
|  | 1 | −0.034 | | | 7/8 < 0 | 0.0234 |
|  | 2 | −0.014 | | | 6/8 < 0 | 0.1953 |
| parvo_highsf (degenerate: U_m ≈ 0) | 0 | −0.143 | | | 8/8 < 0 | 0.0078 |
|  | 1 | −0.082 | | | 8/8 < 0 | 0.0078 |
|  | 2 | −0.108 | | | 8/8 < 0 | 0.0078 |
| thesis checkpoint (progressive schedule) | — | −0.793 | | | | |

**Extended Data Table 2 | The readout weight cortex prefers against the one behaviour prefers.** For each mixing weight w the classifier's own input z(w) = w·y₁ + (1 − w)·y₂ (y₁ magno stream output, y₂ parvo stream output, 512-d each) is the design of a ridge encoding model of every nsdgeneral voxel (same split, folds and alpha grid as the banded model; Methods 4.5). w*_cortex is the w that maximises the median test R² over reliable voxels of the ROI (mean over seeds per subject; median [range] over 8 subjects). "R² at w = 0.5 / max" is the median R² at the paper's readout weight relative to the best w. "concat" is the median R² of the 1,024-d concatenation [y₁, y₂], fitted as a reference and not as a bound: every mixture is a linear map of the concatenation, but the two models are regularised separately on a shared alpha grid, and the best mixture exceeds the fitted concatenation in 36 of the 112 subject × ROI × arm cells, by a median 0.0019 R² (1.5% of the concatenation; largest 0.0054). That excess is the reproducibility scale of these curves and exceeds the median R² the sweep itself buys, 0.0016 from w = ½ to w*, so w*_cortex is read as the location of an optimum and never as an R² difference. What the table supports is the ordinal contrast: extrastriate below V1 in 8/8 subjects (Δ = −0.252 [−0.283, −0.219], 10,000-sample percentile bootstrap over subjects, exact Wilcoxon p = .0078), reversed without the diet (+0.188 [+0.077, +0.285], 7/8 positive, p = .023). "Voxels magno / parvo" is the fraction of the ROI's voxels whose own best w is ≥ 0.75 / ≤ 0.25. The behaviour-optimal weight (trial fit to each of 10 human observers' cue-conflict choices, diet final checkpoints, 3 seeds) is given in the last row, with the cue-conflict shape-bias rate the diet network produces at each weight; it is not identified — the per-fit bootstrap intervals average 0.69 wide and cover 0.33 in 23 of 30 observer-runs, and 2 of 30 fits beat a label-permutation null — so no contrast with w*_cortex is tested (`results/fit_w_observers/`).

| arm | ROI | w*_cortex median [range] | R² at w = 0.5 / max | max R² (median) | concat R² | voxels best w ≥ 0.75 | voxels best w ≤ 0.25 | rate at w*_cortex | n |
|---|---|---|---|---|---|---|---|---|---|
| diet | V1 | 0.58 [0.50, 0.70] | 0.994 | 0.1313 | 0.1402 | 0.19 | 0.11 | 0.72 | 8 |
| diet | V2 | 0.50 [0.45, 0.55] | 0.999 | 0.1024 | 0.1066 | 0.12 | 0.20 | 0.67 | 8 |
| diet | V3 | 0.40 [0.30, 0.50] | 0.987 | 0.0847 | 0.0864 | 0.08 | 0.29 | 0.61 | 8 |
| diet | hV4 | 0.35 [0.20, 0.50] | 0.976 | 0.0945 | 0.0948 | 0.05 | 0.41 | 0.57 | 8 |
| diet | ventral (VO, PHC) | 0.33 [0.25, 0.35] | 0.971 | 0.1339 | 0.1325 | 0.02 | 0.46 | 0.56 | 8 |
| diet | lateral (LO, TO) | 0.33 [0.25, 0.45] | 0.974 | 0.1067 | 0.1060 | 0.03 | 0.40 | 0.56 | 8 |
| diet | dorsal (V3A/B, IPS0–2) | 0.35 [0.25, 0.45] | 0.982 | 0.0683 | 0.0673 | 0.05 | 0.39 | 0.57 | 8 |
| no diet | V1 | 0.65 [0.55, 0.80] | 0.987 | 0.1221 | 0.1259 | 0.43 | 0.09 | nan | 8 |
| no diet | V2 | 0.58 [0.40, 0.95] | 0.995 | 0.0968 | 0.1022 | 0.36 | 0.13 | nan | 8 |
| no diet | V3 | 0.58 [0.50, 1.00] | 0.991 | 0.0845 | 0.0890 | 0.40 | 0.13 | nan | 8 |
| no diet | hV4 | 0.62 [0.40, 0.70] | 0.988 | 0.0965 | 0.1014 | 0.40 | 0.09 | nan | 8 |
| no diet | ventral (VO, PHC) | 0.75 [0.65, 1.00] | 0.975 | 0.1557 | 0.1535 | 0.59 | 0.02 | nan | 8 |
| no diet | lateral (LO, TO) | 0.88 [0.60, 1.00] | 0.949 | 0.1190 | 0.1214 | 0.59 | 0.03 | nan | 8 |
| no diet | dorsal (V3A/B, IPS0–2) | 1.00 [0.75, 1.00] | 0.944 | 0.0805 | 0.0792 | 0.61 | 0.04 | nan | 8 |
| behaviour (diet) | 10 human observers, trial fit | 0.67 [0.49, 0.89] | — | — | — | — | — | 0.78 | 10 |

**Extended Data Table 3 | The readout span without a stream split: one gain on a network's own shape- and texture-predictive subspaces, head fixed.** For each frozen network with logits = W·y + b, the 15 discriminant directions of a 16-class shape LDA and of a 16-class texture LDA on the penultimate features y (PCA to 256 dimensions, Ledoit–Wolf shrinkage; fitted on one stratified half of the 1,200 cue-conflict images) are made mutually orthogonal by symmetric (Löwdin) orthogonalisation, giving orthonormal bases B_S and B_T (15 directions each) and the residual B_R. logits(w) = W·(P_R y + 2w·P_S y + 2(1 − w)·P_T y) + b with the network's own head untouched; w = 0.5 is the trained network. The rate is scored on the other half of the images (10 random splits, both directions; mean, with the range over splits and seeds in brackets), by the paper's primary estimator (112-class, decided images) for our networks and the Geirhos 16-class probability aggregation for the ImageNet networks; “forced-estimator span” is the same sweep on the forced two-way estimator. “gated span” is max − min of the seed-mean held-out curve over the w at which top-1 of the gained features (bases fitted on all 1,200 images) on held-out natural images (MiniEcoset validation for our networks; a stratified 10-per-class ILSVRC-2012 validation subset for the ImageNet networks) stays within 5 points of the trained value, followed by that w-range and the largest change of top-1 anywhere in the sweep. The two stratified halves share object and texture exemplars (98% and 100% of scored images; the 1,200 images reuse 160 object and 48 texture exemplars), so the span is also given with exemplars held out: fitted on images whose object and texture exemplars both lie outside one of three exemplar folds and scored on the images whose exemplars both lie inside it, ten repetitions, ungated (`docs/PLAN_EXEMPLAR_2026-09-21.md`; `results/figures_v2/exemplar_zoo.json`, `exemplar_streams.json`). “refit span” is the block-form lossy sum z(w) = [w·s + (1 − w)·t, B_R^T^y], s = B_S^T^y and t = B_T^T^y paired by discriminant rank, with a linear and a two-layer head refitted at each w (Methods 4.3 budget), our networks only; in parentheses the largest shortfall of validation accuracy, over w, from a head refitted on the unmixed features. “variance in S / T” is the fraction of the feature variance on the fitting half lying in each subspace; “order sensitivity” replaces the symmetric orthogonalisation by sequential Gram–Schmidt with the shape basis or the texture basis first. “median-split gated span” is the gated span of the pre-specified unit-level rule of Supplementary Table S11 on the same networks and images. For the two-stream concatenation, which has no held-out natural-image features, the fraction of the shape basis's energy lying on stream-1 units (all-image bases; held-out bases in parentheses; range over seeds) replaces the gate. Pre-specification: `docs/PLAN_SUBSPACE_2026-09-15.md` (predictions P-A, P-A′, P-B, P-D). Source: `results/figures_v2/subspace_behaviour.json`, aggregated by `tmp_analysis/figs_v2/subspace_behaviour.py` from the outputs of `src/analysis/generality/subspace_factor.py`, `gain_accuracy.py` and `refit_sum.py` (`slurm/subspace_{cc,refit}.sh`); the per-network files are outside the repository.

| network | D | trained rate | w = 0 | w = 1 | span, exemplars shared [range] | span, exemplars held out [seed range] | ρ(w, rate) | gated span (w-range; largest change of top-1) | forced-estimator span | refit span linear / two-layer (largest shortfall of top-1) | variance in S / T | order sensitivity, S first / T first | median-split gated span | seeds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| two-stream, diet, streams concatenated (validation) | 1024 | 0.675 | 0.495 | 0.801 | 0.320 [0.18, 0.45] | 0.249 [0.24, 0.27] | 0.98 | — (shape basis on stream 1: 0.63–0.70) | 0.216 | — | 0.11 / 0.29 | 0.35 / 0.25 | — | 3 |
| single ResNet-34, diet | 512 | 0.625 | 0.516 | 0.735 | 0.226 [0.12, 0.37] | 0.173 [0.14, 0.20] | 0.98 | 0.19 (0.10–0.90; 11.9 pts) | 0.163 | 0.08 / 0.10 (0.8 pts) | 0.14 / 0.20 | 0.24 / 0.20 | 0.07 | 3 |
| single ResNet-18, diet | 512 | 0.557 | 0.433 | 0.674 | 0.249 [0.13, 0.41] | 0.200 [0.18, 0.23] | 0.98 | 0.14 (0.25–0.73; 16.7 pts) | 0.189 | 0.08 / 0.10 (0.6 pts) | 0.15 / 0.23 | 0.25 / 0.20 | 0.09 | 3 |
| single ResNet-18, no diet (`rgb_tsn`) | 512 | 0.445 | 0.311 | 0.651 | 0.344 [0.22, 0.47] | 0.279 [0.23, 0.32] | 0.99 | 0.22 (0.20–0.78; 13.4 pts) | 0.242 | 0.13 / 0.15 (0.5 pts) | 0.13 / 0.26 | 0.29 / 0.34 | 0.08 | 3 |
| ImageNet ViT-B/16 | 768 | 0.399 | 0.287 | 0.570 | 0.282 [0.22, 0.34] | 0.180 | 1.00 | 0.28 (0.00–1.00; 0.2 pts) | 0.318 | — | 0.11 / 0.23 | 0.22 / 0.31 | 0.08 | 1 |
| ImageNet ResNet-152 | 2048 | 0.293 | 0.209 | 0.455 | 0.246 [0.20, 0.33] | 0.166 | 1.00 | 0.25 (0.00–1.00; 1.2 pts) | 0.271 | — | 0.07 / 0.24 | 0.21 / 0.27 | 0.16 | 1 |
| ImageNet ConvNeXt-B | 1024 | 0.269 | 0.178 | 0.512 | 0.334 [0.29, 0.39] | 0.196 | 1.00 | 0.33 (0.00–1.00; 0.2 pts) | 0.371 | — | 0.08 / 0.33 | 0.31 / 0.35 | 0.11 | 1 |
| ImageNet ConvNeXt-S | 768 | 0.248 | 0.170 | 0.458 | 0.287 [0.24, 0.34] | 0.156 | 1.00 | 0.29 (0.00–1.00; 0.1 pts) | 0.353 | — | 0.08 / 0.33 | 0.27 / 0.30 | 0.12 | 1 |
| ImageNet ResNet-50 | 2048 | 0.222 | 0.157 | 0.447 | 0.290 [0.21, 0.34] | 0.208 | 1.00 | 0.29 (0.00–1.00; 1.5 pts) | 0.310 | — | 0.07 / 0.28 | 0.28 / 0.29 | 0.13 | 1 |
| ImageNet VGG-16 | 4096 | 0.110 | 0.061 | 0.329 | 0.268 [0.23, 0.31] | 0.138 | 1.00 | 0.27 (0.00–1.00; 0.5 pts) | 0.322 | — | 0.04 / 0.28 | 0.26 / 0.27 | 0.09 | 1 |

**Extended Data Table 4 | The readout span across the field's model zoo.** Frozen ImageNet networks under the subspace rule of Extended Data Table 3 (PCA to 256 dimensions, 16-class shape and texture LDA, 15 directions each, symmetric orthogonalisation; bases fitted on one stratified half of the 1,200 cue-conflict images and the rate scored on the other, 10 splits both directions), each network's own head untouched, the Geirhos 16-class probability aggregation throughout; “images decided” is the number of the 1,200 assigned to their shape or their texture category at the trained readout, the denominator of the rate. “gate” is the w-range over which top-1 on a stratified 10-per-class ILSVRC-2012 validation subset (10,000 images; bases fitted on all cue-conflict images) stays within 5 points of the trained value, with the largest change of top-1 inside it; “texture end / shape end” are the held-out rates at the smallest and the largest gated w, and “gated span” is their difference, under the subspace rule and under the pre-specified unit-level median split of Supplementary Table S11. “probe” is the linear probe of Methods 4.3 (PCA to 256 dimensions, multinomial logistic regression, 5-fold stratified cross-validation) predicting the shape class and, separately, the texture class from the same features; bias is shape minus texture accuracy (chance 0.0625). The stratified halves and the probe's folds share object and texture exemplars (the 1,200 images reuse 160 object and 48 texture exemplars), so both are also given with exemplars held out (`docs/PLAN_EXEMPLAR_2026-09-21.md`; `results/figures_v2/exemplar_zoo.json`): “gated span, exemplars held out” fits the subspaces on images whose object and texture exemplars both lie outside one of three exemplar folds and scores the rate on the images whose exemplars both lie inside it (ten repetitions); “probe, exemplars held out” is the exemplar-balanced probe of Methods 4.3 (three exemplars per category for both labels, leave one exemplar out, PCA inside the folds, mean of 20 draws). With exemplars held out, over the 38 ImageNet-trained networks: gated span 0.08–0.29 (median 0.19), ≥ 0.20 in 11; 18.6% of the 703 pairs reverse between the trained ranking and the shape-end ranking (forced two-way 16.2%), against 8.9% [6.0%, 12.2%] between two bootstrap resamples of the images at the trained readout; probe bias -0.52 to +0.01 (median -0.36), the 5–95% interval over draws below zero in 37. Preprocessing is Resize 256 / CenterCrop 224 except for Inception-v3 (342/299), EfficientNet-B3 (320/300) and EfficientNet-V2-S (384/384), which were not trained at 224. “κ with observers” is error consistency (Cohen's κ on trial-by-trial agreement, Results 2.5; Supplementary Table S6) with each of the ten human observers, mean over observers, as a percentage of the human–human ceiling κ = 0.234, at the trained readout and, in parentheses, at the texture end and the shape end of the gate under the same gain; computed under this pipeline's preprocessing, so the VGG-16 value differs from the published response file's 7.3% (Supplementary Table S6) as its trained rate does. Over the 38 ImageNet-trained networks: Kendall τ between the ranking by trained rate and by the shape-end rate 0.47 (by the texture-end rate 0.78); 188 of 703 pairs (27%) reverse order; gated span 0.20–0.50 (median 0.30), ≥ 0.20 in 97% and the whole sweep inside the gate in 100%; Spearman ρ between gated span and trained rate +0.04, and trained top-1 -0.22; error consistency at the trained readout 2.4–23.0% of the ceiling (median 7.9%), at most 36.7% anywhere inside a gate, changing by a median of +13.5 points from the texture end to the shape end; ρ(κ, trained rate) +0.80; Kendall τ between trained rate and probe bias 0.44. The three Stylized-ImageNet networks are the published checkpoints of Geirhos et al. (2018a), all ResNet-50: trained on Stylized-ImageNet (SIN), on SIN + IN, and on SIN + IN then fine-tuned on IN. Pre-specification: `docs/PLAN_ZOO_2026-09-15.md` (predictions P-Z1–P-Z4). Source: `results/figures_v2/zoo_behaviour.json`, aggregated by `tmp_analysis/figs_v2/zoo_behaviour.py` from the outputs of `slurm/zoo_cc.sh`; the per-network files are outside the repository.

| network | D | top-1 (subset) | trained rate (images decided) | gate: w-range (largest Δtop-1) | rate at texture end / shape end | gated span, subspace rule | gated span, exemplars held out | ρ(w, rate) | gated span, unit rule | probe: shape / texture accuracy (bias) | probe, exemplars held out: shape / texture (bias) | κ with observers, % of ceiling: trained (texture end → shape end) | whole sweep in gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **ImageNet-trained (torchvision IMAGENET1K_V1)** | | | | | | | | | | | | | |
| ViT-B/32 | 768 | 0.758 | 0.617 (767) | 0.00–1.00 (0.2 pts) | 0.499 / 0.738 | 0.239 | 0.161 | 1.00 | 0.069 | 0.67 / 0.97 (-0.30) | 0.46 / 0.46 (+0.01) | 23.0 (12.9 → 34.0) | yes |
| ViT-B/16 | 768 | 0.815 | 0.399 (818) | 0.00–1.00 (0.2 pts) | 0.287 / 0.570 | 0.282 | 0.180 | 1.00 | 0.085 | 0.56 / 0.99 (-0.43) | 0.38 / 0.57 (-0.19) | 11.6 (7.2 → 19.2) | yes |
| ViT-L/16 | 1024 | 0.801 | 0.319 (885) | 0.00–1.00 (0.2 pts) | 0.207 / 0.545 | 0.337 | 0.193 | 1.00 | 0.078 | 0.57 / 0.99 (-0.43) | 0.35 / 0.58 (-0.23) | 11.5 (5.9 → 21.9) | yes |
| Inception-v3 | 2048 | 0.759 | 0.305 (797) | 0.00–1.00 (0.6 pts) | 0.241 / 0.487 | 0.246 | 0.134 | 0.99 | 0.099 | 0.40 / 0.96 (-0.56) | 0.29 / 0.63 (-0.35) | 11.0 (5.5 → 22.5) | yes |
| MaxViT-T | 512 | 0.842 | 0.295 (899) | 0.00–1.00 (0.2 pts) | 0.230 / 0.469 | 0.239 | 0.142 | 1.00 | 0.114 | 0.40 / 0.97 (-0.57) | 0.22 / 0.60 (-0.38) | 9.1 (4.5 → 20.7) | yes |
| ResNeXt-101 32x8d | 2048 | 0.796 | 0.294 (732) | 0.00–1.00 (0.6 pts) | 0.228 / 0.511 | 0.283 | 0.184 | 0.99 | 0.220 | 0.47 / 1.00 (-0.53) | 0.30 / 0.61 (-0.32) | 7.9 (3.9 → 16.3) | yes |
| ConvNeXt-L | 1536 | 0.848 | 0.293 (924) | 0.00–1.00 (0.1 pts) | 0.196 / 0.528 | 0.332 | 0.189 | 1.00 | 0.079 | 0.51 / 1.00 (-0.49) | 0.28 / 0.66 (-0.38) | 9.2 (4.4 → 18.1) | yes |
| ResNet-152 | 2048 | 0.785 | 0.293 (781) | 0.00–1.00 (1.2 pts) | 0.209 / 0.455 | 0.246 | 0.165 | 1.00 | 0.159 | 0.51 / 1.00 (-0.49) | 0.31 / 0.64 (-0.33) | 9.1 (2.1 → 16.0) | yes |
| EfficientNet-B3 | 1536 | 0.818 | 0.291 (905) | 0.00–1.00 (0.2 pts) | 0.204 / 0.514 | 0.310 | 0.191 | 1.00 | 0.141 | 0.49 / 0.99 (-0.50) | 0.34 / 0.74 (-0.39) | 12.5 (6.8 → 26.0) | yes |
| ResNet-101 | 2048 | 0.777 | 0.284 (770) | 0.00–1.00 (1.4 pts) | 0.216 / 0.493 | 0.276 | 0.206 | 1.00 | 0.143 | 0.48 / 1.00 (-0.52) | 0.30 / 0.60 (-0.30) | 8.7 (5.1 → 14.8) | yes |
| Wide-ResNet-101-2 | 2048 | 0.790 | 0.279 (746) | 0.00–1.00 (0.7 pts) | 0.201 / 0.431 | 0.231 | 0.155 | 1.00 | 0.119 | 0.39 / 0.99 (-0.60) | 0.26 / 0.57 (-0.31) | 7.9 (3.4 → 18.5) | yes |
| DenseNet-201 | 1920 | 0.764 | 0.273 (792) | 0.00–1.00 (1.5 pts) | 0.206 / 0.540 | 0.334 | 0.192 | 1.00 | 0.085 | 0.51 / 1.00 (-0.49) | 0.31 / 0.59 (-0.28) | 8.5 (3.0 → 18.2) | yes |
| EfficientNet-B0 | 1280 | 0.777 | 0.271 (835) | 0.00–1.00 (0.4 pts) | 0.175 / 0.502 | 0.327 | 0.193 | 1.00 | 0.110 | 0.45 / 0.99 (-0.54) | 0.31 / 0.67 (-0.36) | 7.1 (3.6 → 19.0) | yes |
| ConvNeXt-B | 1024 | 0.843 | 0.269 (921) | 0.00–1.00 (0.2 pts) | 0.178 / 0.512 | 0.334 | 0.195 | 1.00 | 0.114 | 0.53 / 1.00 (-0.47) | 0.25 / 0.70 (-0.45) | 8.0 (2.1 → 21.6) | yes |
| MobileNet-V3-L | 1280 | 0.741 | 0.265 (796) | 0.00–1.00 (1.2 pts) | 0.156 / 0.591 | 0.435 | 0.263 | 1.00 | 0.102 | 0.53 / 1.00 (-0.47) | 0.33 / 0.63 (-0.29) | 10.8 (4.0 → 29.0) | yes |
| AlexNet | 4096 | 0.552 | 0.261 (735) | 0.00–1.00 (1.4 pts) | 0.120 / 0.624 | 0.504 | 0.293 | 1.00 | 0.142 | 0.55 / 1.00 (-0.45) | 0.34 / 0.58 (-0.24) | 7.1 (3.0 → 14.3) | yes |
| RegNet-Y-16GF | 3024 | 0.808 | 0.261 (797) | 0.00–1.00 (0.6 pts) | 0.180 / 0.480 | 0.300 | 0.174 | 1.00 | 0.208 | 0.50 / 1.00 (-0.50) | 0.33 / 0.63 (-0.30) | 7.7 (4.6 → 15.3) | yes |
| ResNet-34 | 512 | 0.732 | 0.259 (791) | 0.00–1.00 (2.7 pts) | 0.175 / 0.528 | 0.353 | 0.227 | 1.00 | 0.084 | 0.47 / 1.00 (-0.53) | 0.24 / 0.56 (-0.31) | 6.1 (3.5 → 15.9) | yes |
| GoogLeNet | 1024 | 0.691 | 0.251 (876) | 0.00–1.00 (1.4 pts) | 0.187 / 0.483 | 0.296 | 0.189 | 1.00 | 0.056 | 0.47 / 0.99 (-0.52) | 0.29 / 0.61 (-0.32) | 9.5 (5.4 → 22.1) | yes |
| ConvNeXt-S | 768 | 0.840 | 0.248 (901) | 0.00–1.00 (0.1 pts) | 0.170 / 0.458 | 0.287 | 0.155 | 1.00 | 0.124 | 0.47 / 1.00 (-0.53) | 0.23 / 0.67 (-0.44) | 8.1 (3.6 → 16.9) | yes |
| RegNet-Y-8GF | 2016 | 0.802 | 0.244 (788) | 0.00–1.00 (0.8 pts) | 0.151 / 0.516 | 0.365 | 0.241 | 1.00 | 0.140 | 0.50 / 1.00 (-0.50) | 0.31 / 0.67 (-0.36) | 6.7 (1.8 → 15.9) | yes |
| ConvNeXt-T | 768 | 0.830 | 0.241 (924) | 0.00–1.00 (0.4 pts) | 0.154 / 0.507 | 0.353 | 0.200 | 1.00 | 0.125 | 0.50 / 1.00 (-0.50) | 0.29 / 0.71 (-0.42) | 7.0 (2.4 → 22.3) | yes |
| Swin-S | 768 | 0.836 | 0.236 (880) | 0.00–1.00 (0.2 pts) | 0.154 / 0.440 | 0.287 | 0.169 | 1.00 | 0.077 | 0.44 / 0.99 (-0.55) | 0.25 / 0.68 (-0.42) | 8.8 (5.3 → 16.6) | yes |
| ShuffleNet-V2 x1.0 | 1024 | 0.686 | 0.234 (781) | 0.00–1.00 (2.5 pts) | 0.146 / 0.527 | 0.381 | 0.245 | 1.00 | 0.034 | 0.57 / 1.00 (-0.43) | 0.34 / 0.60 (-0.26) | 9.2 (4.0 → 25.7) | yes |
| Swin-B | 1024 | 0.839 | 0.233 (891) | 0.00–1.00 (0.1 pts) | 0.177 / 0.406 | 0.229 | 0.128 | 1.00 | 0.071 | 0.42 / 1.00 (-0.57) | 0.23 / 0.63 (-0.40) | 7.7 (4.0 → 17.0) | yes |
| ResNet-18 | 512 | 0.695 | 0.233 (849) | 0.00–1.00 (3.8 pts) | 0.163 / 0.479 | 0.316 | 0.195 | 1.00 | 0.050 | 0.47 / 0.99 (-0.53) | 0.25 / 0.59 (-0.34) | 6.3 (3.2 → 16.2) | yes |
| Wide-ResNet-50-2 | 2048 | 0.785 | 0.230 (792) | 0.00–1.00 (0.6 pts) | 0.166 / 0.427 | 0.261 | 0.177 | 0.99 | 0.105 | 0.38 / 0.99 (-0.62) | 0.23 / 0.60 (-0.37) | 5.0 (2.4 → 11.6) | yes |
| MNASNet-1.0 | 1280 | 0.734 | 0.226 (766) | 0.00–1.00 (1.3 pts) | 0.134 / 0.530 | 0.395 | 0.251 | 1.00 | 0.042 | 0.47 / 1.00 (-0.53) | 0.29 / 0.66 (-0.37) | 8.7 (3.7 → 19.7) | yes |
| Swin-T | 768 | 0.820 | 0.225 (902) | 0.00–1.00 (0.2 pts) | 0.150 / 0.460 | 0.309 | 0.165 | 1.00 | 0.084 | 0.48 / 1.00 (-0.52) | 0.29 / 0.68 (-0.39) | 8.4 (4.2 → 18.4) | yes |
| DenseNet-121 | 1024 | 0.740 | 0.224 (839) | 0.00–1.00 (1.9 pts) | 0.147 / 0.494 | 0.347 | 0.211 | 1.00 | 0.071 | 0.45 / 1.00 (-0.55) | 0.23 / 0.56 (-0.33) | 5.7 (3.7 → 14.9) | yes |
| ResNet-50 | 2048 | 0.760 | 0.222 (814) | 0.00–1.00 (1.5 pts) | 0.157 / 0.447 | 0.290 | 0.203 | 1.00 | 0.130 | 0.43 / 1.00 (-0.56) | 0.29 / 0.63 (-0.34) | 5.1 (2.7 → 12.2) | yes |
| ResNeXt-50 32x4d | 2048 | 0.774 | 0.219 (776) | 0.00–1.00 (0.9 pts) | 0.145 / 0.461 | 0.316 | 0.201 | 1.00 | 0.193 | 0.41 / 0.99 (-0.58) | 0.25 / 0.65 (-0.39) | 5.6 (2.1 → 14.1) | yes |
| EfficientNet-V2-S | 1280 | 0.832 | 0.184 (879) | 0.00–1.00 (0.2 pts) | 0.134 / 0.332 | 0.198 | 0.084 | 1.00 | 0.074 | 0.35 / 0.99 (-0.64) | 0.22 / 0.74 (-0.52) | 5.6 (3.6 → 8.1) | yes |
| MobileNet-V2 | 1280 | 0.715 | 0.184 (798) | 0.00–1.00 (4.4 pts) | 0.128 / 0.474 | 0.346 | 0.186 | 1.00 | 0.051 | 0.48 / 1.00 (-0.52) | 0.27 / 0.65 (-0.38) | 5.6 (1.9 → 18.0) | yes |
| VGG-19-BN | 4096 | 0.743 | 0.150 (867) | 0.00–1.00 (0.8 pts) | 0.094 / 0.335 | 0.241 | 0.149 | 1.00 | 0.093 | 0.37 / 0.99 (-0.62) | 0.27 / 0.70 (-0.43) | 4.0 (2.2 → 6.8) | yes |
| VGG-16-BN | 4096 | 0.733 | 0.115 (915) | 0.00–1.00 (0.7 pts) | 0.068 / 0.294 | 0.226 | 0.130 | 1.00 | 0.104 | 0.36 / 1.00 (-0.64) | 0.24 / 0.70 (-0.45) | 2.4 (2.9 → 5.2) | yes |
| VGG-16 | 4096 | 0.711 | 0.110 (918) | 0.00–1.00 (0.5 pts) | 0.061 / 0.329 | 0.268 | 0.137 | 1.00 | 0.093 | 0.33 / 0.99 (-0.66) | 0.23 / 0.72 (-0.50) | 3.4 (2.3 → 6.4) | yes |
| VGG-11-BN | 4096 | 0.703 | 0.105 (927) | 0.00–1.00 (1.0 pts) | 0.059 / 0.294 | 0.235 | 0.110 | 1.00 | 0.092 | 0.42 / 1.00 (-0.57) | 0.27 / 0.69 (-0.42) | 4.2 (2.3 → 4.8) | yes |
| **same architecture, different training recipe** | | | | | | | | | | | | | |
| ResNet-50 (V2 recipe) | 2048 | 0.814 | 0.185 (874) | 0.00–1.00 (0.2 pts) | 0.131 / 0.465 | 0.334 | 0.195 | 1.00 | 0.056 | 0.38 / 1.00 (-0.62) | 0.26 / 0.72 (-0.47) | 0.5 (1.9 → 7.8) | yes |
| **Stylized-ImageNet controls (Geirhos et al., 2018a)** | | | | | | | | | | | | | |
| ResNet-50 SIN | 2048 | 0.582 | 0.810 (758) | 0.00–1.00 (3.9 pts) | 0.701 / 0.867 | 0.166 | 0.124 | 0.99 | 0.140 | 0.79 / 0.90 (-0.11) | 0.54 / 0.30 (+0.23) | 39.5 (28.7 → 49.7) | yes |
| ResNet-50 SIN+IN | 2048 | 0.743 | 0.363 (802) | 0.00–1.00 (2.4 pts) | 0.256 / 0.606 | 0.350 | 0.238 | 1.00 | 0.170 | 0.59 / 0.99 (-0.40) | 0.39 / 0.53 (-0.14) | 11.7 (7.1 → 18.5) | yes |
| ResNet-50 SIN+IN, ft IN | 2048 | 0.770 | 0.227 (797) | 0.00–1.00 (1.9 pts) | 0.170 / 0.447 | 0.277 | 0.175 | 1.00 | 0.154 | 0.41 / 0.99 (-0.58) | 0.26 / 0.60 (-0.33) | 5.9 (4.6 → 14.7) | yes |

**Supplementary Table S1 | Monotonicity of the profile and the exact permutation null.** Spearman ρ between hierarchical rank [1,1,2,2,3,3,4] and stream preference, mean over subjects; the null enumerates all 630 distinct labellings of the seven ROIs (one-sided in the pre-stated direction, floor 1/630 = .0016). “per-voxel index” is the manuscript's statistic (Fig. 3c; Results 2.3): the median over reliable voxels of (U_m − U_p)/(U_m + U_p) per ROI, as in Extended Data Table 1; “ratio of medians” is median(U_m − U_p)/median(U_m + U_p) per ROI, the statistic of an earlier draft, kept for comparison. Source: `results/hierarchy_permutation_null.json` (which labels the two `raw` and `norm`).

| arm | seed | statistic | observed mean ρ | exact p | most extreme of 630 | null 2.5th pct |
|---|---|---|---|---|---|---|
| diet | 0 | per-voxel index | -0.964 | 0.0016 | yes | -0.743 |
| diet | 0 | ratio of medians | -0.964 | 0.0016 | yes | -0.754 |
| diet | 1 | per-voxel index | -0.973 | 0.0016 | yes | -0.743 |
| diet | 1 | ratio of medians | -0.973 | 0.0016 | yes | -0.750 |
| diet | 2 | per-voxel index | -0.966 | 0.0016 | yes | -0.756 |
| diet | 2 | ratio of medians | -0.966 | 0.0016 | yes | -0.748 |
| no diet | 0 | per-voxel index | -0.571 | 0.0206 | no | -0.559 |
| no diet | 0 | ratio of medians | -0.502 | 0.0286 | no | -0.502 |
| no diet | 1 | per-voxel index | -0.661 | 0.0190 | no | -0.594 |
| no diet | 1 | ratio of medians | -0.606 | 0.0190 | no | -0.546 |
| no diet | 2 | per-voxel index | -0.463 | 0.0921 | no | -0.594 |
| no diet | 2 | ratio of medians | -0.580 | 0.0365 | no | -0.616 |

**Supplementary Table S2 | In-silico cue conflict through the fitted encoding models, per seed.** Median over reliable voxels of the per-voxel predicted shape-bias index for each stream, mean over 8 subjects; the contrast is magno − parvo, paired within subject against the no-diet arm of the same seed. Source: `results/encoding_cc/`.

| seed | arm | magno stream | parvo stream | magno − parvo | diet − no diet | sep. | p |
|---|---|---|---|---|---|---|---|
| 0 | diet | +0.791 | −0.743 | +1.534 | +1.424 | 8/8 | 0.0078 |
| 0 | no diet | −0.638 | −0.748 | +0.110 | | | |
| 1 | diet | +0.796 | −0.770 | +1.566 | +1.468 | 8/8 | 0.0078 |
| 1 | no diet | −0.601 | −0.699 | +0.098 | | | |
| 2 | diet | +0.799 | −0.759 | +1.558 | +1.474 | 8/8 | 0.0078 |
| 2 | no diet | −0.613 | −0.696 | +0.083 | | | |

**Supplementary Table S3 | The carrier × transform 2×2 on the probe and on behaviour.** Mean over training seeds with the seed range in brackets; “decided” is the mean number of the 1,200 cue-conflict images the readout classifies as the shape or the texture category (the shape-bias denominator). Cells are read out on stream 2 in three cases and on the diet's magno stream in the fourth. Source: `results/figures/fig1_data.npz` (from `results/shape_texture/`, `results/cue_conflict/`). “probe bias, exemplars held out” is the exemplar-balanced probe of Methods 4.3, scored on object and texture exemplars the probe was not trained on (three exemplars per category for both labels, leave one exemplar out, PCA inside the folds, mean of 20 draws; `docs/PLAN_EXEMPLAR_2026-09-21.md`, `results/figures_v2/exemplar_streams.json`); “probe bias” is the original stratified-fold probe, whose test images share exemplars with its training folds.

| carrier | transform | condition, stream | probe bias | probe bias, exemplars held out | cue-conflict shape bias | decided / 1,200 | seeds |
|---|---|---|---|---|---|---|---|
| chromatic (L−M) | none | no diet, parvo | −0.504 [0.035] | −0.118 [0.021] | 0.364 [0.080] | 270 | 3 |
| chromatic (L−M) | yes | swap, parvo | −0.145 [0.043] | −0.002 [0.022] | 0.495 [0.038] | 165 | 3 |
| achromatic (luminance) | none | parvo_highsf, stream 2 | −0.271 [0.023] | −0.139 [0.012] | 0.389 [0.028] | 598 | 3 |
| achromatic (luminance) | yes | diet, magno | +0.819 [0.014] | +0.337 [0.007] | 0.878 [0.022] | 488 | 3 |

**Supplementary Table S4 | Single-stream controls trained with the identical diet, and the readout ladder.** Best saved checkpoint per seed, except the no-diet row, which is scored at the final checkpoint because its accuracy does not degrade (Fig. 2d; Methods 4.7); mean [range] over three seeds. Accuracy is validation top-1 at the scored checkpoint. Sources: `results/cue_conflict_bestep/`, `results/cue_conflict_single34_bestep/`, `results/cue_conflict_single_bestep/`, run histories, `results/param_matched/`.

| model / readout | parameters | shape bias | decided / 1,200 | accuracy |
|---|---|---|---|---|
| diet, magno stream alone | 22,397,936 | 0.903 [0.888, 0.916] | 504 | 0.556 [0.552, 0.564] |
| diet, both streams (w = 0.5) | 22,397,936 | 0.655 [0.594, 0.718] | 502 | 0.556 [0.552, 0.564] |
| diet, parvo stream alone | 22,397,936 | 0.400 [0.372, 0.429] | 388 | 0.556 [0.552, 0.564] |
| single stream, ResNet-34 (matched) | 21,338,992 | 0.625 [0.604, 0.652] | 537 | 0.623 [0.616, 0.634] |
| single stream, ResNet-18 (half) | 11,230,832 | 0.567 [0.541, 0.590] | 501 | 0.599 [0.589, 0.608] |
| no diet, both streams | 22,397,936 | 0.416 [0.394, 0.433] | 593 | 0.701 [0.698, 0.706] |

**Supplementary Table S5 | Out-of-distribution robustness: relative-accuracy AUC per corruption family and condition.** Mean over three seeds [range]; the thesis's schedule-trained checkpoint is a single run. “sep.” marks complete seed separation between the diet and no-diet arms (the exact 3 v 3 floor, p = 0.100). Families marked ‡ were declared trained-in before the run. Source: `results/ood_bestep/`.

| family | diet | no diet | swap | parvo_highsf | thesis schedule | diet − no diet | sep. |
|---|---|---|---|---|---|---|---|
| uniform-noise | 0.634 [0.619, 0.647] | 0.729 [0.720, 0.735] | 0.749 [0.744, 0.758] | 0.767 [0.758, 0.773] | 0.640 | −0.095 | yes |
| high-pass | 0.678 [0.647, 0.696] | 0.827 [0.820, 0.831] | 0.831 [0.817, 0.841] | 0.783 [0.775, 0.791] | 0.680 | −0.149 | yes |
| false-colour | 0.854 [0.845, 0.860] | 0.938 [0.933, 0.943] | 0.966 [0.960, 0.971] | 0.874 [0.868, 0.880] | 0.839 | −0.084 | yes |
| rotation | 0.574 [0.567, 0.582] | 0.615 [0.613, 0.616] | 0.605 [0.597, 0.612] | 0.611 [0.602, 0.616] | 0.572 | −0.041 | yes |
| phase-scrambling | 0.382 [0.365, 0.391] | 0.373 [0.369, 0.377] | 0.371 [0.365, 0.375] | 0.382 [0.380, 0.383] | 0.362 | +0.009 | no |
| contrast ‡ | 0.481 [0.460, 0.493] | 0.549 [0.530, 0.578] | 0.543 [0.539, 0.550] | 0.511 [0.498, 0.526] | 0.489 | −0.068 | yes |
| low-pass ‡ | 0.444 [0.434, 0.464] | 0.356 [0.353, 0.359] | 0.359 [0.353, 0.362] | 0.385 [0.375, 0.392] | 0.436 | +0.089 | yes |

**Supplementary Table S6 | Error consistency (Cohen's κ) with the ten human observers.** The trained readout (w = 0.5), the magno stream alone (w = 1; the κ = 0.0082, 3.5% of Results 2.5 is the mean of the three final-checkpoint runs) and the best readout over the 21 values of w, per run; κ as a percentage of the human–human ceiling (κ = 0.234, 45 pairs). Published CNN values are from Geirhos et al. (2020). Source: `results/sweep_error_consistency/`.

| run | shape bias at w = 0.5 | κ at w = 0.5 | % ceiling | κ at w = 1 (magno alone) | % ceiling | rate at w = 1 | best κ over w | % ceiling | w at best | rate at best |
|---|---|---|---|---|---|---|---|---|---|---|
| dkl_base_seed0_final | 0.421 | +0.0309 | 13.2 | +0.0404 | 17.3 | 0.438 | +0.0404 | 17.3 | 1.00 | 0.438 |
| dkl_base_seed1_final | 0.433 | +0.0333 | 14.3 | +0.0390 | 16.7 | 0.431 | +0.0421 | 18.0 | 0.75 | 0.429 |
| dkl_base_seed2_final | 0.394 | +0.0372 | 15.9 | +0.0349 | 15.0 | 0.400 | +0.0412 | 17.6 | 0.35 | 0.410 |
| dkl_bcs_seed0_best | 0.594 | +0.0096 | 4.1 | +0.0178 | 7.6 | 0.890 | +0.0178 | 7.6 | 1.00 | 0.890 |
| dkl_bcs_seed0_final | 0.692 | +0.0234 | 10.0 | +0.0361 | 15.5 | 0.872 | +0.0405 | 17.3 | 0.90 | 0.857 |
| dkl_bcs_seed1_best | 0.718 | +0.0269 | 11.5 | +0.0058 | 2.5 | 0.913 | +0.0425 | 18.2 | 0.35 | 0.614 |
| dkl_bcs_seed1_final | 0.686 | +0.0319 | 13.7 | +0.0028 | 1.2 | 0.894 | +0.0437 | 18.7 | 0.45 | 0.651 |
| dkl_bcs_seed2_best | 0.652 | +0.0372 | 15.9 | -0.0055 | -2.3 | 0.908 | +0.0483 | 20.7 | 0.85 | 0.847 |
| dkl_bcs_seed2_final | 0.646 | +0.0280 | 12.0 | -0.0143 | -6.1 | 0.871 | +0.0405 | 17.3 | 0.75 | 0.784 |
| AlexNet (published) | 0.429 | +0.0320 | 13.7 | | | | | | | |
| GoogLeNet (published) | 0.312 | +0.0250 | 10.7 | | | | | | | |
| VGG-16 (published) | 0.172 | +0.0170 | 7.3 | | | | | | | |

**Supplementary Table S7 | The estimator's own bias, and the residual-misfit variants.** Synthetic voxels were generated from each arm's fitted 1,000-component feature spaces with the measured per-ROI noise ceilings imposed and the true magno − parvo difference set to zero everywhere; the recovered contrast is the instrument's signed floor. “Displacement” is the real contrast minus the floor, paired per subject. The six variants differ in how the floor is paired (group or per-subject) and whether the residual misfit of the recovery simulation is also removed (before or after rescaling by the recovered slope). Sources: `results/vp_recovery_realnc/diet_floor_paired_2026-09-07.txt`, `floor_pairing_2026-09-07.txt`, `prediction_scored_2026-09-07.txt`.

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

**Supplementary Table S8 | Stream preference regressed on pRF eccentricity and pRF size within subject.** Per-voxel stream-preference index regressed on log pRF eccentricity and log pRF size (NSD pRF estimates, voxels with pRF R² > 5%, noise ceiling > 0.1); “ecc | size” is the partial slope for eccentricity controlling for size and vice versa; mean over 8 subjects, with the number of subjects on the stated side and the exact Wilcoxon p. Collinearity between the two predictors is modest (mean VIF 1.66). The arm interaction is the paired difference of partial slopes, seed-averaged within arm. Sources: `results/prf_continuous/dissociation_2026-09-08.txt`, `interaction_2026-09-08.txt`, `input_null_2026-09-08.txt`. The paired trained − untrained rows compare each subject's partial slope in the trained seed-0 network with the untrained network on the diet front end; training with the diet changes neither statistic significantly and the two disagree in sign, while training without the diet removes the dissociation on the raw statistic.

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
| untrained network, undieted front end | normalised | | | +0.089 (7/8 pos, .023) | −0.038 (8/8 neg, .0078) |
| trained diet − untrained (diet front end), paired | normalised | | | +0.036 (6/8 pos, .11) | |
| trained diet − untrained (diet front end), paired | raw | | | −0.077 (2/8 pos, .11) | |
| trained no diet − untrained (diet front end), paired | normalised | | | −0.087 (2/8 pos, .15) | |
| trained no diet − untrained (diet front end), paired | raw | | | −0.374 (0/8 pos, .0078) | |
| arm interaction, diet − no diet (seed-averaged) | normalised | | | +0.134 (8/8, .0078) | −0.119 (0/8 pos, .0078) |
| arm interaction, diet − no diet (seed-averaged) | raw | | | +0.305 (8/8, .0078) | −0.502 (0/8 pos, .0078) |

**Supplementary Table S9 | Cue-conflict shape bias under the forced two-way estimator.** The primary estimator (Methods 4.3) restricts the logits to the 16 categories, takes the argmax, and scores shape against texture over the images decided as one of the two, so images decided as a third category drop out of the denominator (“decided”). The forced estimator keeps all 1,200 images and scores, per image, whether the logit of the shape category exceeds the logit of the texture category. Mean over seeds [range]. Source: the stored per-image logits in `results/cue_conflict*/`.

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
| no diet, accuracy-matched (epochs 10, 10, 11), magno alone | 0.450 [0.432, 0.465] | 533 | 0.364 [0.357, 0.370] | 3 |
| thesis checkpoint (progressive schedule), both streams | 0.697 | 620 | 0.636 | 1 |
| untrained, diet front end, magno alone | 0.513 | 154 | 0.510 | 1 |
| untrained, no-diet front end, magno alone | 0.568 | 169 | 0.508 | 1 |

**Supplementary Table S10 | Stream preference beyond hV4, on every NSD atlas.** Median over reliable voxels of the per-voxel index (Um − Up)/(Um + Up) (noise ceiling > 0.1, Um + Up > 10⁻³, ≥ 20 voxels per ROI per subject), mean over the eight subjects (n = subjects with enough voxels); the no-diet, diet, swap and parvo + high-SF arms are each the mean over three training seeds per subject, and the thesis column is the thesis's one checkpoint (Methods 4.5). Positive = magno band, negative = parvo band. Atlases: Wang et al. (2015) Kastner2015 ROIs beyond hV4 (the V1–hV4 rows are Extended Data Table 1 and Fig. 3), NSD's `streams` atlas, and the fLoc category-selective ROIs. Regional contrasts and the exact tests are in Extended Data Fig. 4c and `results/figures_v2/ext_gradient.md`. Source: the stored voxelwise fits in `results/encoding_cc/` and `results/encoding/`, relabelled; no refitting.

| atlas | ROI | untrained (diet front end) | no diet (3 seeds) | diet (3 seeds) | swap (3 seeds) | parvo + high-SF (3 seeds) | thesis ckpt (n = 1) | voxels (median, diet) |
|---|---|---|---|---|---|---|---|---|
| Kastner2015 | VO1 | +0.00 (n=8) | +0.92 (n=8) | −0.87 (n=8) | +0.98 (n=8) | −0.99 (n=8) | −0.94 (n=8) | 196 |
| Kastner2015 | VO2 | +0.12 (n=8) | +0.97 (n=8) | −0.90 (n=8) | +1.00 (n=8) | −1.00 (n=8) | −0.94 (n=8) | 220 |
| Kastner2015 | PHC1 | +0.41 (n=8) | +0.98 (n=8) | −0.89 (n=8) | +1.00 (n=8) | −1.00 (n=8) | −0.94 (n=8) | 148 |
| Kastner2015 | PHC2 | +0.46 (n=8) | +0.99 (n=8) | −0.87 (n=8) | +1.00 (n=8) | −1.00 (n=8) | −0.93 (n=8) | 150 |
| Kastner2015 | LO1 | +0.13 (n=8) | +0.92 (n=8) | −0.83 (n=8) | +0.98 (n=8) | −0.98 (n=8) | −0.90 (n=8) | 230 |
| Kastner2015 | LO2 | −0.14 (n=8) | +0.92 (n=8) | −0.90 (n=8) | +0.98 (n=8) | −1.00 (n=8) | −0.94 (n=8) | 166 |
| Kastner2015 | TO1 | +0.25 (n=8) | +0.96 (n=8) | −0.86 (n=8) | +0.99 (n=8) | −1.00 (n=8) | −0.91 (n=8) | 280 |
| Kastner2015 | TO2 | +0.20 (n=8) | +0.96 (n=8) | −0.89 (n=8) | +0.99 (n=8) | −1.00 (n=8) | −0.90 (n=8) | 36 |
| Kastner2015 | V3A | +0.39 (n=8) | +0.91 (n=8) | −0.81 (n=8) | +0.98 (n=8) | −0.98 (n=8) | −0.84 (n=8) | 278 |
| Kastner2015 | V3B | +0.24 (n=8) | +0.96 (n=8) | −0.87 (n=8) | +0.99 (n=8) | −0.99 (n=8) | −0.94 (n=8) | 254 |
| Kastner2015 | IPS0 | +0.15 (n=8) | +0.98 (n=8) | −0.87 (n=8) | +1.00 (n=8) | −1.00 (n=8) | −0.92 (n=8) | 317 |
| Kastner2015 | IPS1 | −0.06 (n=8) | +0.98 (n=8) | −0.86 (n=8) | +1.00 (n=8) | −0.99 (n=8) | −0.92 (n=8) | 239 |
| Kastner2015 | IPS2 | +0.02 (n=8) | +1.00 (n=8) | −0.92 (n=8) | +1.01 (n=8) | −1.00 (n=8) | −0.94 (n=8) | 47 |
| streams | early | +0.27 (n=8) | +0.86 (n=8) | −0.42 (n=8) | +0.96 (n=8) | −0.93 (n=8) | −0.45 (n=8) | 3133 |
| streams | midventral | −0.15 (n=8) | +0.87 (n=8) | −0.84 (n=8) | +0.97 (n=8) | −0.99 (n=8) | −0.86 (n=8) | 708 |
| streams | midlateral | −0.04 (n=8) | +0.91 (n=8) | −0.86 (n=8) | +0.97 (n=8) | −0.99 (n=8) | −0.90 (n=8) | 571 |
| streams | midparietal | +0.32 (n=8) | +0.94 (n=8) | −0.84 (n=8) | +0.99 (n=8) | −0.99 (n=8) | −0.91 (n=8) | 512 |
| streams | ventral | +0.08 (n=8) | +0.96 (n=8) | −0.91 (n=8) | +0.99 (n=8) | −1.00 (n=8) | −0.95 (n=8) | 2724 |
| streams | lateral | +0.25 (n=8) | +0.97 (n=8) | −0.87 (n=8) | +1.00 (n=8) | −1.00 (n=8) | −0.91 (n=8) | 1842 |
| streams | parietal | +0.12 (n=8) | +0.99 (n=8) | −0.87 (n=8) | +1.00 (n=8) | −1.00 (n=8) | −0.93 (n=8) | 1082 |
| floc-faces | OFA | −0.07 (n=8) | +0.90 (n=8) | −0.89 (n=8) | +0.98 (n=8) | −0.99 (n=8) | −0.92 (n=8) | 367 |
| floc-faces | FFA-1 | +0.01 (n=8) | +0.95 (n=8) | −0.91 (n=8) | +0.99 (n=8) | −1.00 (n=8) | −0.94 (n=8) | 399 |
| floc-faces | FFA-2 | +0.22 (n=7) | +0.98 (n=7) | −0.92 (n=7) | +1.00 (n=7) | −1.00 (n=7) | −0.95 (n=7) | 260 |
| floc-places | OPA | +0.34 (n=8) | +0.98 (n=8) | −0.86 (n=8) | +1.00 (n=8) | −1.00 (n=8) | −0.93 (n=8) | 1070 |
| floc-places | PPA | +0.44 (n=8) | +0.99 (n=8) | −0.88 (n=8) | +1.00 (n=8) | −1.00 (n=8) | −0.94 (n=8) | 620 |
| floc-bodies | EBA | +0.17 (n=8) | +0.97 (n=8) | −0.88 (n=8) | +0.99 (n=8) | −1.00 (n=8) | −0.92 (n=8) | 2174 |
| floc-bodies | FBA-1 | −0.07 (n=6) | +0.96 (n=6) | −0.91 (n=6) | +1.00 (n=6) | −1.00 (n=6) | −0.95 (n=6) | 314 |
| floc-bodies | FBA-2 | +0.13 (n=8) | +0.98 (n=8) | −0.92 (n=8) | +1.00 (n=8) | −1.00 (n=8) | −0.94 (n=8) | 252 |
| floc-words | OWFA | −0.29 (n=8) | +0.88 (n=8) | −0.85 (n=8) | +0.96 (n=8) | −0.99 (n=8) | −0.88 (n=8) | 392 |
| floc-words | VWFA-1 | −0.16 (n=8) | +0.96 (n=8) | −0.91 (n=8) | +0.99 (n=8) | −1.00 (n=8) | −0.96 (n=8) | 454 |
| floc-words | VWFA-2 | −0.02 (n=5) | +0.95 (n=6) | −0.94 (n=6) | +0.99 (n=6) | −1.00 (n=6) | −0.96 (n=6) | 150 |
| floc-words | mfs-words | −0.08 (n=8) | +0.97 (n=8) | −0.93 (n=8) | +0.99 (n=8) | −1.00 (n=8) | −0.96 (n=8) | 121 |


**Supplementary Table S11 | The readout span without a stream split under the pre-specified unit-level rule: fixed-head gain sweep on the shape- and texture-leaning halves of the units.** The special case of Extended Data Table 3 in which the two subspaces are spanned by disjoint sets of units. Each unit of a network's penultimate layer is scored by the fraction of its variance between shape categories minus that between texture categories on the 1,200 cue-conflict images (one-way η²); the median split gives a shape-leaning half S and a texture-leaning half T. logits(w) = head(g(w) ⊙ y) with g = 2w on S and 2(1 − w) on T, the network's own head untouched; w = 0.5 is the trained network. The split is defined on one stratified half of the images and the rate scored on the other (10 random splits, both directions, the same splits as Extended Data Table 3; mean, with the range over splits and seeds in brackets). Rates by the paper's primary estimator (112-class, decided images) for our networks and the Geirhos 16-class probability aggregation for the ImageNet networks; “forced-estimator span” is the same sweep on the forced two-way estimator. “shape-leaning units” is the fraction of units with η²_shape > η²_texture on all images; for the two-stream concatenation the fraction of S that belongs to stream 1 is given in parentheses (P3). “w within 5 pts” is the w-range over which top-1 on held-out natural images (MiniEcoset validation; a stratified 10-per-class ILSVRC-2012 validation subset for the ImageNet networks) stays within 5 points of the trained value. “refit span” is the lossy-sum protocol z(w) = w·y_S + (1 − w)·y_T with a linear and a two-layer head refitted at every w (Methods 4.3 budget), our networks only. The last two columns are a sensitivity analysis added after the pre-specified analysis, at a referee's request: the same held-out sweep with S defined as the absolute shape-leaning set (units with d_j > 0) or as the top decile of units by d_j, and T as the remainder; given are the held-out span with its range over splits and seeds, the minimum Spearman ρ, the mean size of S and the w-range within 5 points of the trained top-1 (no gate exists for the two-stream concatenation, which has no held-out natural-image features). Pre-specification: `docs/PLAN_GENERALITY_2026-09-14.md`. Source: `results/figures_v2/generality_behaviour.json`, aggregated by `tmp_analysis/figs_v2/generality_behaviour.py` from the per-network outputs of `src/analysis/generality/` (`slurm/generality_{cc,mini,inval,rules}.sh`); the per-network files are outside the repository.

| network | units | trained rate | w = 0 | w = 1 | span (held-out) [range] | ρ(w, rate) | forced-estimator span | shape-leaning units | w within 5 pts of trained top-1 | refit span linear / two-layer | span, S = {d > 0} | span, top decile | seeds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| two-stream, diet, streams concatenated (validation) | 1024 | 0.675 | 0.420 | 0.879 | 0.461 [0.39, 0.57] | 1.00 | 0.307 | 0.49 (S from stream 1: 0.99) | — | — | 0.46 [0.39, 0.59] (ρ ≥ 0.98; |S| = 504) | 0.20 [0.11, 0.30] (ρ ≥ 0.56; |S| = 102) | 3 |
| single ResNet-34, diet | 512 | 0.625 | 0.565 | 0.688 | 0.136 [0.03, 0.24] | 0.71 | 0.102 | 0.17 | 0.20–0.78 (top-1 0.626) | 0.06 / 0.05 | 0.12 [0.03, 0.23] (ρ ≥ -0.78; |S| = 89; gate 0.00–0.73) | 0.11 [0.03, 0.20] (ρ ≥ -0.95; |S| = 51; gate 0.00–0.77) | 3 |
| single ResNet-18, diet | 512 | 0.557 | 0.467 | 0.652 | 0.194 [0.11, 0.31] | 0.97 | 0.131 | 0.09 | 0.28–0.72 (top-1 0.547) | 0.09 / 0.06 | 0.11 [0.04, 0.18] (ρ ≥ -0.63; |S| = 44; gate 0.00–0.73) | 0.12 [0.04, 0.21] (ρ ≥ -0.42; |S| = 51; gate 0.00–0.72) | 3 |
| single ResNet-18, no diet (`rgb_tsn`) | 512 | 0.445 | 0.374 | 0.566 | 0.195 [0.11, 0.39] | 0.96 | 0.164 | 0.05 | 0.27–0.75 (top-1 0.678) | 0.10 / 0.09 | 0.15 [0.03, 0.31] (ρ ≥ -0.57; |S| = 24; gate 0.00–0.78) | 0.17 [0.07, 0.29] (ρ ≥ -0.55; |S| = 51; gate 0.00–0.75) | 3 |
| ImageNet ViT-B/16 | 768 | 0.399 | 0.371 | 0.455 | 0.086 [0.07, 0.11] | 0.98 | 0.138 | 0.01 | 0.00–1.00 (top-1 0.815) | — | 0.21 [0.10, 0.34] (ρ ≥ 0.53; |S| = 6; gate 0.00–0.95) | 0.19 [0.14, 0.26] (ρ ≥ 0.81; |S| = 76; gate 0.00–0.95) | 1 |
| ImageNet ResNet-152 | 2048 | 0.293 | 0.227 | 0.440 | 0.214 [0.18, 0.25] | 1.00 | 0.196 | 0.02 | 0.15–0.90 (top-1 0.785) | — | 0.27 [0.22, 0.33] (ρ ≥ 0.86; |S| = 45; gate 0.00–0.85) | 0.30 [0.26, 0.36] (ρ ≥ 0.94; |S| = 204; gate 0.00–0.80) | 1 |
| ImageNet ConvNeXt-B | 1024 | 0.269 | 0.229 | 0.343 | 0.115 [0.09, 0.14] | 0.99 | 0.173 | 0.00 | 0.00–1.00 (top-1 0.843) | — | 0.28 [0.20, 0.35] (ρ ≥ 0.14; |S| = 2; gate 0.00–0.95) | 0.21 [0.13, 0.29] (ρ ≥ 0.73; |S| = 102; gate 0.00–1.00) | 1 |
| ImageNet ConvNeXt-S | 768 | 0.248 | 0.211 | 0.335 | 0.124 [0.11, 0.15] | 0.99 | 0.179 | 0.00 | 0.00–1.00 (top-1 0.840) | — | undefined (empty half) | 0.17 [0.12, 0.20] (ρ ≥ 0.72; |S| = 76; gate 0.00–0.95) | 1 |
| ImageNet ResNet-50 | 2048 | 0.222 | 0.167 | 0.346 | 0.180 [0.14, 0.23] | 0.99 | 0.202 | 0.02 | 0.15–0.90 (top-1 0.760) | — | 0.27 [0.20, 0.32] (ρ ≥ 0.75; |S| = 34; gate 0.00–0.85) | 0.27 [0.20, 0.31] (ρ ≥ 0.85; |S| = 204; gate 0.00–0.80) | 1 |
| ImageNet VGG-16 | 4096 | 0.110 | 0.096 | 0.189 | 0.095 [0.07, 0.11] | 0.96 | 0.106 | 0.01 | 0.00–1.00 (top-1 0.711) | — | 0.40 [0.34, 0.46] (ρ ≥ -0.38; |S| = 56; gate 0.00–0.95) | 0.27 [0.22, 0.34] (ρ ≥ 0.23; |S| = 409; gate 0.00–0.90) | 1 |

**Supplementary Table S12 | Cortex without a stream split: the shape- and texture-predictive components as encoding bands.** Upper block (subspace bands): the three bands are the shape-predictive subspace S = Y·B_S (15 dimensions), the texture-predictive subspace T = Y·B_T (15) and the residual R = Y·B_R of Extended Data Table 3 (bases fitted on all 1,200 images), with U_S = r²_{STR} − r²_{TR} and U_T = r²_{STR} − r²_{SR} from the three-band model and the two two-band models (random search, 30 and 20 samples); “shared” is r²_{STR} − U_S − U_T and “r²” the three-band model's noise-ceiling-normalised r². w*_cortex there is the weight of the 15-dimensional lossy sum z(w) = w·s + (1 − w)·t alone. Lower block (unit halves): for each single-trunk network the penultimate units were split as in Supplementary Table S11 (all 1,200 images) and the two halves entered as two bands of a banded ridge model of every `nsdgeneral` voxel (same split, folds and alpha grid as Methods 4.5); U_shape and U_texture are the noise-ceiling-normalised unique variances and the index is (U_shape − U_texture)/(U_shape + U_texture), median over reliable voxels (noise ceiling > 0.1, ≥ 20 voxels), mean over eight subjects (seeds averaged per subject). Regions: V1–hV4 from `prf-visualrois`; ventral (VO1/2, PHC1/2), lateral (LO1/2, TO1/2) and dorsal (V3A/B, IPS0–2) from Kastner2015. The two-stream diet row applies the same computation to the magno (shape-biased) and parvo (texture-biased) bands of Fig. 3. “shared” is the cross-validated variance the bands account for jointly (the full model's r² minus the unique variances) and “r²” the full model's noise-ceiling-normalised r², medians over the same voxels. w*_cortex is the mixing weight of the lossy sum z(w) = w·y_S + (1 − w)·y_T that maximises an ROI's median test R² (Extended Data Fig. 5 protocol). Exact paired Wilcoxon over subjects. The unit-rule rows for ImageNet ViT-B/16 and ConvNeXt-B were run after the pre-registered analysis, at the identical protocol, to ask whether the unit-rule cortical result depends on architecture; they are post hoc and are marked as such in `docs/PLAN_GENERALITY_2026-09-14.md`. Source: `results/figures_v2/generality_cortex.json` and `results/figures_v2/subspace_cortex.json`, aggregated by `tmp_analysis/figs_v2/generality_cortex.py` from the per-subject fits of `src/analysis/generality/cortex_bands.py` (`slurm/generality_{nsd,cortex}.sh`, `slurm/subspace_{nsd,cortex}.sh`), which are deposited alongside the encoding outputs of Fig. 3 (Code availability).

| network | statistic | V1 | V2 | V3 | hV4 | ventral | lateral | dorsal | hV4 − V1 (n neg, p) | extrastriate − V1 (n neg, p) | ρ(rank, index) V1–hV4 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **subspace bands S, T, R** | |  |  |  |  |  |  |  |  | | |
| single ResNet-18, no diet | index | +0.12 | -0.01 | +0.01 | +0.03 | +0.35 | +0.13 | +0.32 | -0.088 (5/8, 0.4609) | +0.151 (2/8, 0.0781) | -0.25 |
| | U_shape | 0.011 | 0.009 | 0.010 | 0.010 | 0.013 | 0.009 | 0.011 | | | |
| | U_texture | 0.008 | 0.009 | 0.009 | 0.009 | 0.006 | 0.007 | 0.006 | | | |
| | shared | 0.329 | 0.311 | 0.305 | 0.327 | 0.521 | 0.409 | 0.375 | | | |
| | r² (full model, nc-normalised) | 0.351 | 0.332 | 0.327 | 0.351 | 0.542 | 0.429 | 0.396 | | | |
| | w*_cortex (median) | 0.63 | 0.68 | 0.68 | 0.63 | 0.33 | 0.33 | 0.33 | | Δw* extrastriate − V1 -0.23 (6/8 neg, 0.0391) | |
| single ResNet-34, diet | index | -0.03 | -0.13 | -0.08 | +0.13 | +0.19 | -0.02 | +0.14 | +0.153 (2/8, 0.1094) | +0.130 (2/8, 0.3828) | +0.25 |
| | U_shape | 0.007 | 0.007 | 0.007 | 0.010 | 0.011 | 0.007 | 0.010 | | | |
| | U_texture | 0.007 | 0.008 | 0.008 | 0.007 | 0.008 | 0.007 | 0.008 | | | |
| | shared | 0.325 | 0.298 | 0.293 | 0.316 | 0.503 | 0.379 | 0.358 | | | |
| | r² (full model, nc-normalised) | 0.341 | 0.316 | 0.313 | 0.337 | 0.524 | 0.397 | 0.378 | | | |
| | w*_cortex (median) | 0.62 | 0.57 | 0.55 | 0.45 | 0.55 | 0.68 | 0.48 | | Δw* extrastriate − V1 -0.01 (4/8 neg, 0.9453) | |
| ImageNet ResNet-50 | index | -0.33 | -0.19 | -0.23 | +0.07 | -0.16 | -0.19 | -0.24 | +0.395 (1/8, 0.0156) | +0.131 (2/8, 0.1484) | +0.60 |
| | U_shape | 0.004 | 0.005 | 0.005 | 0.008 | 0.005 | 0.004 | 0.004 | | | |
| | U_texture | 0.007 | 0.006 | 0.007 | 0.006 | 0.006 | 0.005 | 0.006 | | | |
| | shared | 0.333 | 0.311 | 0.307 | 0.365 | 0.635 | 0.496 | 0.462 | | | |
| | r² (full model, nc-normalised) | 0.348 | 0.325 | 0.324 | 0.384 | 0.649 | 0.508 | 0.479 | | | |
| | w*_cortex (median) | 0.45 | 0.40 | 0.45 | 0.40 | 0.40 | 0.45 | 0.40 | | Δw* extrastriate − V1 -0.02 (5/8 neg, 0.6406) | |
| ImageNet ViT-B/16 | index | -0.45 | -0.54 | -0.45 | -0.25 | -0.50 | -0.43 | -0.70 | +0.193 (1/8, 0.0781) | -0.097 (3/8, 0.8438) | +0.37 |
| | U_shape | 0.004 | 0.003 | 0.004 | 0.007 | 0.004 | 0.003 | 0.002 | | | |
| | U_texture | 0.006 | 0.006 | 0.007 | 0.009 | 0.010 | 0.005 | 0.008 | | | |
| | shared | 0.254 | 0.242 | 0.254 | 0.292 | 0.555 | 0.404 | 0.382 | | | |
| | r² (full model, nc-normalised) | 0.267 | 0.256 | 0.271 | 0.310 | 0.573 | 0.416 | 0.398 | | | |
| | w*_cortex (median) | 0.50 | 0.40 | 0.45 | 0.15 | 0.50 | 0.40 | 0.55 | | Δw* extrastriate − V1 +0.08 (3/8 neg, 0.3828) | |
| ImageNet ConvNeXt-B | index | +0.17 | +0.01 | -0.17 | +0.38 | +0.00 | -0.30 | -0.51 | +0.212 (3/8, 0.1953) | -0.440 (7/8, 0.0156) | +0.08 |
| | U_shape | 0.006 | 0.005 | 0.005 | 0.008 | 0.006 | 0.004 | 0.003 | | | |
| | U_texture | 0.004 | 0.003 | 0.004 | 0.003 | 0.004 | 0.006 | 0.005 | | | |
| | shared | 0.256 | 0.240 | 0.233 | 0.285 | 0.540 | 0.408 | 0.369 | | | |
| | r² (full model, nc-normalised) | 0.267 | 0.253 | 0.246 | 0.301 | 0.554 | 0.422 | 0.385 | | | |
| | w*_cortex (median) | 0.45 | 0.55 | 0.55 | 0.00 | 0.75 | 0.70 | 0.75 | | Δw* extrastriate − V1 +0.20 (1/8 neg, 0.0391) | |
| **unit halves S, T** | |  |  |  |  |  |  |  |  | | |
| two-stream, diet (magno − parvo) | index | -0.06 | -0.44 | -0.70 | -0.84 | -0.88 | -0.87 | -0.87 | — | — | — |
| | U_shape | 0.087 | 0.045 | 0.026 | 0.016 | 0.014 | 0.014 | 0.013 | | | |
| | U_texture | 0.099 | 0.119 | 0.145 | 0.180 | 0.246 | 0.190 | 0.181 | | | |
| | shared | 0.289 | 0.286 | 0.261 | 0.219 | 0.230 | 0.184 | 0.168 | | | |
| | r² (full model, nc-normalised) | 0.485 | 0.457 | 0.439 | 0.428 | 0.496 | 0.390 | 0.373 | | | |
| single ResNet-34, diet | index | +0.05 | +0.02 | +0.05 | -0.03 | -0.22 | -0.09 | -0.06 | -0.080 (7/8, 0.0156) | -0.178 (7/8, 0.0156) | -0.25 |
| | U_shape | 0.032 | 0.030 | 0.028 | 0.025 | 0.017 | 0.016 | 0.020 | | | |
| | U_texture | 0.028 | 0.028 | 0.025 | 0.027 | 0.027 | 0.020 | 0.022 | | | |
| | shared | 0.263 | 0.245 | 0.245 | 0.274 | 0.458 | 0.353 | 0.320 | | | |
| | r² (full model, nc-normalised) | 0.327 | 0.304 | 0.300 | 0.329 | 0.506 | 0.392 | 0.366 | | | |
| | w*_cortex (median) | 0.53 | 0.60 | 0.57 | 0.53 | 0.53 | 0.55 | 0.57 | | Δw* extrastriate − V1 +0.01 (4/8 neg, 0.9453) | |
| single ResNet-18, no diet | index | +0.17 | +0.13 | +0.12 | +0.17 | -0.15 | -0.14 | -0.14 | -0.005 (5/8, 0.9453) | -0.316 (8/8, 0.0078) | -0.15 |
| | U_shape | 0.038 | 0.035 | 0.030 | 0.031 | 0.019 | 0.016 | 0.018 | | | |
| | U_texture | 0.026 | 0.026 | 0.024 | 0.022 | 0.025 | 0.022 | 0.024 | | | |
| | shared | 0.272 | 0.258 | 0.262 | 0.288 | 0.483 | 0.371 | 0.335 | | | |
| | r² (full model, nc-normalised) | 0.340 | 0.321 | 0.319 | 0.347 | 0.530 | 0.415 | 0.381 | | | |
| | w*_cortex (median) | 0.80 | 0.80 | 0.68 | 0.68 | 0.42 | 0.48 | 0.37 | | Δw* extrastriate − V1 -0.32 (7/8 neg, 0.0234) | |
| ImageNet ResNet-50 | index | -0.38 | -0.38 | -0.32 | -0.50 | -0.44 | +0.06 | -0.07 | -0.120 (5/8, 0.3828) | +0.235 (1/8, 0.0391) | -0.13 |
| | U_shape | 0.008 | 0.008 | 0.008 | 0.006 | 0.006 | 0.010 | 0.009 | | | |
| | U_texture | 0.019 | 0.018 | 0.017 | 0.020 | 0.015 | 0.009 | 0.011 | | | |
| | shared | 0.312 | 0.292 | 0.297 | 0.365 | 0.606 | 0.489 | 0.446 | | | |
| | r² (full model, nc-normalised) | 0.343 | 0.322 | 0.327 | 0.397 | 0.633 | 0.512 | 0.471 | | | |
| | w*_cortex (median) | 0.10 | 0.10 | 0.00 | 0.05 | 0.00 | 0.85 | 0.70 | | Δw* extrastriate − V1 +0.07 (3/8 neg, 0.8125) | |
| ImageNet ViT-B/16 | index | -0.19 | -0.13 | -0.20 | -0.23 | -0.31 | -0.23 | -0.27 | -0.039 (5/8, 0.6406) | -0.081 (6/8, 0.2500) | -0.35 |
| | U_shape | 0.036 | 0.036 | 0.032 | 0.037 | 0.047 | 0.042 | 0.038 | | | |
| | U_texture | 0.053 | 0.047 | 0.050 | 0.062 | 0.092 | 0.070 | 0.069 | | | |
| | shared | 0.162 | 0.154 | 0.161 | 0.189 | 0.381 | 0.292 | 0.258 | | | |
| | r² (full model, nc-normalised) | 0.252 | 0.239 | 0.245 | 0.290 | 0.527 | 0.409 | 0.366 | | | |
| | w*_cortex (median) | 0.25 | 0.20 | 0.10 | 0.15 | 0.25 | 0.15 | 0.15 | | Δw* extrastriate − V1 -0.18 (5/8 neg, 0.5469) | |
| ImageNet ConvNeXt-B | index | -0.45 | -0.40 | -0.37 | -0.31 | -0.30 | -0.32 | -0.09 | +0.140 (2/8, 0.1094) | +0.218 (2/8, 0.0547) | +0.42 |
| | U_shape | 0.012 | 0.013 | 0.013 | 0.017 | 0.022 | 0.020 | 0.025 | | | |
| | U_texture | 0.030 | 0.029 | 0.029 | 0.035 | 0.039 | 0.035 | 0.029 | | | |
| | shared | 0.214 | 0.200 | 0.203 | 0.248 | 0.476 | 0.357 | 0.319 | | | |
| | r² (full model, nc-normalised) | 0.259 | 0.246 | 0.249 | 0.304 | 0.540 | 0.420 | 0.374 | | | |
| | w*_cortex (median) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | | Δw* extrastriate − V1 +0.10 (0/8 neg, 0.5000) | |

**Supplementary Table S13 | Error consistency with the ten observers at a matched cue-conflict rate.** Every frozen network of Extended Data Table 4 under the same gain (Methods 4.8), swept over w ∈ {0, 0.05, …, 1}; for each network the row is the **gated** readout whose rate is closest to 0.40 (tolerance ± 0.05; networks whose gate does not reach it are absent, 5 of 38: EfficientNet-V2-S, VGG-11-BN, VGG-16-BN, VGG-16, VGG-19-BN). κ is Cohen's κ on trial-by-trial agreement over the images both parties decided, mean over the ten observers, as a percentage of the human–human ceiling κ = 0.234 (Methods 4.3; Supplementary Table S6). “κ at the trained readout” is the same quantity at w = 0.5, the number Extended Data Table 4 reports and the form in which the field reports error consistency. “κ / κ_max” normalises by the largest κ attainable at the two marginals, κ_max = (1 − |p_model − p_human| − p_e)/(1 − p_e), expressed as a percentage of the same ratio for human pairs (35% of their own bound); values above 100% mean a network attains a larger share of its much smaller attainable agreement than two observers do of theirs, not that it is more human-like than a human. Over the 798 gated readouts of the 38 ImageNet-trained networks, raw κ is a function of the rate alone (one quadratic in the rate, R² = 0.90, RMSE 1.7 points of ceiling; per-network offsets −2.2 to 2.8 points, unrelated to the probe bias, ρ = 0.04), the bound itself running from 0.006 to 0.219. Normalised by the bound, the rank dependence on the rate disappears (pooled rank R² 0.001 against 0.891; within-network median Spearman ρ 0.20 against 0.99). Ranking the 33 networks by κ at their trained readouts against by κ here gives Kendall τ = 0.33 with 178 of 528 pairs (34%) reversing. The Stylized-ImageNet ResNet-50 cannot be gained below 0.65 inside its gate; at a common rate of 0.65 it is ResNet-50 SIN 135% [104, 165]; ViT-B/32 120% [64, 174]; AlexNet 79% [61, 96] on the normalised scale (95% percentile bootstrap over the ten observers). Pre-specification: `docs/PLAN_KAPPA_2026-09-17.md` (predictions K-1 to K-6: four failed and a fifth could not be tested as specified; the normalisation by κ_max is post hoc). Source: `results/figures_v2/kappa_rate.json`, aggregated by `tmp_analysis/figs_v2/kappa_rate.py` from the per-network sweeps of `src/analysis/generality/kappa_sweep.py` (`slurm/zoo_kappa.sh`); the per-network files are outside the repository. `\*` not ImageNet-trained.

| network | w at the matched rate | rate | κ, % of human–human ceiling | κ at the trained readout | κ / κ_max, % of the human-pair value | probe bias | trained rate |
|---|---|---|---|---|---|---|---|
| GoogLeNet | 0.90 | 0.397 | 18.7 | 9.5 | 185.9 | −0.52 | 0.251 |
| ShuffleNet-V2 x1.0 | 0.80 | 0.401 | 18.1 | 9.2 | 158.7 | −0.43 | 0.234 |
| Inception-v3 | 0.80 | 0.406 | 17.8 | 11.0 | 160.2 | −0.56 | 0.305 |
| Swin-S | 0.95 | 0.413 | 17.5 | 8.9 | 172.7 | −0.55 | 0.236 |
| MobileNet-V3-L | 0.80 | 0.407 | 17.0 | 10.8 | 173.6 | −0.47 | 0.265 |
| Swin-B | 1.00 | 0.418 | 17.0 | 7.7 | 169.1 | −0.57 | 0.233 |
| MaxViT-T | 0.85 | 0.401 | 16.0 | 9.1 | 171.2 | −0.57 | 0.295 |
| Swin-T | 0.90 | 0.407 | 15.4 | 8.4 | 163.6 | −0.52 | 0.225 |
| ViT-L/16 | 0.75 | 0.408 | 15.4 | 11.5 | 168.1 | −0.43 | 0.319 |
| EfficientNet-B3 | 0.75 | 0.396 | 15.1 | 12.5 | 138.4 | −0.50 | 0.291 |
| DenseNet-201 | 0.85 | 0.403 | 14.9 | 8.5 | 137.8 | −0.49 | 0.273 |
| MobileNet-V2 | 0.95 | 0.389 | 14.4 | 5.6 | 139.4 | −0.52 | 0.184 |
| Wide-ResNet-101-2 | 0.90 | 0.402 | 13.9 | 7.9 | 137.8 | −0.60 | 0.279 |
| EfficientNet-B0 | 0.85 | 0.407 | 13.9 | 7.1 | 130.2 | −0.54 | 0.271 |
| ResNet-101 | 0.95 | 0.405 | 13.8 | 8.7 | 150.6 | −0.52 | 0.284 |
| ResNet-152 | 0.90 | 0.402 | 13.7 | 9.1 | 126.5 | −0.49 | 0.293 |
| ResNet-50 SIN+IN, ft IN \* | 0.95 | 0.397 | 13.5 | 5.9 | 127.8 | −0.58 | 0.227 |
| ResNet-50 SIN+IN \* | 0.60 | 0.399 | 13.3 | 11.7 | 140.6 | −0.40 | 0.363 |
| RegNet-Y-16GF | 0.90 | 0.409 | 13.0 | 7.7 | 131.8 | −0.50 | 0.261 |
| ConvNeXt-B | 0.80 | 0.398 | 13.0 | 8.0 | 132.1 | −0.47 | 0.269 |
| ViT-B/32 | 0.00 | 0.418 | 12.9 | 23.0 | 130.4 | −0.30 | 0.617 |
| ResNeXt-50 32x4d | 0.90 | 0.383 | 12.9 | 5.6 | 130.6 | −0.58 | 0.219 |
| ResNet-18 | 0.90 | 0.407 | 12.7 | 6.3 | 107.3 | −0.53 | 0.233 |
| ResNet-34 | 0.85 | 0.403 | 12.5 | 6.1 | 131.0 | −0.53 | 0.259 |
| DenseNet-121 | 0.90 | 0.402 | 12.5 | 5.7 | 114.0 | −0.55 | 0.224 |
| MNASNet-1.0 | 0.85 | 0.406 | 12.5 | 8.7 | 113.7 | −0.53 | 0.226 |
| ConvNeXt-S | 0.85 | 0.399 | 12.4 | 8.1 | 122.7 | −0.53 | 0.248 |
| ConvNeXt-T | 0.85 | 0.395 | 12.3 | 7.0 | 115.3 | −0.50 | 0.241 |
| ResNet-50 | 1.00 | 0.394 | 12.2 | 5.1 | 116.9 | −0.56 | 0.222 |
| AlexNet | 0.75 | 0.393 | 12.0 | 7.1 | 124.8 | −0.45 | 0.261 |
| ConvNeXt-L | 0.80 | 0.401 | 11.8 | 9.2 | 122.4 | −0.49 | 0.293 |
| ViT-B/16 | 0.50 | 0.399 | 11.6 | 11.6 | 126.2 | −0.43 | 0.399 |
| RegNet-Y-8GF | 0.90 | 0.407 | 11.5 | 6.7 | 103.1 | −0.50 | 0.244 |
| Wide-ResNet-50-2 | 0.95 | 0.405 | 10.5 | 5.0 | 100.5 | −0.62 | 0.230 |
| ResNeXt-101 32x8d | 0.80 | 0.400 | 10.0 | 7.9 | 107.5 | −0.53 | 0.294 |
| ResNet-50 (V2 recipe) \* | 1.00 | 0.356 | 7.8 | 0.5 | 70.8 | −0.62 | 0.185 |

**Supplementary Table S14 | What the gain acts on: the span decomposed.** Every frozen network of Extended Data Table 4, at its trained readout. For each cue-conflict image the Geirhos rule gives a margin Δ = log p_shape − log p_texture (p_c the mean softmax probability over category c's ImageNet classes; the forced two-way estimator calls the image “shape” iff Δ > 0). The gain of Methods 4.8 adds (2w − 1)·W(P_S − P_T)y to the logits exactly, so to first order it changes Δ by (2w − 1)(e_S − e_T), with e_S and e_T the evidence the head receives through the shape and the texture subspace (u·W P_S y and u·W P_T y, u the gradient of Δ with respect to the logits). “Shape term” is the mean of e_S over the 1,200 images (evidence added at the shape end), “texture term” the mean of −e_T (texture evidence removed there), both in units of the margin; texture share = texture term / (shape term + texture term). M0 is the first-order forced-choice span from the per-image margins, M1 the two-number span Φ(z + λ) − Φ(z − λ) with z = mean Δ / sd Δ and λ = mean(e_S − e_T) / sd Δ. “Reported” is the held-out span of Extended Data Table 4 (primary estimator, on halves that share exemplars); “in sample” is the same estimator and “forced” the forced two-way estimator, both with the bases fitted on all 1,200 images. Random-direction span: ten draws (seeds 0–9) of two mutually orthogonal random 15-dimensional subspaces inside each network's 256-dimensional PCA space of the cue-conflict features, same gain, estimator (in sample) and gate; every random sweep stays inside its gate. Pre-specification: `docs/PLAN_SPAN_2026-09-18.md`, predictions S-1 to S-7, written after Extended Data Table 4 and before this analysis; the whole analysis is post hoc with respect to the manuscript's first three plans. Met: S-1 (M0 reproduces the forced-choice curve, median largest error 0.009; ρ(M0, forced span) = 0.99), S-5b (the Stylized-ImageNet texture term 0.31 is below every ImageNet network's, 0.93–3.95) and S-7 (median absolute mean signed random span 0.002; the reported span exceeds the largest random span in 38 of 38 ImageNet networks and 42 of 42). Failed: S-2 (M1 ranks the forced spans at ρ = 0.95 but the reported spans at 0.59, against 0.7), S-3 (ρ(λ, reported span) = 0.56, against 0.7), S-4 (ρ(λ, z) = −0.32, against |ρ| ≤ 0.3), S-5a (the Stylized-ImageNet λ 0.53 lies inside the ImageNet range 0.43–0.85) and S-6 (a decision-free head-weighted subspace share ranks the reported spans at ρ = 0.35, against 0.5, no better than the unweighted variance share, 0.40). Post hoc: the reported and the forced spans rank the 38 ImageNet networks at ρ = 0.58 (in-sample primary 0.81 with the reported and 0.89 with the forced); the forced span rises with the trained forced rate (ρ = 0.50 [0.20, 0.73]) where the reported one does not (0.04); among the 39 networks (the 38 and itself) the Stylized-ImageNet span is the smallest reported, the 9th smallest in sample and the 16th smallest forced, and the 3rd smallest with exemplars held out. The texture term exceeds the shape term in 37 of 38 ImageNet networks (texture share median 0.89, range 0.45–0.95; the exception is ViT-B/32), and the texture share tracks the probe bias of Extended Data Table 4 (ρ = −0.81 [−0.91, −0.62]; −0.86 over all 42) more closely than the trained rate does (0.59 [0.29, 0.80]; difference in |ρ| 0.22 [0.04, 0.44]); against the exemplar-balanced probe scored on held-out exemplars (Extended Data Table 4) it is −0.58 [−0.78, −0.29], no closer than the trained rate. Intervals are 95% percentile bootstraps over networks (10,000 resamples), descriptive of this fixed zoo. On the forced estimator the ranking by trained rate and by the shape-end rate agree at Kendall τ = 0.58 with 146 of 703 pairs (21%) reversing (in-sample primary τ = 0.50, 25%; held-out primary 0.47, 27% with exemplars shared and 0.63, 18.6% with exemplars held out). Source: `results/figures_v2/span_model.json`, aggregated by `tmp_analysis/figs_v2/span_model.py` from `src/analysis/generality/span_model.py` (`slurm/zoo_span.sh`); the per-network files are outside the repository.

| network | trained rate: primary / forced | span: reported / in sample / forced | forced span, M0 / M1 | z / λ | shape term / texture term (texture share) | random-direction span, absolute: median / largest |
|---|---|---|---|---|---|---|
| **ImageNet-trained (torchvision IMAGENET1K_V1)** | | | | | | |
| VGG-11-BN | 0.105 / 0.117 | 0.235 / 0.153 / 0.197 | 0.206 / 0.300 | −1.11 / 0.69 | 0.17 / 3.24 (0.95) | 0.007 / 0.018 |
| VGG-16-BN | 0.115 / 0.117 | 0.226 / 0.157 / 0.177 | 0.174 / 0.289 | −1.07 / 0.64 | 0.17 / 3.01 (0.95) | 0.012 / 0.025 |
| Wide-ResNet-50-2 | 0.230 / 0.225 | 0.261 / 0.274 / 0.306 | 0.312 / 0.348 | −0.61 / 0.54 | 0.16 / 2.71 (0.94) | 0.012 / 0.036 |
| VGG-16 | 0.110 / 0.115 | 0.268 / 0.216 / 0.253 | 0.253 / 0.336 | −1.04 / 0.72 | 0.25 / 3.95 (0.94) | 0.006 / 0.020 |
| VGG-19-BN | 0.150 / 0.149 | 0.241 / 0.177 / 0.190 | 0.195 / 0.296 | −0.90 / 0.56 | 0.19 / 2.88 (0.94) | 0.009 / 0.023 |
| Wide-ResNet-101-2 | 0.279 / 0.253 | 0.231 / 0.271 / 0.287 | 0.292 / 0.309 | −0.50 / 0.45 | 0.19 / 2.42 (0.93) | 0.008 / 0.022 |
| EfficientNet-V2-S | 0.184 / 0.183 | 0.198 / 0.189 / 0.258 | 0.273 / 0.393 | −0.76 / 0.68 | 0.16 / 1.87 (0.92) | 0.006 / 0.016 |
| ResNeXt-50 32x4d | 0.219 / 0.219 | 0.316 / 0.298 / 0.305 | 0.307 / 0.361 | −0.66 / 0.58 | 0.27 / 3.19 (0.92) | 0.014 / 0.026 |
| ResNet-50 | 0.222 / 0.210 | 0.290 / 0.254 / 0.264 | 0.276 / 0.328 | −0.74 / 0.55 | 0.28 / 2.72 (0.91) | 0.005 / 0.019 |
| Swin-S | 0.236 / 0.238 | 0.287 / 0.322 / 0.362 | 0.375 / 0.445 | −0.59 / 0.70 | 0.23 / 2.15 (0.90) | 0.011 / 0.022 |
| GoogLeNet | 0.251 / 0.247 | 0.296 / 0.278 / 0.278 | 0.286 / 0.387 | −0.58 / 0.60 | 0.21 / 1.95 (0.90) | 0.014 / 0.033 |
| ResNet-18 | 0.233 / 0.232 | 0.316 / 0.326 / 0.320 | 0.330 / 0.390 | −0.67 / 0.64 | 0.39 / 3.46 (0.90) | 0.014 / 0.031 |
| Swin-B | 0.233 / 0.240 | 0.229 / 0.263 / 0.333 | 0.349 / 0.416 | −0.60 / 0.65 | 0.24 / 2.15 (0.90) | 0.012 / 0.021 |
| ResNet-34 | 0.259 / 0.242 | 0.353 / 0.344 / 0.351 | 0.349 / 0.434 | −0.66 / 0.71 | 0.38 / 3.36 (0.90) | 0.003 / 0.032 |
| ResNet-101 | 0.284 / 0.258 | 0.276 / 0.219 / 0.203 | 0.215 / 0.292 | −0.58 / 0.44 | 0.26 / 2.21 (0.89) | 0.006 / 0.019 |
| DenseNet-121 | 0.224 / 0.202 | 0.347 / 0.363 / 0.355 | 0.349 / 0.432 | −0.70 / 0.72 | 0.45 / 3.78 (0.89) | 0.014 / 0.038 |
| MobileNet-V2 | 0.184 / 0.185 | 0.346 / 0.330 / 0.306 | 0.318 / 0.405 | −0.82 / 0.73 | 0.41 / 3.41 (0.89) | 0.009 / 0.034 |
| Swin-T | 0.225 / 0.224 | 0.309 / 0.354 / 0.408 | 0.428 / 0.480 | −0.63 / 0.78 | 0.29 / 2.34 (0.89) | 0.011 / 0.017 |
| ConvNeXt-T | 0.241 / 0.246 | 0.353 / 0.379 / 0.393 | 0.415 / 0.502 | −0.65 / 0.83 | 0.28 / 2.26 (0.89) | 0.011 / 0.021 |
| ConvNeXt-S | 0.248 / 0.234 | 0.287 / 0.361 / 0.413 | 0.424 / 0.497 | −0.62 / 0.81 | 0.28 / 2.23 (0.89) | 0.006 / 0.019 |
| RegNet-Y-8GF | 0.244 / 0.227 | 0.365 / 0.324 / 0.337 | 0.332 / 0.382 | −0.64 / 0.61 | 0.44 / 3.21 (0.88) | 0.008 / 0.036 |
| Inception-v3 | 0.305 / 0.283 | 0.246 / 0.305 / 0.302 | 0.300 / 0.360 | −0.41 / 0.51 | 0.24 / 1.68 (0.88) | 0.013 / 0.039 |
| ConvNeXt-B | 0.269 / 0.261 | 0.334 / 0.410 / 0.425 | 0.437 / 0.502 | −0.56 / 0.79 | 0.31 / 2.18 (0.88) | 0.009 / 0.019 |
| DenseNet-201 | 0.273 / 0.228 | 0.334 / 0.325 / 0.338 | 0.333 / 0.419 | −0.60 / 0.66 | 0.51 / 3.50 (0.87) | 0.010 / 0.025 |
| MNASNet-1.0 | 0.226 / 0.210 | 0.395 / 0.391 / 0.349 | 0.352 / 0.436 | −0.73 / 0.75 | 0.55 / 3.62 (0.87) | 0.009 / 0.022 |
| ResNeXt-101 32x8d | 0.294 / 0.265 | 0.283 / 0.303 / 0.322 | 0.326 / 0.363 | −0.48 / 0.53 | 0.44 / 2.89 (0.87) | 0.017 / 0.036 |
| ConvNeXt-L | 0.293 / 0.289 | 0.332 / 0.360 / 0.372 | 0.388 / 0.476 | −0.51 / 0.72 | 0.35 / 2.19 (0.86) | 0.009 / 0.023 |
| MaxViT-T | 0.295 / 0.297 | 0.239 / 0.272 / 0.338 | 0.353 / 0.406 | −0.43 / 0.58 | 0.33 / 1.88 (0.85) | 0.005 / 0.017 |
| ResNet-152 | 0.293 / 0.255 | 0.246 / 0.235 / 0.219 | 0.226 / 0.286 | −0.55 / 0.43 | 0.38 / 2.12 (0.85) | 0.013 / 0.030 |
| ShuffleNet-V2 x1.0 | 0.234 / 0.248 | 0.381 / 0.414 / 0.350 | 0.352 / 0.414 | −0.62 / 0.66 | 0.48 / 2.71 (0.85) | 0.010 / 0.069 |
| RegNet-Y-16GF | 0.261 / 0.242 | 0.300 / 0.291 / 0.317 | 0.318 / 0.352 | −0.56 / 0.53 | 0.53 / 2.93 (0.85) | 0.021 / 0.076 |
| AlexNet | 0.261 / 0.244 | 0.504 / 0.497 / 0.447 | 0.435 / 0.502 | −0.68 / 0.85 | 0.77 / 3.50 (0.82) | 0.010 / 0.020 |
| EfficientNet-B0 | 0.271 / 0.278 | 0.327 / 0.361 / 0.361 | 0.373 / 0.450 | −0.49 / 0.67 | 0.43 / 1.94 (0.82) | 0.010 / 0.016 |
| EfficientNet-B3 | 0.291 / 0.275 | 0.310 / 0.361 / 0.386 | 0.395 / 0.465 | −0.42 / 0.68 | 0.46 / 1.88 (0.80) | 0.010 / 0.022 |
| MobileNet-V3-L | 0.265 / 0.268 | 0.435 / 0.450 / 0.434 | 0.437 / 0.508 | −0.57 / 0.81 | 0.92 / 3.50 (0.79) | 0.009 / 0.020 |
| ViT-L/16 | 0.319 / 0.310 | 0.337 / 0.363 / 0.408 | 0.421 / 0.481 | −0.40 / 0.70 | 0.59 / 2.03 (0.77) | 0.008 / 0.017 |
| ViT-B/16 | 0.399 / 0.362 | 0.282 / 0.357 / 0.376 | 0.376 / 0.422 | −0.24 / 0.57 | 0.61 / 1.48 (0.71) | 0.005 / 0.015 |
| ViT-B/32 | 0.617 / 0.520 | 0.239 / 0.356 / 0.415 | 0.418 / 0.453 | 0.14 / 0.61 | 1.15 / 0.93 (0.45) | 0.008 / 0.023 |
| **same architecture, different training recipe** | | | | | | |
| ResNet-50 (V2 recipe) | 0.185 / 0.191 | 0.334 / 0.234 / 0.288 | 0.304 / 0.389 | −0.72 / 0.66 | 0.12 / 1.48 (0.92) | 0.013 / 0.024 |
| **Stylized-ImageNet controls (Geirhos et al., 2018a)** | | | | | | |
| ResNet-50 SIN+IN, ft IN | 0.227 / 0.223 | 0.277 / 0.265 / 0.273 | 0.267 / 0.341 | −0.66 / 0.55 | 0.26 / 2.75 (0.91) | 0.011 / 0.029 |
| ResNet-50 SIN+IN | 0.363 / 0.327 | 0.350 / 0.358 / 0.370 | 0.373 / 0.428 | −0.37 / 0.60 | 0.73 / 2.20 (0.75) | 0.019 / 0.026 |
| ResNet-50 SIN | 0.810 / 0.703 | 0.166 / 0.255 / 0.318 | 0.328 / 0.347 | 0.58 / 0.53 | 2.43 / 0.31 (0.11) | 0.015 / 0.028 |
