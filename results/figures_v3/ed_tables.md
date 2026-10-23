

**Extended Data Table 2 | The readout weight cortex prefers against the one behaviour prefers.** For each mixing weight w the classifier's own input z(w) = w·y₁ + (1 − w)·y₂ (y₁ magno stream output, y₂ parvo stream output, 512-d each) is the design of a ridge encoding model of every nsdgeneral voxel (same split, folds and alpha grid as the banded model; Methods 4.5). w*_cortex is the w that maximises the median test R² over reliable voxels of the ROI (mean over seeds per subject; median [range] over 8 subjects). "R² at w = 0.5 / max" is the median R² at the paper's readout weight relative to the best w. "concat" is the median R² of the 1,024-d concatenation [y₁, y₂], fitted as a reference and not as a bound: every mixture is a linear map of the concatenation, but the two models are regularised separately on a shared alpha grid, and the best mixture exceeds the fitted concatenation in 15 of the 112 subject × ROI × arm cells, by a median 0.0023 R² (2.3% of the concatenation; largest 0.0048). That excess is the reproducibility scale of these curves and exceeds the median R² the sweep itself buys, 0.0012 from w = ½ to w*, so w*_cortex is read as the location of an optimum and never as an R² difference. What the table supports is the ordinal contrast: extrastriate below V1 in 8/8 subjects (Δ = −0.323 [−0.371, −0.269], 10,000-sample percentile bootstrap over subjects, exact Wilcoxon p = .0078), reversed in the control (+0.308 [+0.194, +0.425], 8/8 positive, p =.008). "Voxels magno / parvo" is the fraction of the ROI's voxels whose own best w is ≥ 0.75 / ≤ 0.25. The behaviour-optimal weight (trial fit to each of 10 human observers' cue-conflict choices, the model's final checkpoints, 3 seeds) is given in the last row for comparison, with the cue-conflict shape-bias rate the model produces at each weight.

| arm | ROI | w*_cortex median [range] | R² at w = 0.5 / max | max R² (median) | concat R² | voxels best w ≥ 0.75 | voxels best w ≤ 0.25 | rate at w*_cortex | n |
|---|---|---|---|---|---|---|---|---|---|
| control | V1 | 0.43 [0.35, 0.50] | 0.996 | 0.1096 | 0.1185 | 0.21 | 0.23 | nan | 8 |
| control | V2 | 0.40 [0.20, 0.45] | 0.982 | 0.0890 | 0.0990 | 0.18 | 0.32 | nan | 8 |
| control | V3 | 0.43 [0.30, 0.45] | 0.993 | 0.0784 | 0.0863 | 0.20 | 0.28 | nan | 8 |
| control | hV4 | 0.50 [0.40, 0.60] | 0.998 | 0.0920 | 0.0996 | 0.25 | 0.18 | nan | 8 |
| control | ventral (VO, PHC) | 0.68 [0.60, 1.00] | 0.990 | 0.1647 | 0.1654 | 0.50 | 0.04 | nan | 8 |
| control | lateral (LO, TO) | 0.62 [0.45, 1.00] | 0.986 | 0.1239 | 0.1273 | 0.50 | 0.07 | nan | 8 |
| control | dorsal (V3A/B, IPS0–2) | 0.78 [0.50, 1.00] | 0.984 | 0.0872 | 0.0869 | 0.50 | 0.08 | nan | 8 |
| model | V1 | 0.65 [0.50, 0.75] | 0.989 | 0.1219 | 0.1313 | 0.41 | 0.11 | 0.75 | 8 |
| model | V2 | 0.55 [0.45, 0.60] | 0.997 | 0.0920 | 0.1018 | 0.26 | 0.20 | 0.70 | 8 |
| model | V3 | 0.45 [0.35, 0.50] | 0.998 | 0.0819 | 0.0877 | 0.17 | 0.29 | 0.66 | 8 |
| model | hV4 | 0.33 [0.10, 0.55] | 0.980 | 0.0912 | 0.0977 | 0.09 | 0.40 | 0.59 | 8 |
| model | ventral (VO, PHC) | 0.30 [0.20, 0.35] | 0.985 | 0.1603 | 0.1642 | 0.03 | 0.41 | 0.58 | 8 |
| model | lateral (LO, TO) | 0.35 [0.20, 0.45] | 0.980 | 0.1201 | 0.1234 | 0.05 | 0.44 | 0.61 | 8 |
| model | dorsal (V3A/B, IPS0–2) | 0.35 [0.25, 0.40] | 0.990 | 0.0837 | 0.0861 | 0.07 | 0.39 | 0.61 | 8 |
| behaviour (model) | 10 human observers, trial fit | 0.79 [0.33, 0.97] | — | — | — | — | — | 0.80 | 10 |

