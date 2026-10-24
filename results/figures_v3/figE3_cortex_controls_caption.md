# Extended Data Figure 3 — caption

**The cortical gradient against its nuisance axes: the raw statistic, eccentricity and receptive-field size.**

**(a, b)** The stream-preference profile of Fig. 3a on both statistics: (a) the normalised per-voxel index (U_m − U_p)/(U_m + U_p), median over reliable voxels per ROI, and (b) the raw per-voxel difference U_m − U_p in units of noise-ceiling-normalised r². Lines are means over 8 subjects, shading ± SEM; the model, the control and the channel swap average three seeds per subject, and the untrained network on their shared front end is a single network. The untrained network already descends on both statistics (gradient −0.480 normalised, −0.195 raw). The model keeps that descent and steepens it on the normalised index (−1.27; −0.19 raw), whereas the control (−0.070; −0.004) and the channel swap (−0.019; +0.002) end flat. The two-statistics rule of Methods 4.7 therefore licenses "preserves", not "steepens".

**(c)** The gradient hV4 − mean(V1v, V1d) on the normalised index, computed within each of NSD's five eccentricity bands (`prf-eccrois`), so that V1 and hV4 voxels are compared only with voxels of matched eccentricity. One point per subject per band (seed mean), bars = mean. A cell needs at least 20 voxels per ROI or the subject is dropped from that band; in the foveal band this drops subject 02 in every arm.
- The model's gradient is negative in every band: 6/7 subjects in the foveal band, where subject 05 is positive, and 8/8 in the other four.
- It grows with eccentricity, from −0.58 in the foveal band to −1.33 beyond 4°.
- The control stays between −0.07 and +0.16 in every band, and the channel swap between −0.06 and +0.05.

Labels give the number of subjects in which the model's gradient is below the control's, and the exact Wilcoxon p for that band (7/7, p = .016, in the foveal band; 8/8, p = .0078, in the other four). Eccentricity modulates the model's gradient but does not produce it.

**(d)** The same gradient within pRF-size tertiles. Tertile boundaries are set per subject on the pooled V1 + hV4 voxels, using NSD's pRF size estimates (voxels with pRF R² > 5%).
- The model's gradient is negative in every tertile (−1.28, −1.26, −1.32; every subject).
- The control (−0.13, −0.13, −0.09) and the channel swap (−0.10, −0.08, −0.03) stay near zero.
- In the small tertile, subjects 03 and 08 have too few voxels in the model, so its comparison with the control rests on 6 subjects (6/6, p = .031); the other two tertiles give 8/8, p = .0078.

The hierarchical contrast is therefore not a receptive-field-size contrast in disguise. The within-subject regressions on the continuous predictors (Supplementary Note S3) give the same picture.

Source: `results/figures_v3/cortex_controls.json` (`tmp_analysis/fs2_cortex_controls.py`). The BA network's version of this figure is PAPER_v2 Extended Data Fig. 3.
