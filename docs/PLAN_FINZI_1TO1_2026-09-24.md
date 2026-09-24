# Pre-registration: do the TSN streams prefer dorsal, lateral or ventral cortex under a 1-to-1 unit-to-voxel mapping?

Registered 2026-09-24, before any beta, feature or correlation for this analysis was computed.
Branch `finzi-1to1`. Instrument: Finzi, Margalit, Kay, Yamins & Grill-Spector (2026),
*Nat Commun*, doi:10.1038/s41467-026-76098-y (Fig. 2c; code github.com/dawnfinzi/spacestream,
`scripts/fitting_one_to_one_unit2voxel.py`).

## 1. Question

Finzi et al. tested "one task per stream" and found that task-specific networks map to their
predicted stream at near chance. TSN is the account they did not test: one *input* per stream
(luminance + developmental transform vs L−M), one objective. The question: under a competitive
1-to-1 assignment, are voxels of the dorsal (or lateral) high-level stream won by magno units more
often than voxels of the ventral stream?

Why this can say something the banded-ridge fits cannot: the per-voxel stream index of §2.3 is
saturated in every high-level `streams` ROI (Supplementary Table S10; diet −0.91 / −0.87 / −0.87,
no diet +0.96 / +0.97 / +0.99), so it cannot rank regions. A fixed-quota assignment forces a ranking.

## 2. Data

- **Voxels.** NSD `betas_fithrf_GLMdenoise_RR`, func1pt8mm. `lh.streams` / `rh.streams` labels
  5 (ventral), 6 (lateral), 7 (parietal = dorsal), each hemisphere separately. Keep voxels with
  noise ceiling for the 3-repetition mean NC = 100·ncsnr²/(ncsnr² + 1/3) ≥ 10 % (Finzi's trimming
  threshold). No `nsdgeneral` restriction (about half of every stream ROI lies outside it).
- **Responses.** Betas /300 → PSC, z-scored per voxel within session (`nsd_data.load_betas`),
  averaged over each image's available repetitions.
- **Split** (`nsd_data.build_train_test_split`): test = shared-515 (3 repetitions each); train =
  every other image the subject saw. The assignment uses train images only; test correlations use
  the 515 only.

## 3. Networks

Final checkpoints (`checkpoint_last.pt`), as in §2.3, with each arm's own front end:

| arm | runs | front end |
|---|---|---|
| diet (primary) | `dkl_bcs_seed{0,1,2}` | luminance + blur + CS \| L−M |
| no diet | `dkl_base_seed{0,1,2}` | luminance \| L−M |
| swap | `dkl_bcs_swap_seed{0,1,2}` | luminance \| L−M + blur + CS |
| untrained, diet front end | `build_model(None, seed∈{0,1,2})` | as diet |
| untrained, no-diet front end | `build_model(None, seed∈{0,1,2})` | as no diet |

Units = output of each trunk's `layer4` (post-ReLU of the last BasicBlock), 512 channels × 8 × 8
positions per stream at 256 px. **Group A = magno (stream 1), group B = parvo (stream 2).**

Positive-control network: TDANN (SimCLR, spatial weight α = 0.25, `relu_rescue` checkpoint
`simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3`, seed 0, from spacestream's OSF
deposit), `base_model.layer4.1`, 512 × 7 × 7, NSD images resized to 224 with ImageNet
normalisation (spacestream `NSD_TRANSFORMS`).

## 4. Units and the fixed-quota assignment

- **Superset.** Per TSN network, 128 channels per grid position per stream are drawn once
  (RNG seed 0) → 16,384 units. Units with zero variance over the train images are dropped
  (Finzi drops "visually non-responsive" units).
- **Draw.** For a subject × hemisphere with V voxels, n = ⌈V / 128⌉ channels are sampled per
  position per stream from the responsive superset units, so both streams have the same count at
  every position (retinotopy cancels in the magno − parvo contrast) and U = 128·n ≥ V. If a position
  has fewer than n responsive units in either stream, both streams take the smaller count there.
  D = 10 draws per network × subject × hemisphere.
- **Assignment.** Cost = 1 − Pearson r (train images); `scipy.optimize.linear_sum_assignment` on
  the U × V rectangle: every voxel gets exactly one unit, and at most 127 units stay unassigned. No
  iterative neighbourhood refinement (the current Finzi paper uses the initial functional
  assignment too).

## 5. Statistics

- share_A(S) = fraction of stream-S voxels (one hemisphere, one draw) assigned to group-A units.
- **Primary statistic** Δ_DV = share_A(dorsal) − share_A(ventral), **noise-ceiling stratified**:
  voxels of the three stream ROIs are pooled per subject × hemisphere and cut into NC quintiles;
  Δ_DV is the mean over quintiles holding ≥ 20 voxels of each of the two streams of the per-bin
  difference. Reason: the dorsal ROI has the highest median NC in 8/8 subjects, and under a fixed
  quota the weaker stream inherits the voxels the stronger one correlates least with.
- Aggregation: mean over draws → mean over hemispheres (equal weight) → mean over the three seeds
  of an arm, per subject. **Unit of inference = subject (n = 8).** Seeds are never pooled as
  replicates.
- Also reported (secondary): unstratified Δ_DV; Δ_LV = share_A(lateral) − share_A(ventral);
  the overall magno share; NC-normalised test r (r / √NC) of assigned pairs per stream and group.

## 6. Decision rule (primary: diet arm, Δ_DV, NC-stratified)

Smallest effect size of interest δ = 0.05 (five percentage points of voxel share). Exact two-sided
Wilcoxon signed-rank over the 8 subjects; 90 % t-interval over subjects for equivalence.

| outcome | condition |
|---|---|
| **MAGNO → DORSAL** | p < .05, mean > 0, mean ≥ δ |
| **MAGNO → VENTRAL** | p < .05, mean < 0, \|mean\| ≥ δ |
| **SMALL** | p < .05 but \|mean\| < δ (a reliable preference below the SESOI) |
| **NO PREFERENCE** | 90 % interval inside (−δ, +δ) |
| **UNRESOLVED** | none of the above |

The same rule, labelled secondary, is applied to: Δ_LV (diet); Δ_DV and Δ_LV of each other arm;
diet − untrained (diet front end) per subject (is a preference learnt or input-given?); diet − no
diet. Secondary outcomes do not change the primary label.

## 7. Positive controls (gate, evaluated before the primary is read)

A null from an instrument that cannot see a preference certifies nothing. Two known-positive
controls use the primary statistic itself, with groups chosen on *other* subjects:

- **Leave-one-subject-out (LOSO) groups.** For held-out subject s and each superset unit u:
  d_u = mean over the other 7 subjects and both hemispheres of [mean train r(u, dorsal voxels) −
  mean train r(u, ventral voxels)]. At each grid position the n units with the highest d_u form
  group A and the n with the lowest form group B (streams pooled). The fixed-quota assignment then
  runs in subject s, and Δ_PC = NC-stratified Δ_DV with that A. D = 5 draws (the n per position are
  sampled from the top and bottom 25 % of that position's units ranked by d_u).
- **Negative control.** Same procedure with A/B a random position-balanced split of the superset.

| gate | network | PASS condition |
|---|---|---|
| PC-TDANN (instrument valid) | TDANN α = 0.25, seed 0 | Δ_PC > 0 in ≥ 7/8 subjects, exact Wilcoxon p < .05, mean ≥ 2δ = 0.10 |
| PC-TSN (sensitivity in the model under test) | diet seed 0 | same |
| NEG (null calibrated) | diet seed 0 | 90 % interval of Δ_null inside (−δ, +δ) |

- PC-TDANN FAIL or NEG FAIL → the pipeline is not trusted; no TSN result is interpreted.
- PC-TDANN PASS, PC-TSN FAIL → the primary is reported descriptively only. Conclusion: "TSN
  layer-4 units carry no subject-replicable stream preference on this instrument." No
  NO PREFERENCE label is issued.
- All PASS → the primary label stands.

## 8. Descriptive reproduction (not gating)

TDANN α = 0.25 with Finzi's own design (all 25,088 units, rectangular assignment): NC-normalised test
r per stream beside the published 0.22 dorsal / 0.25 lateral / 0.34 ventral. These are not expected
to match exactly: this pipeline uses volume voxels and a threshold trim; Finzi used fsaverage
vertices and hand-trimmed ROIs.

## 9. Known deviations from Finzi et al.

func1pt8mm volume instead of fsaverage; threshold trimming instead of hand-drawn trimmed ROIs; a
two-group fixed quota instead of three task sub-models; no spatial-similarity metric (TSN units
have no cortical-sheet positions).

## 10. Outputs

Code `src/analysis/finzi_1to1/`, Slurm `slurm/finzi_1to1_*.sh`, results
`results/finzi_1to1/`. Caches and intermediates under `/scratch/byrong/TSN/finzi_1to1/`.
Any change to this file after results exist is recorded as a dated amendment below and never
edited in place.

## Amendments

**Amendment 1 — 2026-09-25, before any real beta, feature or correlation was computed.**
The LOSO group statistic of §7 is changed from a unit's *mean* train r over a stream's voxels to
the *90th percentile* of its train r over that stream's voxels (d_u = P90(r_u, dorsal) −
P90(r_u, ventral)). Reason: the synthetic end-to-end check
(`python -m src.analysis.finzi_1to1.synthetic`, then `mapping` and `stats`) planted a clean
dorsal/ventral preference in half of each network's units. There the mean-r grouping gave
Δ_PC = −0.03 (TDANN) and −0.02 (TSN), because units' loadings on voxel features have mixed signs
and the mean cancels. The P90 grouping gave +1.00 and +1.00. The magno/parvo primary recovered its
planted +0.42 and a null network gave NO PREFERENCE (90 % interval [+0.004, +0.031]); NEG passed
([−0.014, +0.013]). The gate uses P90. The registered mean-r grouping is still computed and
reported beside it, labelled descriptive. Nothing else changes.
