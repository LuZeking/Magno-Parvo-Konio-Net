# PLAN — the cortex-to-decision test (pre-registered 2026-10-26)

Status: **registered, not run.** Committed before any encoding run of this plan was
submitted. Nothing below is changed after the first result file exists; additions go in a
dated addendum at the end, marked post hoc.

## 0. Why this test, and what the paper asks

The paper's question (PAPER_v3, Introduction): *how can human object recognition be
shape-biased while visual cortical representations remain sensitive to texture?*

Everything the paper has so far links two of three things at a time:

| link | where | stimuli |
|---|---|---|
| model streams ↔ cortex | Results 2.3 (banded ridge, NSD) | natural NSD photographs |
| model readout ↔ observers | Results 2.4–2.6 (rate, κ, per-image correspondence) | the 1,200 Geirhos cue-conflict images |
| predicted cortex ↔ observers | **nothing** | — |

The missing link is the one the paradox is about. This plan supplies it: the cue-conflict
images are pushed through each subject's *fitted* encoding model (Methods 4.5; the
`--cue-conflict` design that already exists in `run_encoding.py`, whose per-image
predictions were never stored), and the predicted response pattern of each ROI is asked
two questions on the same images the observers judged:

1. **Is it texture-biased?** — the cortical half of the paradox, now on the cue-conflict
   images themselves rather than inferred from NSD fits.
2. **Does it nevertheless carry the evidence the observers' decisions follow, image by
   image?** — the only reading of "texture-sensitive cortex, shape-biased behaviour" that
   does not require the decision to come from somewhere cortex is not.

If both hold, the paradox is reproduced inside one predicted representation: texture-biased
by the cue-conflict criterion, yet not withholding the shape evidence a downstream readout
needs. If (1) holds and (2) fails, the readout account loses its cortical half on these
images and the paper says so. If (1) fails, predicted cortex is not texture-biased on
these images and the paper says that.

This replaces the former Fig. 6 (frozen single-trunk networks under subspace gains, the
model zoo), which answered a different question (*how far can a rate be moved by
re-weighting a frozen representation?*) and is dropped from the manuscript.

## 1. Inputs (all exist; nothing is trained)

- **Encoders.** The banded ridge of Methods 4.5 for every `nsdgeneral` voxel of subjects
  1–8, refit from the cached features
  (`/scratch/byrong/mpk/encoding/cache/2026-10-16_fs2/<run>_subjNN.npz`) by the identical
  code path and seed. Each refit **must reproduce the stored fit**
  (`/scratch/byrong/mpk/encoding/results/2026-10-16_fs2/<run>_subjNN.npz`):
  max |Δ| of `r2_joint`, `unique_magno`, `unique_parvo` over the stored `reliable_mask`
  ≤ 1 × 10⁻³, recorded in the output as `repro_max_abs_diff`; a cell above tolerance is
  excluded from every statistic and listed. (The paper already reports that a batched
  refit reproduces the stored fit at voxelwise r = 1.00000 with the largest per-ROI index
  difference below 10⁻⁴.) This time the per-image predictions and the estimators' own
  coefficients are saved, so no third refit is ever needed.
- **Runs** (11 × 8 subjects = 88 encoding tasks):

  | arm | runs | role |
  |---|---|---|
  | model | `fs2_dvd300_avg_w050_ls_seed{0,1,2}` | the model of record |
  | control | `fs2_dvdfix300_avg_w050_ls_seed{0,1,2}` | identical test input, no trajectory |
  | channel swap | `fs2_dvd300swap_avg_w050_ls_seed{0,1,2}` | ramp on L−M |
  | untrained | `untrained_dvd300_seed0` | architecture floor |
  | input floor | `gabor_dvd300` (new; `--features gabor`, 300-month front end) | oriented energy on each stream's own input: no network |

  Front ends exactly as `slurm/encode_fs2.sh` / `slurm/paper_fs2_encoding.sh`;
  `check_frontend_matches_checkpoint` refuses a mismatch.
- **Stimuli pushed through every fit** (one extra design per fit, projected with the PCA
  fit on that subject's NSD training images and the band statistics of the training
  split, exactly as the stored `--cue-conflict` runs did):
  - the 1,200 Geirhos cue-conflict images of Methods 4.3 (`list_cue_conflict_images()`,
    same-cue images excluded) — the **test set**;
  - the 80 same-cue images of the same release (shape category = texture category) —
    **manipulation check only**, never scored against the observers;
  - **templates**: the MiniEcoset validation images of the 16 Geirhos categories
    (50 per class, 800 images; `val/` split of `texture2shape_miniecoset_256px.h5`,
    classes mapped by `cue_conflict.load_class_index()`), natural photographs the
    networks were not trained on and that share no exemplar with the Geirhos set.
- **Observers.** The ten observers' per-image profiles and ceilings already computed for
  Fig. 5c (`/scratch/byrong/mpk/perimage_behavior/results/2026-09-07_geirhos_perimage/`,
  `tmp_analysis/human_perimage.py`): per-image shape rate h_i = P(shape | shape or
  texture), per-image resolution rate v_i = P(shape or texture at all), each raw and
  with shape- and texture-category main effects removed (`residualise`). Ceilings are
  the exhaustive 126-partition split-half reliabilities, Spearman–Brown corrected:
  shape rate .658 (.624 residual), resolution rate .794 (.728 residual), Pearson. These
  numbers are fixed here and reused, not recomputed.

## 2. Quantities (per run × subject × ROI × band)

ROIs, over reliable voxels only (the stored `reliable_mask`; a cell with fewer than 20
reliable voxels is missing, as in Methods 4.5):

| ROI | atlas | labels |
|---|---|---|
| V1 | `prf-visualrois` | 1, 2 |
| V2 | `prf-visualrois` | 3, 4 |
| V3 | `prf-visualrois` | 5, 6 |
| hV4 | `prf-visualrois` | 7 |
| ventral | `streams` | 5 |
| lateral | `streams` | 6 |
| parietal | `streams` | 7 |

Bands: `joint` (the two-band model; **primary**), `magno`, `parvo` (the single-band
fits; the decomposition of P-C3).

For a cell, P is the (1,200 × n_vox) predicted pattern matrix of the test set and
Q the (800 × n_vox) matrix of the templates.

1. **Voxel normalisation** from the templates only: each voxel is centred and scaled by
   its mean and s.d. over the 800 template images; voxels with zero s.d. are dropped.
   The same transform is applied to P.
2. **Templates** T_c = mean normalised pattern over the 50 template images of category c
   (16 × n_vox).
3. **Evidence** E_ic = Pearson correlation over voxels between P_i and T_c, Fisher-z
   transformed (r clipped to ±0.999 999). For image i with shape category s(i) and
   texture category t(i): e_S,i = E_i,s(i), e_T,i = E_i,t(i).
4. **Shape margin** m_i = e_S,i − e_T,i. **Cue margin** c_i = max(e_S,i, e_T,i) −
   max over the other 14 categories of E_ic (pairs with v_i, as `cue_margin` does in
   `perimage_correspondence.py`).
5. **Cortical shape-bias rate** B = mean over the 1,200 images of [m_i > 0]; 95% CI by
   bootstrap over images (2,000 draws).
6. **Correspondence** r_h = Pearson(residualise(m), residualise(h)) and
   r_v = Pearson(residualise(c), residualise(v)), with the projector of
   `human_perimage.residualise` (one indicator per shape category and per texture
   category) applied identically to both sides; reported as **% of ceiling** =
   100 · r / ceiling (.624 and .728), the unit of Fig. 5c. Raw (un-residualised) r and
   Spearman ρ are reported beside them, never in their place.
7. **Manipulation check** (gate): nearest-template classification of the 80 same-cue
   images (argmax_c E_ic against the image's category); accuracy and its exact binomial
   p against 1/16. A cell with p ≥ .01 is **NOEST** for P-C2 and reported as such.
8. **Per-ROI encoder quality**: median test r² (`r2_joint`) over the cell's voxels,
   reported beside every number so that regional differences are read against it.

Seeds are averaged per subject at the level of m_i and c_i (the mean over the three seeds'
per-image margins), then B and r are computed on the averaged margins; per-seed values
are reported too. Seeds are never pooled with subjects.

## 3. Predictions (the two-stream account's, for the model; all directional ones are
pre-stated here)

- **P-C1 — texture-biased predicted cortex.** In the model (joint band), B < 0.5 in hV4,
  ventral and lateral in 8/8 subjects, every seed, and B < 0.682 (the same network's
  behavioural rate, Table 1) in every ROI. No prediction for V1–V3 (reported).
- **P-C2 — the evidence is there.** In the model (joint band), r_h > 0 in hV4, ventral and
  lateral in 8/8 subjects (exact Wilcoxon p = .0078 at n = 8), each above the input
  floor (`gabor_dvd300`, same voxels, same subject) in 8/8 subjects, and above the
  permutation null (§4) at p < .01 in each ROI on the seed-averaged margins (the null
  of the subject-mean r). The shape target decides P-C2. **P-C2b**: the same three
  criteria for r_v against the resolution target v, reported beside it and not part of
  the reading.
- **P-C3 — the two streams carry opposite evidence in the same voxels.** In every ROI,
  B(magno band) > B(parvo band) in 8/8 subjects and 3/3 seeds, and B(joint) lies
  between them.
- **P-C4 — the trajectory, on identical input.** Extrastriate correspondence (mean of
  hV4, ventral, lateral; % of ceiling) on the **resolution** target v: every model seed
  above every control seed and every swap seed (exact Mann–Whitney, 3 v 3, p = 0.100,
  the floor). On the **shape** target h: **no directional prediction** — Fig. 5c found
  the three arms interleaved on the cue picked — reported either way.
- **P-C5 — regional profile.** Which ROI tracks the decisions best is reported as a
  profile with no prediction, against each cell's voxel count and median r².

**Decision rules** (the reading of record is fixed by these, in this order):

| outcome | reading | what the paper states |
|---|---|---|
| P-C1 met and P-C2 met | `PARADOX_REPRODUCED_IN_PREDICTED_CORTEX` | the lead of Results 2.5 and the Discussion as drafted |
| P-C1 met, P-C2 not met | `TEXTURE_BIASED_NO_EVIDENCE` | predicted cortex withholds the evidence beyond the input floor; the readout account keeps only its behavioural half; stated as a failure |
| P-C1 not met | `CORTEX_NOT_TEXTURE_BIASED_ON_THESE_IMAGES` | the paradox is not reproduced in predicted cortex; stated as a failure |
| gate fails (≥ 2 of hV4/ventral/lateral NOEST) | `NOEST` | no cortical reading; the method is reported |

P-C3 and P-C4 qualify the reading; they never change it. "Met" for an 8/8 count means
8/8; 7/8 is reported as not met with its exact p.

## 4. Nulls, floors and sensitivity (fixed now)

- **Permutation null** for r_h and r_v: h (v) permuted across images **within**
  shape × texture cells (240 cells of 5 images), 2,000 permutations, the same permutation
  sequence (seed 0) for every ROI and arm; p = (1 + #{|r_perm| ≥ |r|}) / 2,001.
- **Input floor**: `gabor_dvd300` — the identical encoders on oriented-energy features of
  the two streams' transformed inputs (Methods 4.5's input-level null), so that a
  correspondence carried by low-level similarity between a stylised image and its
  texture or shape source is measured, not assumed away.
- **Architecture floor**: `untrained_dvd300_seed0`.
- **Exemplar structure**: the 1,200 images reuse 160 object and 48 texture exemplars.
  The CI of every r is also computed by cluster bootstrap over the 160 object exemplars
  (1,000 draws; images inherit their exemplar's draw); the permutation null above is
  reported beside it.
- **Sensitivity analyses** (reported, never substituted for the primary; each is a
  flag of `cortex_decision.py` and a run of `slurm/ctd_analysis.sh`):
  1. Spearman ρ over voxels in place of Pearson for E (`--method spearman`);
  2. V1 defined as `streams` early (label 1) instead of `prf-visualrois` 1–2 (`--v1-streams`);
  3. all voxels of a ROI, no reliability mask (`--all-voxels`).

## 5. What is not claimed

- Not that human cortex uses this readout: the responses are predicted by linear
  encoders fitted on natural NSD photographs and applied to stylised images outside
  that distribution; the manipulation check (§2.7) guards only that the predicted
  patterns carry category.
- Not a mapping of the computational streams onto anatomical M/P pathways or onto
  dorsal/ventral cortex.
- Not a measurement of the human readout weight (Results 2.4).
- Not a comparison with the dropped frozen-network analyses.

## 6. Execution

- Code: `src/analysis/encoding/run_encoding.py` gains `--save-predictions`,
  `--template-design miniecoset16`, `--save-encoder` and `--reproduce-check`, all off by
  default (existing runs are byte-for-byte unaffected); `src/analysis/encoding/
  template_design.py` builds the extra designs; `src/analysis/encoding/cortex_decision.py`
  computes §2–§4 and writes `ctd_analysis.json` + `ctd_summary.md`;
  `tmp_analysis/test_cortex_decision.py` covers the arithmetic on synthetic data and the
  saved-encoder path against himalaya's own predictions.
- Slurm: `slurm/ctd_encode.sh` (main, `--qos=standard`, array 0–87 %16, 8 CPU, 44 G,
  14 h; reuses the fs2 feature caches, so each task is the ridge only: 3–8 h) then
  `slurm/ctd_analysis.sh` (main, 4 CPU, 16 G, afterok). Outputs under
  `/scratch/byrong/mpk/encoding/results/2026-10-26_ctd/`; logs under
  `/scratch/byrong/mpk/encoding/logs/2026-10-26_ctd/`. Submit with `env -u TMPDIR sbatch`
  from a pinned code snapshot (`fusion_stage1_code/<sha>`), as every encoding run of this
  project. Estimated 450 CPU-hours, about 1.5 days wall at %16; ≈ 40 GB on scratch
  (float16 predictions, float32 coefficients, float16 PCA bases).
- Launch requires the user's explicit "launch".
- Figure 6 (new): (a) B per ROI, model / control / swap / untrained / input floor, with the
  behavioural rates (model 0.68, observers 0.96) as reference lines; (b) r_h and r_v per
  ROI as % of ceiling with the permutation null and floors; (c) B under the magno, parvo
  and joint bands per ROI; (d) one ROI's residualised margin against the observers'
  residualised shape rate, one point per image.
