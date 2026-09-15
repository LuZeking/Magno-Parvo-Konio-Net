# Readout re-weighting of shape- and texture-predictive subspaces — pre-specification

Written and committed 2026-09-15 **before any of the analyses below were run**
(workstream W1 of `docs/PLAN_NMI9_2026-09-15.md`). It is a new test, not a rerun of
the pre-specified unit-split test of `docs/PLAN_GENERALITY_2026-09-14.md`, whose
outcomes stand as reported (Results 2.4; Extended Data Tables 13–14).

## Question

The unit-split test asked whether a network with no imposed stream split still has a
readout-controlled cue-conflict rate, and found the direction in every network but a
magnitude of 0.09–0.21 against a pre-stated 0.20 — with only 0–17% of units
shape-leaning in absolute terms. Shape information in a trunk may be distributed
across units rather than owned by them, in which case a per-unit factorisation is the
wrong instrument. The v2b referee's question was in fact about "shape- and
texture-*predictive subspaces*". We therefore ask: with the shape- and
texture-predictive parts of a frozen representation defined as low-dimensional
subspaces, does one gain on the shape subspace, read through the network's own head,
move the rate by an amount comparable to the two-stream span **within a five-point
accuracy budget**, and does human cortex load on the texture-predictive subspace?

## Design

### D1. Subspaces
For a frozen network with penultimate features y ∈ ℝ^D (the input of its final linear
layer) and a *fitting set* of cue-conflict images:
1. Centre y on the fitting set and reduce by PCA (fit on the fitting set) to
   k = min(256, D, n_fit − 17) components, basis V (D × k).
2. In PCA coordinates, fit a 16-class linear discriminant analysis to the shape
   categories (sklearn `LinearDiscriminantAnalysis`, `solver="eigen"`,
   `shrinkage="auto"` = Ledoit–Wolf) and take its 15 discriminant directions A_S
   (k × 15); likewise A_T for the 16 texture categories.
3. Map back to unit space, U_S = V·A_S, U_T = V·A_T (D × 15), and orthonormalise each
   (QR).
4. Make the two bases mutually orthogonal by **symmetric (Löwdin) orthogonalisation**
   of [U_S, U_T]: B = U (UᵀU)^{−1/2}, B_S its first 15 columns, B_T its last 15. This
   treats shape and texture identically and changes each basis as little as possible;
   the mean cosine between each original and orthogonalised direction is recorded.
   Sensitivity (reported, not thresholded): sequential Gram–Schmidt in both orders.
5. Projectors P_S = B_S B_Sᵀ, P_T = B_T B_Tᵀ, P_R = I − P_S − P_T (the residual, D − 30
   dimensions). B_R is an orthonormal basis of the residual (QR completion of
   [B_S, B_T]).

For behaviour the fitting set is one half of the 1,200 images, stratified by
shape × texture pair, with the same generator and seed as the unit-split analysis, so
the ten random splits are identical to those of Extended Data Table 13; everything is
scored on the other half, both directions. For cortex and for the accuracy gate the
bases are fit on all 1,200 images ("all-image bases"), there being no circularity with
natural images.

### D2. Fixed-head gain sweep (the analogue of Fig. 4a)
y(w) = P_R y + 2w·P_S y + 2(1 − w)·P_T y, logits(w) = W·y(w) + b through the network's own
untouched head, w ∈ {0, 0.05, …, 1}; at w = 0.5, y(w) = y and the network is exactly
itself. Rates by the primary estimator (112-class, decided images) for our networks and
the Geirhos 16-class rule for the ImageNet networks; the forced two-way estimator
alongside. Per curve: span = max − min over w, Spearman ρ(w, rate). Accuracy gate: top-1
of the gained features (all-image bases) on the held-out natural images of Extended
Data Table 13 (MiniEcoset validation; the stratified 10-per-class ILSVRC-2012 subset);
the **gated span** is max − min of the seed-mean held-out curve over the w whose top-1
lies within 5 points of the trained value.

### D3. Lossy-sum refit (the analogue of Fig. 4b), our nine MiniEcoset networks
s = B_Sᵀ y and t = B_Tᵀ y (15-d each, in basis order) are mixed lossily and the residual
passed unchanged: z(w) = [w·s + (1 − w)·t, B_Rᵀ y], of dimension D − 15. A linear head and
a two-layer head (hidden 512) are refitted on MiniEcoset training features at each of
the 21 weights with the budget of Methods 4.3 (Adam, lr 10⁻³, batch 1024, 20 epochs,
one initialisation seed), bases fit on the fitting half, rate read on the other half,
both directions of the first split; validation accuracy at every w; a refit on the
unmixed y is the ceiling.

### D4. Cortex
NSD banded ridge (eight subjects; split, grouped folds, alpha grid, reliability rule
and regions as Methods 4.5 and `PLAN_GENERALITY` D4) with **three bands**: S = Y·B_S
(15-d), T = Y·B_T (15-d), R = Y·B_R (D − 30), each prepared as in the paper (centred,
unit RMS on training trials). Fits per subject: the three-band model (random search,
30 samples), the two two-band models without S and without T (20 samples), and the
single bands S and T. Unique variances U_S = r²_{STR} − r²_{TR}, U_T = r²_{STR} − r²_{SR},
noise-ceiling-normalised; index (U_S − U_T)/(U_S + U_T), median over reliable voxels per
region (noise ceiling > 0.1, U_S + U_T > 10⁻³, ≥ 20 voxels); regions V1, V2, V3, hV4 and
the Kastner2015 ventral, lateral and dorsal groups. Cortex-optimal mixing weight: the
15-d lossy sum z(w) = w·s + (1 − w)·t alone (the residual is w-independent and would only
dilute the comparison) as the design of a single-band ridge on an 11-point grid; w* is
the w maximising a region's median test r².

### Models
- Behaviour (D1–D2): the 18 networks of Extended Data Table 13 — `rgb_tsn` seeds 0–2
  (single ResNet-18, no diet), `dkl_bcs_single` seeds 0–2 (ResNet-18, diet),
  `dkl_bcs_single34` seeds 0–2 (ResNet-34, diet), the two-stream `dkl_bcs` seeds 0–2 with
  streams concatenated (validation), and torchvision ResNet-50, ResNet-152, VGG-16,
  ViT-B/16, ConvNeXt-S, ConvNeXt-B (IMAGENET1K_V1).
- Refit (D3): the nine single-trunk MiniEcoset networks.
- Cortex (D4): `rgb_tsn` ×3, `dkl_bcs_single34` ×3, ResNet-50 (stored NSD features) and,
  newly extracted, ViT-B/16 and ConvNeXt-B — five network families.
All at the same checkpoints as the unit-split analysis (final).

## Predictions, written before running

- **P-A (magnitude on standard networks).** The gated span is ≥ 0.20 in at least 4 of
  the 6 ImageNet networks and, as a seed mean, in each of our three single-trunk
  families; the seed-mean ρ(w, rate) is ≥ 0.90 in every network. Failure: gated spans
  < 0.20 in 3 or more ImageNet networks, in which case the readout account is stated
  for standard networks "in kind, at a third to half of the two-stream span", as now.
- **P-A′ (refit).** For the lossy sum, the two-layer head's span is ≥ 0.15 in each of
  the three families with ρ ≥ 0.90, the linear and two-layer spans agree within 0.05,
  and validation accuracy stays within 5 points of the unmixed ceiling at every w
  (the residual carries the accuracy).
- **P-B (validation).** In the two-stream concatenation, ≥ 90% of the energy of the
  all-image B_S lies on stream-1 units and ≥ 90% of B_T on stream-2 units, per seed, and
  the held-out span is ≥ 0.40.
- **P-C (cortex).** P-C1: the extrastriate − V1 contrast of the index is negative in
  ≥ 7/8 subjects in each of `rgb_tsn`, `dkl_bcs_single34` **and ResNet-50** (the cell
  that failed under the unit split). P-C2: the same holds in at least one of ViT-B/16
  and ConvNeXt-B. P-C3: w* is lower in extrastriate cortex than in V1 in ≥ 7/8 subjects
  in at least 3 of the 5 families. Failure of P-C1 for ResNet-50: the cortical
  generality claim stays scoped to networks trained on our images.
- **P-D (distributed information).** The gated subspace span exceeds the gated
  unit-split span of Extended Data Table 13 in every ImageNet network (6/6) and in every
  one of our three families (seed means).

Descriptives reported without thresholds: alignment cosines of the Löwdin step; the
fraction of feature variance in S, T and R; U_R and the shared variance; the
sequential-order sensitivity; the number of reliable voxels per region.

Statistics as Methods 4.7: exact paired Wilcoxon over eight subjects (floor .0078);
seeds reported separately and never pooled with subjects; no correction claimed, claims
resting on complete separation and replication. Whatever the outcome, every prediction
is reported, and the unit-split results are not revised.

## Compute (all CPU, `main`/`standard`, ≤ 48G, arrays with `%A_%a` logs)
- `slurm/subspace_cc.sh`: 18 elements, minutes each (bases, held-out sweeps, gate).
- `slurm/subspace_refit.sh`: 9 elements, 1–3 h each.
- `slurm/subspace_nsd.sh`: NSD features of ViT-B/16 and ConvNeXt-B, ≈ 2–4 h each.
- `slurm/subspace_cortex.sh`: 9 networks × 8 subjects = 72 elements for the variance
  partition (≈ 2–6 h each) and 72 for the w sweep; outputs under
  `/scratch/byrong/mpk/generality/cortex/2026-09-15_sub/`.

## Outcomes
(appended after the runs)
