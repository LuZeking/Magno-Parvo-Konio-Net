# Generality of the readout account beyond the two-stream architecture — pre-specification

Written and committed 2026-09-14 **before any of the analyses below were run**.
It answers the open referee item that the 2026-09-10 NMI simulation called the
experiment that "decides fit" (`docs/NMI_REVIEW_v2b_2026-09-10.md`, major
comment 2): the readout-span claim of Results 2.4 is so far demonstrated only
inside the two-stream architecture, where the mixture w·y₁ + (1 − w)·y₂ is
given by construction.

## Question

In a network with **no imposed stream split** — one trunk, one image — is the
cue-conflict shape-bias rate still a property of the readout over frozen
features, and does human cortex still load on the texture-leaning part of the
representation while behaviour can be tilted to the shape-leaning part?

## Design

### D1. Unit factorisation
For a frozen network, y is the penultimate feature vector (the input of the
final linear layer; 512-d for the ResNet-18/34 trunks, 2048-d for ResNet-50,
768/1024-d for ViT-B/16 and ConvNeXt, 4096-d for VGG-16). On the 1,200
cue-conflict images each unit j receives a shape sensitivity η²_s(j) and a
texture sensitivity η²_t(j): the fraction of the unit's variance between the
16 shape categories (resp. the 16 texture categories), one-way. Its
shape-leaning score is d_j = η²_s(j) − η²_t(j). Units are split at the median
of d into a shape-leaning half S and a texture-leaning half T. The split is
equal-sized because the two-stream case it generalises is 512 v 512.

For behaviour the split is defined on one half of the images (a random half,
stratified by shape × texture pair) and everything is scored on the other
half; 10 random splits, both directions, reported as mean and range. For
cortex there is no circularity with cue-conflict images, so the split uses all
1,200.

### D2. Fixed-head gain sweep (the analogue of Fig. 4a)
logits(w) = head(g(w) ⊙ y), with g_j(w) = 2w for j ∈ S and 2(1 − w) for j ∈ T,
w ∈ {0, 0.05, …, 1}. At w = 0.5 this is the trained network exactly. The head
is the network's own trained linear layer, untouched. Shape bias by the paper's
primary estimator (112-class decided images) and the forced two-way estimator
(Methods 4.3); for ImageNet networks the Geirhos 16-class probability
aggregation (`src/analysis/generality/imagenet16.json`), which decides every
image.

### D3. Lossy-sum refit sweep (the analogue of Fig. 4b)
A per-unit gain can be undone by a refitted linear head, so the refit test
uses a lossy mixture as the two-stream model does: z(w) = w·y_S + (1 − w)·y_T,
y_S and y_T the S and T halves paired by rank of d (256-d each). Linear and
two-layer heads are refitted on MiniEcoset training features at each w with
the budget of Methods 4.3 (Adam, lr 10⁻³, batch 1024, 20 epochs, one init
seed); validation accuracy is reported at every w and the span is also given
restricted to heads within 5 points of the best. Our trunks only (the ImageNet
networks have no MiniEcoset head to refit).

### D4. Cortex
NSD banded ridge with two bands, S and T (the penultimate units), eight
subjects, same split, folds and alpha grid as Methods 4.5; unique variance
U_S and U_T; per-voxel index (U_S − U_T)/(U_S + U_T), median over reliable
voxels per ROI, on V1v…hV4 and on the Kastner2015 and `streams` regions of
Extended Data Fig. 4. Plus the cortex-optimal w of the lossy sum z(w), as in
Extended Data Fig. 5.

### Models
- `rgb_tsn` seeds 0–2: both trunks receive the identical three-channel image
  and stay bit-identical, so it is a single ResNet-18 on the untransformed
  image — the no-diet, no-split case.
- `dkl_bcs_single` seeds 0–2 (ResNet-18) and `dkl_bcs_single34` seeds 0–2
  (ResNet-34): one trunk, diet on luminance, both channels stacked.
- Validation: the two-stream `dkl_bcs` seeds 0–2 with y = [y₁, y₂]
  concatenated (1,024-d), where the factorisation should recover the streams.
- Field reference (behaviour): torchvision ImageNet ResNet-50, ResNet-152,
  VGG-16, ViT-B/16, ConvNeXt-S/B, with their own heads. Cortex: ResNet-50.

## Predictions, written before running

- **P1 (span exists without a split).** For every single-trunk model the
  fixed-head rate rises monotonically with w (Spearman ρ(w, rate) > 0 in every
  seed) and the span max − min over w is ≥ 0.20 on the primary estimator. The
  two-stream affine span is 0.50. Failure: span < 0.10 in most models, in
  which case the readout span is a property of the imposed split and the
  paper says so.
- **P2 (span survives a refit).** For the lossy sum, the refitted linear and
  two-layer spans are ≥ 0.15, monotone in w, and agree with each other within
  0.05.
- **P3 (validation).** In the two-stream concatenation ≥ 75% of the S half are
  stream-1 units, per seed.
- **P4 (cortex loads on T beyond V1).** The S − T index falls from V1 to hV4
  (hV4 − mean(V1v, V1d) < 0) in ≥ 7/8 subjects for `rgb_tsn` and
  `dkl_bcs_single34`, and every extrastriate group of Extended Data Fig. 4 is
  more T-explained than V1. Failure: no ordering, in which case the cortical
  dissociation needs the architectural split and the paper says so.
- **P5 (cortex prefers the T-heavy mixture).** The cortex-optimal w of the
  lossy sum is lower in extrastriate cortex than in V1 in ≥ 7/8 subjects.

Statistics as in Methods 4.7: exact paired Wilcoxon over eight subjects
(floor .0078); seeds reported separately and never pooled with subjects; no
correction claimed, claims resting on complete separation and replication.
Whatever the outcome, every prediction is reported.

## Outcomes (appended 2026-09-15, after the runs; Extended Data Tables 13–14, Fig. 6)

- **P1** — direction met in every single-trunk network and seed (ρ ≥ 0.71);
  magnitude not met: spans 0.09–0.21 (means 0.19, 0.19, 0.14 for our three
  trunk families; 0.08–0.21 for the six ImageNet networks) against the 0.20
  threshold. Two-stream concatenation 0.46.
- **P2** — direction met (ρ 0.72–0.98); magnitude not met: linear 0.05–0.11,
  two-layer 0.05–0.13, validation accuracy unchanged.
- **P3** — met: 99% of the shape-leaning half is stream 1 in every seed.
- **P4** — hV4 − V1 < 0 in ≥ 7/8: met in the diet ResNet-34 (7/8, p = .016),
  not in the no-diet ResNet-18 (5/8) or ResNet-50 (5/8). Extrastriate more
  T-explained than V1: met in the no-diet ResNet-18 (8/8, p = .0078) and the
  diet ResNet-34 (7/8, p = .016); reversed in ImageNet ResNet-50 (1/8, +0.24,
  p = .039), where the texture-leaning half already carries V1–hV4 and ventral
  cortex and lateral/dorsal cortex are balanced.
- **P5** — met in the no-diet ResNet-18 (7/8, p = .023); not met in the diet
  ResNet-34 (4/8, optimum ≈ 0.55 everywhere) or ResNet-50 (3/8, lateral and
  dorsal cortex prefer the shape-leaning half).
