# Pre-registration: exemplar-held-out probes and spans

Written 2026-09-21, before `src/analysis/generality/exemplar.py` or `src/analysis/behavior/stream_features.py`
existed or had run on any network. Committed before the analysis. Prompted by the seventh referee
simulation (`docs/NMI_REVIEW_v7_2026-09-21.md`, Weakness 1 and Major comment 1), and **post hoc with respect
to every earlier plan** (`PLAN_GENERALITY`, `PLAN_SUBSPACE`, `PLAN_ZOO`, `PLAN_KAPPA`, `PLAN_SPAN`). It will be
labelled post hoc wherever it is reported. The pre-registered part is the estimator set and the prediction set.

## The problem

The 1,200 cue-conflict images (`<shape><i>-<texture><j>.png`, i ≠ j) are built from **160 object exemplars**
(10 per category, each in 3–12 images) and **48 texture exemplars** (3 per category, each in 15–35 images).
Two parts of the analysis ignore this structure:
- The linear probe (Methods 4.3; `stream_shape_texture.probe`) uses `StratifiedKFold` on the label.
- The zoo's held-out halves (Methods 4.8; `unit_factor.stratified_halves`) are stratified on the shape ×
  texture *label* pair.

As a result, 98% of scored images share their object exemplar with the fitting set, and 100% share their
texture exemplar. Texture exemplars recur about three times as often as object exemplars, so exemplar
memorisation pays off more for the texture label.

## Known before this plan was written

These results were produced by the v7 referee and editor checks. They are recorded here so that nothing below
is presented as a prediction of them.

- **K1. Exemplar-balanced probe, PCA fitted on all 1,200 images** (Slurm job 26513461;
  `/scratch/byrong/mpk/review_v7/results/2026-09-21_balanced_probe/`). The design uses 3 exemplars per class
  for both labels, scored leave-one-exemplar-out with image counts matched, over 10 draws.
  - The 39 ImageNet networks run from −0.071 to +0.112 (median +0.011).
  - The 5–95% interval spans zero in 31 of the 39.
  - SIN ResNet-50 is highest at +0.162, and ViT-B/32 is next at +0.112.
  - The ordering agrees with the paper's probe at ρ = 0.80.
- **K2. Exemplar-grouped probe** (`StratifiedGroupKFold`, 5 folds; R1). The bias is positive in 38 of 38
  networks read. This design favours shape, because it trains on about 8 object exemplars against 1–2
  texture exemplars per class.
- **K3. Texture-exemplar-disjoint halves** (1|2 texture exemplars per class; R1, four networks). Each network
  gives the paper's span, then the disjoint span:
  - ResNet-50: 0.290 → 0.172
  - VGG-16: 0.268 → 0.114
  - ViT-B/16: 0.282 → 0.174
  - SIN: 0.166 → 0.147
- **K4. Test–retest reversal of the trained ranking** (binomial resampling; R1): 12.9% [9.2%, 17.1%] of the
  703 pairs.

## Networks and inputs

**Zoo.** The 42 networks of `PLAN_ZOO`: stored features in `/scratch/byrong/mpk/generality/features/
2026-09-15_zoo`, all-image bases and gates in `.../behaviour/2026-09-15_zoo`. Across-network statistics use
the **38 ImageNet-trained (IMAGENET1K_V1) networks**. The V2 ResNet-50 and the three Stylized-ImageNet
ResNet-50s are reported separately.

**Our trunks** (stored features in `.../features/2026-09-14_gen`, 3 seeds each):
- `concat_dkl_bcs`: the two-stream diet model, streams concatenated, with `stream_of_unit`;
- `run_dkl_bcs_single`: the dieted single ResNet-18;
- `run_dkl_bcs_single34`: the dieted single ResNet-34;
- `run_rgb_tsn`: the no-diet single ResNet-18.

**Streams of the two-stream model.** These are the 20 configurations of `slurm/shape_texture.sh`, which
produced Table 1, Fig. 2 and ED Table 4. Their per-stream penultimate features were never stored. They are
re-extracted from the same checkpoints and front-end settings by a new script (`stream_features.py`) that
calls the same functions (`dkl_frontend.magno_parvo_inputs`, `features.build_model`,
`features.extract_stream_features`). Validation: for the three diet seeds, the extracted features must
reproduce the stored `concat_dkl_bcs` features' stream blocks to within float32 tolerance, and the paper's
ungrouped probe on them must reproduce `results/shape_texture/*.json` exactly.

## Estimators (fixed here)

Exemplar identity is parsed from the image stem. The object exemplar is `<shape><i>`; the texture exemplar is
`<texture><j>`.

### Spans

The bases, the gain and the scoring are the manuscript's own: `subspace_factor.subspace_bases` (Löwdin),
`gain_features` and `scoring.Scorer`, unchanged. Only the fit/score partition changes.

- **E0, reference.** The paper's pair-stratified halves, 10 splits in both directions, with seed 0 and the
  same RNG path. It must reproduce the stored held-out span (`<tag>_subspace.json`,
  `summary.primary.span_mean`) to within 1e-9. If it does not, the task fails.
- **E1, primary: fully exemplar-disjoint 3×3 folds.** In each of 10 repetitions (seed r):
  - Each texture category's 3 exemplars are randomly assigned to folds 0–2. Each shape category's 10 object
    exemplars are randomly permuted and assigned to folds by position mod 3, giving 4/3/3.
  - For fold k, the scored images have both their texture fold and their object fold equal to k, and the
    fitting images have both folds ≠ k.
  - The fitting set therefore always holds 2 texture exemplars per category, and roughly 6–7 object
    exemplars.
  - The three folds' scored images are pooled into one curve per repetition, about 400 images.
  - The span is the mean over repetitions of each repetition's (max − min) over the 21-point w grid, as in
    `summarise`. It is reported on both estimators, primary (decided-image rate) and forced two-way.
- **E2, texture-exemplar-disjoint only.** 3 folds by texture exemplar, with object exemplars shared. Every
  image is scored once per repetition, over 10 repetitions.
- **E3, object-exemplar-disjoint only.** 3 folds by object exemplar, with texture exemplars shared. 10
  repetitions.

**Gating.** The gate is the stored all-image gate (`<tag>_gain_accuracy_subspace.json`, top-1 on the ILSVRC
subset within 5 points of w = ½). Where the whole sweep is inside the gate (all 38 ImageNet networks), the
gated span equals the span. For our trunks, the gated span is taken over the w inside the stored gate.

**Null.** The random-subspace null of `PLAN_SPAN` (ED Table 18) is label-free, so a split cannot leak into
it. It is compared as stored: each network's largest gated span over its ten draws.

**Ranking.** The trained rate is taken on all 1,200 images, as in the manuscript. The shape-end rate is the
E1 mean curve at the upper end of the gate. The reversal fraction is the share of the 703 pairs of the 38
networks whose order differs between the two rankings, on both estimators.

**Noise floor.** The noise floor is the trained ranking's own test–retest reversal:
- Draw two independent bootstrap resamples of the 1,200 images, shared across networks.
- Rank the networks by their trained rates in each resample, and count reversals between the two rankings.
- Repeat 2,000 times (seed 0) and report the mean and the 95% percentile interval.

"Resolved pairs" are the pairs whose paired bootstrap 95% interval for the trained-rate difference excludes
zero. The reversal fraction among resolved pairs is reported beside the total.

### Probes

- **Q0, reference.** The paper's probe: PCA-256 on all images, then `StratifiedKFold(5)` and multinomial
  logistic regression (C = 1). It must reproduce the stored `<tag>_probe_bias.json` (zoo) or
  `results/shape_texture/*.json` (streams) exactly.
- **Q1, primary: exemplar-balanced.**
  - The shape probe uses 3 of each category's 10 object exemplars, drawn at random. The texture probe uses its
    3 exemplars.
  - Texture images are subsampled per category to the shape subset's count.
  - Both probes are scored leave-one-exemplar-out: 3 folds, trained on 2 exemplars per category and tested on
    the third.
  - **PCA and scaling are fitted inside each fold** on the training images, with k = min(256, n_train − 1).
  - 20 draws (seeds 0–19). Report the mean bias and the 5–95% interval over draws.
- **Q1b.** As Q1, but with PCA fitted on all 1,200 images. This is K1's estimator, re-run in the pipeline for
  provenance.
- **Q2.** Exemplar-grouped `StratifiedGroupKFold(5)`, grouped by the probed label's exemplar, with PCA inside
  the folds. This is K2's design.

**Chance** is 1/16 for every probe. Accuracies are reported beside every bias.

## Predictions

All predictions below apply to point estimates. Every across-network statistic is descriptive: Spearman ρ or
Kendall τ, with a 95% bootstrap interval over networks (10,000 resamples, seed 0).

**Zoo spans (the 38 ImageNet networks unless stated).** The E1 values are not known for any network.

- **P-X1 (the span exists on new exemplars).** The E1 primary span exceeds the network's largest stored
  random-subspace span in at least 40 of the 42 networks.
- **P-X2 (monotone).** ρ(w, E1 mean curve) ≥ 0.9 in all 42 networks.
- **P-X3 (magnitude).**
  - (a) The median E1 primary span is ≥ 0.15.
  - (b) Fewer than 19 of the 38 reach 0.20, which is `PLAN_ZOO`'s threshold. In other words, that threshold
    is predicted to fail for the majority on new exemplars.
- **P-X4 (which exemplars matter).** The median E2 span (texture-disjoint) is lower than the median E3 span
  (object-disjoint). The inflation is predicted to come mainly from texture exemplars.
- **P-X5 (re-ordering beyond noise).** The E1 reversal fraction on the primary estimator exceeds the upper
  bound of the trained ranking's 95% test–retest interval.
- **P-X6 (the rate does not predict the span).** |ρ(trained rate, E1 span)| ≤ 0.3. The paper reports +0.04
  for E0.

**Zoo probes.** Q1b is K1. The Q1 values, with PCA inside the folds, are not known.

- **P-X7 (the representations are not texture-biased).**
  - The median Q1 bias over the 38 lies within [−0.10, +0.10].
  - Fewer than 10 of the 38 have a 5–95% interval entirely below zero.
- **P-X8 (the ordering survives).** ρ(Q1, Q0) ≥ 0.7 over the 42.
- **P-X9 (the SIN network).** SIN ResNet-50 has the highest Q1 bias of the 42. This is confirmatory: under Q1b
  (K1) it is already known to be highest.
- **P-X10 (the texture share).** ρ(texture share, Q1 bias) ≤ −0.5 over the 38, where the texture share is
  −mean e_T / (mean e_S − mean e_T), taken from the stored `<tag>_span_model.json`. The paper reports −0.81
  for Q0.

**Streams and trunks (3 seeds each, paired by seed).** None of these values is known.

- **P-X11 (the stream contrast survives).** In the diet model, the Q1 bias of magno minus the Q1 bias of parvo
  is > 0 in 3 of 3 seeds.
- **P-X12 (the parvo stream is texture-leaning).** The diet parvo stream's Q1 bias is < 0 in 3 of 3 seeds.
  This is the least certain prediction here.
- **P-X13 (the input supplies the magno lean).** The untrained magno stream on the diet front end
  (`untrained_blur_cs`) has a Q1 bias > 0.
- **P-X14 (learning moves the parvo stream toward texture).** The diet parvo stream's Q1 bias is below the
  untrained parvo stream's (`untrained_blur_cs`) in 3 of 3 seeds.
- **P-X15 (division of labour in the representation).** The dieted single ResNet-18's Q1 bias is below the
  diet magno stream's in 3 of 3 seeds.
- **P-X16 (the 2×2 interaction survives).** The carrier × transform interaction of Fig. 2b–c, computed
  exactly as the manuscript computes it but on Q1 biases, is ≥ +0.20. The manuscript reports +0.73 for Q0.

## How the outcomes change the manuscript (decided now)

- **P-X1 and P-X2 hold.** "The rate moves on new exemplars" stays in the abstract. The magnitude is restated
  as the E1 range and median, and "0.20–0.50" is replaced. If P-X1 fails in more than 2 networks, the zoo
  claim is restated as existing only in the networks where it holds, and the abstract drops the zoo range.
- **P-X5 holds.** The abstract says "re-orders X% of pairs, N times the trained ranking's own test–retest
  reversal". **If it fails**, the ranking-reversal clause leaves the abstract, and the reversal is reported
  in Results against its floor.
- **P-X7 holds.** Every "texture-biased representation" statement for the ImageNet networks is replaced with
  the Q1 values, and the zoo half of the argument is re-led as "roughly balanced representations under
  texture-biased readouts". The SIN network's "its bias sits in the representation" is kept only in the
  ordinal form P-X9 supports.
- **P-X11 fails.** The two-stream "opposite directions" claim is restated at the behavioural level only, and
  Table 1's probe column is replaced.
- **P-X12 fails.** "The texture-biased stream" leaves the abstract and Results 2.3, in favour of "the
  chromatic stream" or "the parvo stream". The stream's texture bias is then described by its behavioural
  readout alone, which the manuscript's own thesis says is a readout property.
- **P-X15.** Whichever way it comes out, the single-trunk comparison enters Results 2.2 and the abstract's
  magno sentence is rewritten to name the second stream.
- **Q0 always stays in the manuscript as the reference**, labelled "exemplar-shared". No Q0 number is dropped.

## Compute

- **Array A** (zoo and trunks, stored features): 54 tasks on `main`, CPU only, 4 CPUs and 8 GB each, logs
  `%A_%a`. Outputs go to `/scratch/byrong/mpk/exemplar/results/2026-09-21_exemplar/`.
- **Array B** (stream features and stream probes): 20 tasks on `main`, CPU only, 8 CPUs and 16 GB, the same
  sizing as `shape_texture.sh`, which ran at ≤ 32 GB. Features go to
  `/scratch/byrong/mpk/exemplar/features/2026-09-21_streams/`.
- **Aggregation** writes `results/figures_v2/exemplar_zoo.json` and `exemplar_streams.json`. The verifier
  gains checks that re-derive every number reported from them.

## What will not be done

- No retraining.
- No change to the w grid, the gate, the bases or the scorer.
- No new stimulus set.
- No exemplar-held-out rerun of the unit rule or of the cortical analyses. The cortical fits use NSD images
  and never see cue-conflict splits, but the unit split's η² is fitted on all cue-conflict images; its
  exemplar calibration is noted as a limitation.
- The behaviour-preferred weight (κ vs κ/κ_max) is a separate item and is not part of this plan.

## Clarifications recorded before any result existed (2026-09-21, same day)

Written after `exemplar.py` was drafted and before any output of it or of `stream_features.py` existed; the
one-network test job (26513795) had not finished.

1. **The stream features of Table 1 are not the stored penultimate features.** `stream_shape_texture.py`
   probes `features.extract_stream_features`, the four ResNet stages each pooled to 4 × 4 and concatenated
   (15,360 dimensions per stream), not the 512-dimensional penultimate block stored in `concat_dkl_bcs`. The
   cross-check "extracted features reproduce the stored concat stream blocks" therefore cannot hold and is
   withdrawn. The extraction is validated by Q0 reproducing `results/shape_texture/*.json` exactly (on the
   float32 features, as the original script ran). P-X11 to P-X14 and P-X16 are scored on these stage-pooled
   features, because they are the features behind the numbers they test.
2. **P-X15 is scored on penultimate features**, the only level at which both the single trunk and the
   magno stream are stored: the Q1 bias of `run_dkl_bcs_single_seed{s}` (512 units) against the Q1 bias of the
   stream-1 units of `concat_dkl_bcs_seed{s}` (512 units), paired by seed. The stream-level Q1 on
   stage-pooled features is reported beside it but does not score the prediction.

## Note after the one-network test runs, before the full arrays (2026-09-21)

**The two test elements were seen before the full run.** They are `tv_resnet18` (job 26513795) and
`dkl_bcs_seed0_trained` (job 26513798). Both belong to scored sets, and their outputs are recorded here so that
nothing is presented later as unseen.
- Every reproduction check passed exactly: E0 at |d| = 0, and Q0 at |d| = 0 for the zoo network and for both
  streams.
- ResNet-18: E0 span 0.316, E1 0.200, E2 0.210. Probes: Q0 −0.528, Q1 −0.338 [−0.422, −0.234] (shape 0.249,
  texture 0.586), Q1b +0.008, Q2 −0.139.
- Diet seed 0, magno stream: Q0 +0.810, Q1 +0.337 (shape 0.418, texture 0.081), Q1b +0.092.
- Diet seed 0, parvo stream: Q0 −0.482, Q1 −0.173 (shape 0.271, texture 0.444), Q1b −0.047.

**A preprocessing sensitivity, found post hoc.** Q1 and Q1b disagree in sign for ResNet-18, so the balanced
design was re-run on ResNet-18 and SIN ResNet-50 under eight preprocessings, three draws each (exploratory;
script kept under `/scratch/byrong/mpk/review_v7/scripts/`). The variants were:
- PCA on all images with k = 16, 64 or 256, each standardised;
- PCA-256 on all images, not standardised;
- raw standardised features;
- PCA inside the folds with k = 16, 64 or n − 1.

**Q1b is the outlier.** Q1b is PCA-256 on all images, then standardised; it is also K1's estimator and the
paper's own. Every other variant puts ResNet-18 at −0.21 to −0.56, with texture decoded on held-out exemplars at
0.49–0.72 against shape at 0.16–0.32, and SIN at +0.17 to +0.26. Q1b alone gives −0.014 and +0.198.

The likely mechanism is this. With about 250 training images, standardising 256 components gives the many
low-variance components the same scale as the few components that carry category information, and the texture
probe loses most of its generalisation.

**Consequences:**
- **K1's reading is withdrawn.** K1 read the zoo's ImageNet representations as "roughly neutral once exemplars
  are held out". That reading came from this preprocessing, not from the representations. The same holds for
  the v7 review's statement that the paper's −0.30 to −0.66 "measures exemplar memorisation", and for the story
  note built on it. Both files carry a correction.
- **P-X7 and P-X9 were written with K1 in view.** They are left exactly as written and will be scored as
  written.
- **Two estimators are added to every probe**, as post-hoc sensitivities that are reported but not scored:
  **S64** (PCA-64 on all images, standardised) and **S256n** (PCA-256 on all images, not standardised).
- The pre-registered Q0, Q1, Q1b and Q2, the spans and every prediction are unchanged.
