# Extended Data

**Companion to `PAPER_DRAFT.md`, created 2026-09-06.** This file holds material
moved out of the main text during the length pass, so that Results carries the
argument and the supporting apparatus — control tables, seed coverage,
denominator and checkpoint caveats, and the qualifications each result carries —
remains available in full.

**Nothing here was cut for being inconvenient.** Every block below is the
original text, moved verbatim; the main text cites the section it went to. The
failures, the degenerate controls, the single-seed arms and the bounded nulls
are all still reported — they are reported *here* rather than interleaved with
the claims they qualify. Where the main text compresses a passage, the
uncompressed original is the corresponding block in this file.

## E1 — §2.1, the two streams and their routes

Moved from §2.1: the single-model qualification on the untrained/trained comparison, and the full statement of the bit-identical-initialisation identity including what it does *not* establish.

*This comparison is single-model.* The untrained references carry no training
seed, and the trained rows of Table 1 are seed 0 (the 3-seed trained values are
rows a of that table). The route asymmetry is large relative to the trained arm's seed
range of 0.014, but it is reported at one seed per cell and should be read that
way.

**Every divergence between the streams is attributable to their inputs, by
construction.** `TwoStreamNetwork` initialises the two trunks bit-identically:
0 of 120 stream-1/stream-2 tensor pairs differ at initialisation, for every
condition. Given identical inputs, identical initialisation yields identical
gradients, and the trained `rgb_tsn` control -- both streams fed the same image
-- returns all 120 pairs still bit-identical after 150 epochs and a stream
difference of exactly **+0.0000** on every fold of every seed. This is an
algebraic identity, and we report it as one: it is *not* evidence that a
two-stream architecture fails to specialise spontaneously, a hypothesis this
architecture cannot test (Limitations). What it establishes is that no stream
difference reported anywhere in this paper can be a lucky draw in one stream's
weights -- the input is the only thing that can differentiate them. That is why
§2.4's front-end 2x2 can carry causal weight.


## E2 — §2.2, the cortical instrument

Moved from §2.2: the full raw-statistic paragraph, the three-seed replication of the in-silico cue conflict, the five supporting points with the six-control table, the koniocellular substitution, the per-control seed coverage, and the RSA comparison in full.

The raw statistic agrees: **+0.616 with the diet against −0.033 without** (and
+0.658 / −0.039, +0.632 / −0.053 in seeds 1 and 2, p=.0078 and 8/8 throughout).
**We draw no inference from the sign of the no-diet values.** An earlier version
described the raw statistic as separating the arms "by sign rather than by
degree" and read the no-diet arm as marginally texture-organised; both are
withdrawn. The values themselves are sound — the cell has been
re-derived independently from the stored per-band η² fields, and all three
no-diet seeds are negative on the raw statistic — but they sit near zero, no test
addresses which side of zero they fall on, and the claim this section makes (that
the arms separate, at the exact Wilcoxon floor in 8/8 subjects on both
statistics) does not need them to fall on any particular side.

**Seed replication (n=8 subjects × 3 seeds).** Because §2.5 showed that
single-checkpoint levels are not always reproducible on this project's
instruments, the cortical contrast was re-run for seeds 1–2 of both arms:

| seed | diet | no diet | difference | p | subjects |
|---|---|---|---|---|---|
| 0 | +1.534 | +0.110 | +1.424 | .0078 | 8/8 |
| 1 | +1.566 | +0.098 | +1.468 | .0078 | 8/8 |
| 2 | +1.558 | +0.083 | +1.474 | .0078 | 8/8 |

The diet arm's seed range here is **0.032**, against **0.120** on the
dimensionality ratio of §2.5 — a 3.7-fold difference in seed stability across the
same three training runs. The two are not the same weight files: this section is
scored at the final checkpoint and §2.5 at the best saved one (see Methods), so
the comparison is between instruments applied to the same runs, not to the same
checkpoints. (An earlier version of this comparison gave 28-fold, using a §2.5
seed range that was itself a final-checkpoint artefact; Supplementary Note S1.)
The raw statistic agrees on every seed, so the two-statistics rule is satisfied three
times over.

Five points make this more than a single fitted number:

- **Training decides the direction; the input only supplies the starting point.**
  An untrained network on the diet front end already sits at −0.471, so filtering
  alone supplies roughly half the gradient before any learning. Training with the
  diet takes it to −0.862 (1.8× amplification); training without it — from an
  untrained no-blur baseline of −0.488 — ends at −0.101, a **79% reduction**. The
  diet does not create the gradient. It decides whether training amplifies or
  destroys the one the input hands it, which is the §2.5 mechanism appearing on a
  second, independent instrument.
- **It replicates across codebases.** Our standalone-trained model and the
  independent thesis checkpoint give −0.8618 and −0.8593 — two trainers, two
  augmentation pipelines, two seeds, agreeing to 0.0025 — and are statistically
  indistinguishable from one another on both statistics (p = .84), which is the
  behaviour a convergent replication should show (§17).
- **It survives every front-end control, on both statistics.** Each control is
  compared against the diet arm, paired within subject, n=8:

  | control | normalised | raw | p (both) | subjects |
  |---|---|---|---|---|
  | no diet | −0.101 | −0.034 | .0078 | 8/8 |
  | `swap` | −0.027 | −0.043 | .0078 | 8/8 |
  | `parvo_highsf` | −0.119 | −0.108 | .0078 | 8/8 |
  | untrained, diet front end | −0.471 | −0.118 | .0078 | 8/8 |
  | Gabor bank, diet front end | −0.367 | −0.042 | .0078 | 8/8 |
  | Gabor bank, no-blur front end | −0.621 | −0.101 | .0078 | 8/8 |

  See §2.7 for what the `swap` and `parvo_highsf` controls cost, and the
  qualification they force on how this control result may be stated.
- **It is not explained by a global stream preference.** Over retinotopic voxels
  the diet-trained and no-diet models sit on *opposite* sides of the global
  stream-bias axis (−0.495 against +0.849), yet the strong regional gradient is
  in the model whose global preference points the other way (§17.2). A model that
  simply preferred magno everywhere could not produce this.
- **The chromatic side is specific to L−M, not to chromaticity as such.**
  Substituting the koniocellular axis S−(L+M) for L−M in stream 2, holding the
  luminance stream bit-identical, makes the ventral prediction **worse, and
  worse where colour lives** — hV4 −0.029, VO −0.024, the pre-declared colour
  set −0.026, each at p=.0078 with 0/8 subjects positive on both statistics —
  and the deficit is concentrated rather than global (colour set minus the other
  17 ROIs, −0.014, p=.0078, 0/8), unchanged when the konio channel keeps its
  native 2.05× contrast advantage. Both cone-opponent axes have now been tested
  against colour-biased cortex and neither predicts it, so what the parvo stream
  contributes at the ventral end is specific to L−M without being a claim about
  colour selectivity. These fits are **untrained and single-seed**, and no third
  stream was added. Full result, pre-registration and limits: Supplementary
  Note S2.

*Seed coverage, stated plainly.* The diet and no-diet arms have three seeds each
on this instrument. The `swap` and `parvo_highsf` cortical fits are **seed 0
only**, so those two control comparisons rest on one training run per condition;
their behavioural and dimensionality counterparts are seeded three times (§2.4,
§2.5) and agree, but the cortical control itself is n=1 seed and should be read
that way.

**This is the step the paradox needs, and it is the step a correlational
analysis cannot take.** Asked the same question with representational
similarity analysis -- correlating each stream's layers against ROI-level
geometry -- the answer is a *global* stream preference rather than a regional
split: one stream correlates better everywhere, ventral and dorsal alike, and
the correlation cannot separate that general factor from regional structure.
Partitioning unique variance removes it. The global-preference control above is
the demonstration: the diet-trained and no-diet models sit on opposite sides of
the global stream-bias axis (-0.495 against +0.849), and the strong regional
gradient belongs to the model whose global preference points the *other* way.

### E2.1 — the artefact floor, and a correction to how we first computed it

**What the two statistics of §2.2 actually are.** The word *raw* has caused
three separate errors in this analysis, so it is defined here once and used
nowhere loosely. Both of §2.2's statistics are computed on the
noise-ceiling-divided unique variances (`banded_ridge.variance_partition`
returns `unique_magno = unique_m / nc`; `tmp_analysis/visualrois.py` reads those
and prints them under the header *"Median unique variance / NC"*):

| §2.2 label | construction | seed-0 diet | seed-0 no diet |
|---|---|---|---|
| **normalised** | ratio index, median over voxels of (uₘ−uₚ)/(uₘ+uₚ), then hV4 − mean(V1v,V1d) | −0.834 (paper −0.862) | −0.079 (paper −0.101) |
| **raw** | difference, median over voxels of (uₘ−uₚ), then hV4 − mean(V1v,V1d) | −0.166 (paper −0.176) | −0.027 (paper −0.034) |

So *normalised* means normalised **by (uₘ+uₚ)**, not by the noise ceiling —
both statistics are already nc-divided, and the ratio index is in any case
exactly invariant to that division (it is unchanged to 4 dp when recomputed on
the stored `unique_magno_raw` arrays). The statistic is a **hV4 − V1 contrast**,
not a value at hV4 and not a median over the ROI profile; all four published
numbers reproduce under that reading and under no other.

**The floor.** Synthetic voxels were built from the fitted feature spaces, the
true magno and parvo unique variances were solved so their difference is exactly
zero in every ROI (verified: max |true contrast| = 0), and the measured per-ROI
noise ceiling was imposed. Scored through the identical contrast:

| condition | normalised (ratio index) | raw (difference) | monotonicity ρ |
|---|---|---|---|
| **B_manufacture** (truth = 0) | **+0.051**, 8/8 positive, p = .0078, range +0.007 to +0.094 | **+0.0009**, 4/8, p = .84 | **+0.68** |
| **A_recovery** (truth graded) | +1.700 (truth +1.355; **×1.26**) | +0.499 (truth +0.700; **×0.71**) | **+1.000** in 8/8 |

Two things follow, and a third does not. The artefact under the real
noise-ceiling profile is **positive and rising** with hierarchical rank where the
reported effect is negative and falling: wrong sign twice over, 6% of the
effect's magnitude on the normalised statistic and indistinguishable from zero on
the raw one. It cannot manufacture the result — it opposes it. And the estimator
is not inert: given a true gradient it recovers the ROI ordering perfectly in
every subject. What does **not** follow is that −0.862 is a lower bound. The
recovery condition is not magnitude-faithful and the two statistics are biased in
*opposite* directions — the ratio index inflated 1.26×, the difference compressed
to 0.71×. An earlier draft of this section claimed the observed contrast was a
floor rather than a ceiling; that was wrong, and §2.2 now reads the result as
sign-and-ordering rather than as a calibrated effect size.

**How we first got this wrong, twice.** The first version of this simulation
used a hardcoded noise-ceiling ramp that *ascends* from V1v to hV4 (0.15 → 0.45).
The measured profile does the opposite — it declines (0.408 → 0.322, ρ = −0.893,
p = .007, n = 8; independently confirmed at ρ = −0.929) — so that run drove the
nuisance backwards and inverted the artefact's sign, producing a negative
contrast that read as a serious threat to §2.2 for as long as it stood. It was
not a threat; it was this same artefact pointing the wrong way. Its alarming
magnitude was additionally a low-resolution effect: re-run at the 1,000
components used here it shrinks by more than an order of magnitude, so no number
from that pilot is quoted. The second error was ours in the opposite direction —
the floor was first reported as a median over the ROI profile (+0.126) rather
than as §2.2's hV4 − V1 contrast (+0.051), which is a 2.5× overstatement of the
artefact and was corrected before it left the draft. The mechanism in both cases
is the 1/noise-ceiling normalisation. Verifying that a simulation is internally
correct is not verifying that it simulates the right thing, and neither is
checking that a number reproduces — a number can reproduce exactly and still be
the wrong statistic.

Computed on the no-diet feature geometry, the harder case, since that model
predicts BOLD better (§2.7); the diet-arm geometry is running (job 26442320) and
the artefact depends on feature geometry, so it may differ. Code:
`tmp_analysis/vp_recovery_realnc.py`, checked by
`tmp_analysis/verify_new_claims.py`; results under `results/vp_recovery_realnc/`.

**Two limits on the floor, both disclosed rather than resolved.** First, the
floor is computed with a *different estimator* from the effect it bounds. §2.2
takes the median over voxels of the per-voxel ratio (median-of-ratios); the
recovery simulation stores only per-ROI medians of uₘ and uₚ, so its ratio is
necessarily a ratio-of-medians. On real data those two differ at the contrast
level by 0.017 (diet) and 0.009 (no diet) — about a third of the floor itself.
The +0.051 floor therefore carries an estimator-mismatch uncertainty of that
order, and fixing it properly means re-running condition B recording the
per-voxel ratio median. Until that is done the floor is approximate, and the
comparisons in §2.2 (16× for the diet arm, 1.5× for no-diet) should be read with
that tolerance. Second, the ×1.26 and ×0.71 magnitude-fidelity figures are
**ratios of the across-subject means** of the estimated and true contrasts
(n = 8). An OLS slope of estimated on true, pooled over the 56 ROI × subject
cells, gives 1.34 rather than 1.26 for the ratio index (r = +0.99); the two
estimators agree on direction and disagree by 6%, so no magnitude claim is made
from either, and neither is quoted without naming which it is. That ambiguity —
a number reported without its estimator — is the same failure that produced this
entire correction.

### E2.2 — the provenance search behind §2.2's superseded table

§2.2's table previously reported normalised −0.862 / −0.802 / −0.781 and raw
−0.176 / −0.151 / −0.160. Those values do not regenerate. All twelve were more
negative than every recomputation, by 0.006 to 0.036, and the previously
reported dz of −11.70 is not attainable at all. The configurations searched:

| degree of freedom | values tried |
|---|---|
| noise-ceiling threshold | 0.05, 0.1, 0.15, 0.2, 0.3 |
| (uₘ+uₚ) > 1e-3 mask | on, off |
| minimum voxels per ROI | 0, 20 |
| voxel aggregation | median-of-ratios, ratio-of-medians |
| negative unique variance | as-is, clipped at 0, absolute denominator, voxels dropped |
| contrast definition | hV4−mean(V1v,V1d), hV4−mean(V1–V3), hV4 alone, V3−V1 |
| V1 reference | mean(V1v,V1d), V1v only, V1d only, mean of six V1–V3 banks |
| across-subject aggregation | mean, median |
| output directory | `results/encoding/`, `results/encoding_cc/` |

The closest joint match leaves the normalised values 0.014–0.021 off and reaches
dz = −8.74 against the −11.70 reported. Two findings came out of the search and
both are kept. `results/encoding_cc/` is the directory that holds all three
seeds for both arms — `results/encoding/` has the diet arm at seed 0 only — and
where the two overlap they are numerically identical, so `encoding_cc` is now
the declared source for §2.2. And the contrast reading is confirmed: it is
hV4 − V1 (`visualrois.py:72-75`), since no other definition reaches the right
order of magnitude on all four seed-0 values.

**The subj01 question, unresolved.** Per-subject no-diet contrasts at seed 0
(normalised): −0.249, −0.070, −0.027, −0.053, −0.043, −0.084, −0.030, −0.073.
Subject 01 is 4.65× the sibling median. It is *not* an outlier in the diet arm
(1.23×), and the same asymmetry holds on the raw statistic (3.45× versus 1.11×).
`results/encoding/quarantine/README.md` condemned a file with exactly this
signature on 2026-09-01 and required a provenance-stamped replacement to be
copied in rather than the original restored; the present file carries no
provenance stamp, and the quarantine directory now holds only the README. We
cannot tell which file this is. Sensitivity, both statistics, all three seeds:

| statistic | seed | no diet, n=8 | no diet, n=7 | diet | p (n=8) | p (n=7) | separation |
|---|---|---|---|---|---|---|---|
| normalised | 0 | −0.079 | −0.054 | −0.834 | .0078 | .0156 | 8/8 |
| normalised | 1 | −0.096 | −0.085 | −0.785 | .0078 | .0156 | 8/8 |
| normalised | 2 | −0.049 | −0.031 | −0.745 | .0078 | .0156 | 8/8 |
| raw | 0 | −0.027 | −0.021 | −0.166 | .0078 | .0156 | 8/8 |
| raw | 1 | −0.031 | −0.030 | −0.145 | .0078 | .0156 | 8/8 |
| raw | 2 | −0.023 | −0.021 | −0.153 | .0078 | .0156 | 8/8 |

Every row keeps complete separation and the exact Wilcoxon floor for its sample
size. The exclusion only shrinks the no-diet magnitude, which moves that arm
*closer* to the artefact floor and therefore supports the reading §2.2 adopts.
The claim does not depend on the answer; the number does, so both are given.


## E3 — §2.3, the readout sweep and the single-stream control

Moved from §2.3: the tabulated sweep (now Figure 3a), the endpoint and decided-image caveats, the single-stream control table with its seed-level separation, and the two limits on that control.

| `w` (weight on the magno stream) | 0.00 | 0.25 | 0.50 | 0.75 | 1.00 |
|---|---|---|---|---|---|
| final checkpoints | 0.419 | 0.508 | **0.675** | 0.827 | 0.879 |
| best saved checkpoints | 0.399 | 0.508 | **0.655** | 0.807 | 0.904 |

*Validation and two caveats.* At w = 0.5 the sweep reproduces the published
per-seed values **exactly** (0.6922 / 0.6861 / 0.6464 at the final checkpoint;
0.5942 / 0.7177 / 0.6522 at the best saved), which is the anchor that shows the
sweep and the main pipeline are the same computation. The endpoints are *not*
formally identical to the `magno_only` / `parvo_only` ablations, which hold
w = 0.5 and zero one term: the classifier is affine, so rescaling features moves
the argmax through the bias term. Empirically the difference is negligible —
w = 1.00 gives 0.879 against the ablation's 0.878, and w = 0.00 gives 0.419
against 0.418 — but they are different quantities and we do not equate them.
Second, the decided-image count varies across `w` (0.24–0.45 of the 1,200), so
the extreme-`w` cells carry the denominator caveat of §2.4; the monotone trend
holds across the well-decided middle of the range regardless.

| | single stream | two-stream `dkl_bcs` | Δ |
|---|---|---|---|
| cue-conflict shape bias, best saved ckpt | 0.567 | **0.655** | **−0.088** |
| cue-conflict shape bias, final ckpt | 0.557 | **0.675** | **−0.118** |
| top-1 accuracy, best saved ckpt | **0.599** | 0.556 | +0.043 |
| trainable parameters | 11,230,832 | 22,397,936 | 0.501× |

Seed-level separation is **complete under both checkpoint policies** (single
0.541/0.590/0.570 against two-stream 0.594/0.718/0.652 at the best saved
checkpoint), and the decision rates match to within 0.003, so this is not the
denominator artefact of §2.4. **The split buys shape bias and pays for it in
accuracy.** The ordering that matters for this section is

*Two limits.* This tests the split for the **diet-on-luminance arm only** and
must not be generalised to `swap` or `parvo_highsf`. And the capacity confound
is retired for the accuracy comparison (the smaller model wins) but **not** for
the shape-bias comparison: a parameter-matched single stream (`resnet34`,
~21.3M) might close the 0.088 gap. That arm was pre-registered and has not been
run; it is 3 seeds × ~26 h of GPU and its absence is a decision, not an
oversight (PLAN §72.6).

## E4 — §2.4, the front-end 2×2

Moved from §2.4: the seed-stability note, the denominator caveat on cue-conflict rates, the capacity control on `parvo_highsf`, and the accuracy-matched control in full including its pre-registered fatal outcome and its n=1 status.

*A note on which instruments are seed-stable.* The conjunction cell's probe bias
varies by **0.014** across seeds — the tightest number in this paper — against
**0.120** for §2.5's magno/parvo dimensionality ratio on the same three
checkpoints, both scored at the best saved checkpoint. §2.5 claims a side rather
than a level because its within-arm spread exceeds its between-arm gap, not
because its instrument is unstable (Supplementary Note S1).

*Denominator caveat.* Cue-conflict shape bias is computed only over images a
stream decides as shape *or* texture, and that count varies ~4× across the four
cells (165 / 270 / 583 / 488 of 1,200). A stream deciding ~13% of the set
returns something near a coin flip by construction, so the three
non-conjunction behavioural cells are read as **"no strong preference"** rather
than as precise values; the interaction is carried by the conjunction cell being
unambiguous, not by fine distinctions among the other three. The probe does not
have this problem — it classifies every image — and gives the same pattern,
which is why both modalities are reported (§51.3).

*Capacity control.* `parvo_highsf` gives its second stream an extra input
channel. The confound that would matter is if the extra channel *weakened* that
stream and faked the null. It does not: 16-way accuracy rises to 0.486 from
~0.379 in the standard diet models, a ~28% relative gain (§25.2). The extra
channel helped, and the stream still did not become shape-biased.

*Accuracy control.* The diet-trained models are less accurate than the no-diet
controls (§2.6.1, §2.7), so the conjunction effect could in principle be an
accuracy effect: any network held near 0.55 might look shape-biased. We tested
this directly rather than arguing about it, and pre-registered the fatal outcome
first — *if a no-diet model at the diet's accuracy shows a shape-biased magno
stream, §2.4 is an accuracy artefact and this paper's central claim is wrong*
(PLAN §69.3–§69.4).

No saved no-diet checkpoint is low enough to match — the earliest is already
0.658 — so we retrained `dkl_base` with per-epoch checkpointing and selected, by
rule, the epoch nearest the diet arm's 0.556. That is epoch 10 at **0.548**,
matched to within 0.008, which is tighter than the diet arm's own seed range
(0.012).

**Matching accuracy closes 7.6% of the probe gap and 7.0% of the behavioural
gap** — two instruments built on different machinery agreeing to within 0.6
percentage points. The matched model's probe bias remains *negative*: it is still
on the texture side of the sign flip, with every no-diet seed. Its behavioural
value sits 0.021 above the no-diet seed maximum, a shift smaller than that arm's
own seed spread (0.036).

Lowering a no-diet network's accuracy to the diet's does move it slightly toward
shape, and we report that residual rather than round it to zero — but it moves
about a fourteenth of the way. **The conjunction effect is not an accuracy
effect.** This control currently rests on **one** matched training run; two
further seeds are training, and until they land it should be read as n=1.

## E5 — §2.5, side of parity

Moved from §2.5: the front-end/trained ratio table and its estimator note, the three-seed replication table, the checkpoint-policy and checkpoint-invariance analyses, the within-training trajectory, the static-versus-progressive-diet check, and the provenance of the two decisive untrained references.

| arm | diet applied to | model | front end | after training |
|---|---|---|---|---|
| diet | luminance | `dkl_bcs_seed0` | 1.402 ±0.095 | **1.703** ±0.059 |
| diet | luminance | `dkl_bcs_parvo_highsf_seed0` | **0.586** ±0.014 | **1.640** ±0.032 |
| diet | **red−green** | `dkl_bcs_swap_seed0` | **1.909** ±0.133 | **2.327** ±0.085 |
| no diet | — | `dkl_base_seed0` | 1.634 ±0.126 | 0.947 ±0.038 |
| no diet | — | `dkl_base_seed1` | 1.634 ±0.126 | 0.988 ±0.043 |

*`±` here and in Extended Data §E6.1 is the sample standard deviation across the eight
subjects (ddof = 1), unified 2026-09-06. Seventeen cells across the two tables
previously mixed two estimators; fifteen have been restated and two were already
ddof = 1. **No mean, ordering, separation or claim changes** — every dispersion
that moved moved by the same √(8/7) = 1.069 — so a reader comparing against an
earlier version will see seventeen numbers change and no result change.*

**Seed replication (n=8 subjects x 3 seeds x 4 conditions).** Array 26413641
extracted seeds 1-2 for all four conditions. Every seed is scored at its **best
saved checkpoint** (PLAN §71.8, §71.10), so no arm is handicapped by the
late-training degradation of §2.6.1; for the six seeds whose best checkpoint is the
final one this is the same weight file, and those six reproduce to four decimals:

| arm | front end | seed 0 | seed 1 | seed 2 | seed range |
|---|---|---|---|---|---|
| diet / luminance | 1.402 | 1.703 | 1.722 | 1.602 | 0.120 |
| diet / parvo_highsf | **0.586** | 1.640 | 1.497 | 1.664 | 0.166 |
| diet / red−green | 1.909 | 2.327 | 1.541 | 1.995 | 0.787 |
| no diet | 1.634 | 0.947 | 0.988 | 0.914 | 0.074 |

We nonetheless report no level, no set point, and no compression factor: the
within-arm seed range reaches 0.79, which exceeds the 0.075 gap between the two
diet-on-luminance arm means, so any statement of the form "training converges to
X" remains unsupported at n=3. What replicates is the side. *(This retires two
earlier readings of our own data — a "convergence to within 2%" between the two
diet-on-luminance arms, whose 0.037 gap is still smaller than that arm's 0.120
seed noise, and a channel-specific set point, which n=3 seeds cannot separate:
PLAN §58.3.)*

**The seed ranges above are best-checkpoint values.** Scored at final
checkpoints they would read 0.904 and 2.388, both inflated by the late-training
degradation of §2.6.1; on validation-selected weights they fall to 0.120 and
0.787. The diet arm is therefore **not** markedly less reproducible than the
control on either instrument (Supplementary Note S1).

**When during training the dissociation appears.** One seed per arm has a
cached checkpoint trajectory, which lets us ask when the two arms separate. The
diet arm is above parity in 8/8 subjects at *every* checkpoint from epoch 25
onward (ratios 1.60–1.96). The no-diet arm is not: it sits **above** parity
through epoch 100 (1.16, 1.24, 1.51, 1.51) and crosses below only in the final
quarter of training (1.02 at epoch 125, 0.947 at epoch 150), while its accuracy
stays flat at 0.658–0.699 throughout. At epochs 75–100 the two arms are on the
same side of parity and there is no dissociation.

We report this rather than let it be discovered. It does not affect the claim as
stated — the section asks which side training *ends* on, and the ordering at the
end of training is what every table here reports — but it means the effect is
carried by a late change in the control arm that is not driven by accuracy, and
the honest form of the result is that **the diet stabilises which side of parity
a network sits on, while no-diet training drifts and ends below.** With one
trajectory per arm we do not promote that to a claim (PLAN §71.9).

Checkpoint choice does not drive any of this. Varying the checkpoint *within* a
seed across a wider accuracy span than separates the seeds leaves the ratio
essentially unmoved (r = −0.17, p = 0.75 for the diet arm; r = −0.18, p = 0.77
for no-diet), so the seed spread above is seed-to-seed variation rather than
checkpoint degradation.

*Does anything develop?* Our diet is a **static** front end matched to 48-month
acuity, not a trajectory — a deliberate choice that isolates the mechanism from
the schedule. `thesis_dvd_trans`, trained on an actual progressive age transition
in a different codebase, lands at **1.158, 8/8 subjects above parity**: the same
side as every diet arm, at a lower level. The schedule does not change the side,
which is the claim. Reported as a convergent check across implementations, not a
matched arm (§45.6).

*The references are exact for the two decisive arms.* `untrained_parvo_highsf`
and `untrained_swap` each carry provenance identical to their trained model on
every recorded key except the checkpoint, so those are matched front-end
pairings rather than approximate ones (§49.2, §50.1).

## E6.1 — the third dissociation: effective dimensionality is set by carrier bandwidth

**Moved out of the main text in full on 2026-09-07**, where it was §2.6.1. The
result is unchanged and still counts as one of the paper's three dissociations;
§2.6 now states it in a single paragraph and cites this section for the
evidence. Everything below is the original text.

*Source: PAPER_PLAN §47.*

Running the same four front ends through a participation-ratio analysis
(effective dimensionality of each stream's features on 1,000 NSD images per
subject, n=8) gives a **monotone ordering**, pre-registered before measurement
and confirmed on both references — a learning-free Gabor bank and an untrained
network — in 8/8 subjects, every adjacent pair at p=.0078.

| front end | stream 2 receives | Gabor bank | untrained net |
|---|---|---|---|
| `parvo_highsf` | **lum+rg**, no transform | **0.56** ±0.020 | **0.59** ±0.014 |
| `blur_cs` (diet) | rg, no transform | 1.38 ±0.091 | 1.40 ±0.095 |
| `noblur` (no diet) | rg, no transform | 2.23 ±0.143 | 1.63 ±0.126 |
| `swap` | rg + **transform** | **3.37** ±0.174 | **1.91** ±0.133 |

*The `±` column is printed at three decimals while the means stay at two, and
the mismatch is deliberate.* Three decimals are necessary because two flatten the
`parvo_highsf` row, whose dispersions are 0.0203 and 0.0137 — a 48% difference
that displays as 0.02 against 0.01. The means are deliberately **not** raised:
the `swap`/Gabor mean is 3.3665, exactly on the three-decimal boundary, and would
render 3.366 or 3.367 depending on the rounding convention, so the extra digit
would encode a choice rather than a measurement. A reader comparing against an
earlier version will see this column gain a digit and the two `blur_cs` cells
resolve from 0.09 into 0.091 and 0.095; that is added precision, not a revised
result. Nothing in the claim depends on the dispersions: the ordering is carried
by the means, separated by factors of 2–6 and agreeing on both references at
p=.0078.

(magno/parvo ratio; ordering pre-registered before measurement, confirmed on both
references, 8/8 subjects, every adjacent pair p=.0078.)

Three things follow.

**The asymmetry is generic and reversible.** It reproduces with **no network at
all** — the Gabor column is a learning-free oriented-energy bank — so it is not a
property of the architecture. And `parvo_highsf` falls *below 1.0* in 8/8
subjects: give the second stream the luminance carrier and **it becomes the more
dimensional stream**. There is nothing privileged about "magno"; dimensionality
tracks the bandwidth of the carrier a stream is handed.

**Only the stream whose input changed moves.** Relative to the diet baseline,
`parvo_highsf` doubles parvo while leaving magno at **×1.00**, and `noblur`
raises magno while leaving parvo at **×1.00**. The two ×1.00 entries are exact
because those cells share a front-end channel with the baseline — `parvo_highsf`
does not touch the luminance input and `noblur` does not touch the chromatic
one — so they are identity checks confirming that the untouched stream did not
move, not predictions that could have failed. The carrier account's two real
predictions are which stream moves and in which direction, and it gets both
right.

**Dimensionality is not a proxy for shape bias.** §2.4's probe factorises as an
interaction; this measure factorises additively, over *identical* inputs. The
decisive case is `parvo_highsf`, where the second stream becomes the **more
dimensional** one (0.56) while remaining **texture-biased** (probe −0.266). High
effective dimensionality does not confer shape bias, and we do not present these
as two views of one construct.

## E6.2 — §2.6.1, out-of-distribution robustness

Moved from §2.6.1: the seven-corruption table, the checkpoint-policy analysis, the model-selection and accuracy-cost accounting including the superseded 37% and 16.5% figures, and the full statement of the phase-scrambling equivalence result.

Relative accuracy (accuracy under corruption ÷ clean accuracy), area under the
degradation curve:

Scored on each seed's **best saved checkpoint** (PLAN §71.6), so the diet arm is
not handicapped by the late-training degradation described below:

| corruption | diet | no diet | swap | `parvo_highsf` | diet vs no-diet |
|---|---|---|---|---|---|
| uniform noise | 0.634 | 0.729 | 0.749 | 0.767 | −0.095, complete sep. |
| high-pass | **0.678** | 0.827 | 0.831 | 0.783 | −0.149, complete sep. |
| rotation | 0.574 | 0.615 | 0.605 | 0.611 | −0.041, complete sep. |
| false colour | 0.854 | 0.938 | 0.966 | 0.874 | −0.084, complete sep. |
| phase-scrambling | 0.383 | 0.373 | 0.371 | 0.382 | +0.009, **overlap** |
| *low-pass (trained-in)* | *0.444* | *0.356* | *0.359* | *0.385* | *+0.089* |
| *contrast (trained-in)* | *0.481* | *0.549* | *0.543* | *0.511* | *−0.068* |

**The diet-trained model is less robust on four of the five corruptions it was
not trained on**, with complete seed separation on all four and the same
ordering in both statistics (§8). On phase-scrambling the two arms are
indistinguishable (relative overlaps at the 3v3 floor, p = 0.70), so we claim
nothing there. The two conditions without a shape bias (`swap`,
`parvo_highsf`) pattern with the no-diet model on 8 of 10 comparisons.
Equivalence: the difference is **within ±0.033** (Δ = +0.0092, 90% CI
[−0.0145, +0.0330], TOST p = .0287), below the smallest OOD effect this paper
claims (rotation, 0.041). This is the weakest of the paper's four equivalence
results and is labelled as such: with 3 seeds per arm it rests on a Welch
interval at **2.3 degrees of freedom**, and PLAN §77.4 declared in advance that
no *nonparametric* equivalence result is obtainable at 3v3 at any bound.

**The progressive-transition arm.** `thesis_dvd_trans` (thesis codebase, one
seed, one saved checkpoint at epoch 147) carries the actual developmental age
transition rather than our fixed 48-month front end. Relative-accuracy AUC,
against the static arms above:

| corruption | `dvd_trans` | static diet (3-seed range) | no diet | inside diet range |
|---|---|---|---|---|
| uniform noise | 0.640 | 0.634 [0.619, 0.647] | 0.729 | yes |
| high-pass | 0.680 | 0.678 [0.647, 0.696] | 0.827 | yes |
| rotation | 0.572 | 0.574 [0.567, 0.582] | 0.615 | yes |
| false colour | 0.839 | 0.854 [0.845, 0.860] | 0.938 | no — *below* diet |
| phase-scrambling | 0.362 | 0.382 [0.365, 0.391] | 0.373 | no — *below* diet |
| *low-pass (trained-in)* | *0.436* | *0.444 [0.434, 0.464]* | *0.356* | *yes* |
| *contrast (trained-in)* | *0.489* | *0.481 [0.460, 0.493]* | *0.549* | *yes* |

Inside the static diet arm's seed range on five of seven; closer to the diet
arm than to the no-diet arm on six of seven; on the two exceptions it is below
the diet arm rather than toward the no-diet control. Clean accuracy 0.5682,
against 0.5530 (diet) and 0.6951 (no diet). At one seed no separation statistic
is available, so the main text claims containment in the observed diet range
and not equivalence. The single checkpoint is a late one and the diet arm is
the arm that degrades late, so the agreement is conservative.

*Checkpoint policy.* Four of five corruptions separate under **both** the final-
and the best-checkpoint policy; only which one is excluded changes. The
pre-registered crossover prediction (H2, that the diet should degrade *more*
under phase-scrambling) was supported only on degraded weights and is not
reported as supported (PLAN §54.2; Supplementary Note S1).

**Model selection, and the accuracy cost.** Training (§4.2) has no
validation-based model selection, and the diet arm — uniquely — degrades after
its per-epoch peak, losing −0.060, −0.246 and −0.136 from its own maximum
against −0.008, −0.013 and −0.003 for the control. Weights were saved every 25
epochs, so every result here uses the **best saved checkpoint** per seed (diet:
epochs 125, 75, 50; no-diet: 150). On that basis the diet arm reaches **0.556**
(range 0.012) against the control's **0.701** (range 0.007) — an accuracy cost
of **20.7%** relative.

Two things follow. The diet is **not** markedly less seed-stable than the
control (0.012 against 0.007); the apparent instability was an artefact of
scoring the final checkpoint. And **the diet destabilises late training**, which
is a separate finding. Crucially the §2.4 effect is *not* produced by
degradation: re-scored on best checkpoints, cue-conflict shape bias is **0.903**
(range 0.028) against 0.878, and the probe cell **+0.806** (range 0.013) against
+0.819, with the magno−parvo seed range tightening from 0.081 to 0.034. If
anything, degradation dilutes the effect. Earlier versions of this manuscript
gave the accuracy cost as 37% and then 16.5%; both are superseded and the
arithmetic of each is in Supplementary Note S1.

This is the paper's second dissociation of properties the field treats as
coupled: §E6.1 separates shape bias from representational dimensionality over
identical inputs, and the present section separates it from robustness.

## E6.3 — §2.6.2, error consistency

Moved from §2.6.2: the seven-model kappa table, the full pre-registration result with its self-correction, the equivalence bound in full, the pipeline validation run before the result was read, the marginal-compression caveat, and the consequences drawn for this paper's own wording.

| | shape bias | kappa vs human | % of ceiling |
|---|---|---|---|
| diet, whole model (3 seeds) | 0.646–0.692 | 0.023–0.032 | 10–14% |
| no diet, whole model | 0.394–0.433 | 0.031–0.037 | 13–16% |
| diet, magno stream | 0.870–0.892 | −0.015–0.036 | −6–15% |
| no diet, magno stream | 0.398–0.434 | 0.036–0.041 | 15–18% |
| *AlexNet* (published) | *0.429* | *0.032* | *14%* |
| *GoogLeNet* (published) | *0.312* | *0.025* | *11%* |
| *VGG* (published) | *0.172* | *0.017* | *7%* |

**We pre-registered the prediction that the diet-trained model would be more
human-consistent than its control. It is not**, on either readout. The sharpest
true statement the table supports is stronger than a comparison against any one
baseline: **our most shape-biased model has the lowest error consistency of all
seven**. The diet-trained magno stream carries shape bias 0.878 — 5× VGG-16's
0.172 — and returns a mean kappa of **0.0089**, 3.8% of the human–human ceiling,
against VGG-16's 0.0170 (7.3%). Moving a model 5× up the field's most-cited
human-alignment metric moved it *down* on trial-level agreement with humans.
*(An earlier version of this paragraph said "no model of ours exceeds AlexNet".
That is false by the table three lines above it — the no-diet magno stream
reaches 0.036–0.041 against AlexNet's 0.032 — and is corrected here;
PLAN §78.4.)*

*The null is bounded, not merely unrejected.* Equivalence intervals were fixed
before any interval was computed (PLAN §77.5, §78). The diet-vs-no-diet
difference in kappa is **within ±0.013**, i.e. ±5.4% of the human–human ceiling
(Δ = −0.0060, 90% CI [−0.0127, +0.0006], TOST p = .0030). Across the observed
0.706 span of shape bias the data **exclude a kappa change larger than 0.037 —
15.8% of the ceiling**. The claim is therefore not "we failed to detect a
relationship": spanning 5× in shape bias moves trial-level agreement with humans
by at most a sixth of the ceiling.

*Validation, run before the result was read.* The 10 observers reproduce the
published shape bias of 0.958 exactly (range 0.919–0.987); kappa returns +1.000
for identical raters, −0.043 for independent raters at a matched rate, and
−0.036 for a simulated model matching a human's rate but choosing independently.
The pipeline was checked against published values before it was trusted.

*The caveat that does not rescue it.* Kappa compresses when marginals are skewed,
and humans are at 0.958 — which is why the human–human ceiling is 0.234 and not
something near 1. This is handled by reporting every model as a percentage of
that same ceiling. It does not explain why the diet model scores *below* its own
no-diet control.

*Consequence for this paper.* The word "human-like" has been removed from the
title. §2.4's comparison against human observers is a comparison of aggregate
**rates**, and additionally compares a single stream to a whole observer; both
qualifications are now stated wherever that comparison appears.

