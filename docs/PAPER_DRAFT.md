# Shape-biased behaviour does not require shape-biased representations

**Status: SECOND DRAFT, 2026-09-06.** Reframed from the first draft
(2026-09-02), which led with the carrier x transform conjunction. Structure and
argument are settled; prose is second-pass. Every number is tagged with its
`PAPER_PLAN.md` section so it can be traced to the run that produced it.
Nothing here is written from memory.

**Positioning (revised 2026-09-06).** The paper now leads with the question the
originating thesis posed and could not answer: why human behaviour is
shape-biased when ventral-stream representations are texture-like. The
mechanistic results that previously led (the front-end 2x2, the side-of-parity
result) are retained in full as the answer to *how the shape-biased component
arises and survives training*, not as the paper's claim. Claims the earlier
framing invited and this one does not make are listed under *Claims
deliberately not made*.

---

## Abstract

Humans confronted with an object whose shape and texture disagree name the
shape; standard convolutional networks name the texture. Set against this is a
second body of evidence pointing the other way: regions of the human ventral
stream, the pathway most identified with object recognition, carry
representations better described as texture-like than shape-like. How a visual
system can behave shape-biased while its ventral representations look
texture-biased has remained unresolved.

We show the two facts are compatible, and give a mechanism. A two-stream
network receives approximations of the magnocellular and parvocellular signals
reaching cortex; we evaluate it behaviourally against published human
observers and against eight subjects of the Natural Scenes Dataset.

**The two streams end training with opposite biases** (linear-probe
shape-texture bias **+0.819** and **-0.518**, 3 seeds), **and they get them by
different routes**: the luminance stream's shape bias is supplied almost
entirely by its input statistics -- an untrained network on the same front end
already reaches **+0.796** of the trained **+0.810** -- while the chromatic
stream's texture bias is built by learning (**-0.087 -> -0.482**). The two
trunks are bit-identical at initialisation, so every divergence between them is
attributable to the difference in their inputs by construction.

**The texture-biased stream is the one that explains the ventral end of the
hierarchy.** Fitting each stream separately to cortex, the stream-preference
profile declines from V1v to hV4 by **-0.862** with the developmental front end
against **-0.101** without (p=.0078, 8/8 subjects, both statistics, three
seeds), monotonically with hierarchical rank (rho = -0.964), and it is not a
global stream preference: the two models sit on opposite sides of the global
stream-bias axis while the gradient belongs to the one pointing the wrong way.

**Behaviour reads out from both, and the mixture is shape-biased.** The
integrated readout reaches **0.675** while the shape-biased stream alone reaches
**0.878** -- integration *costs* 0.203. A readout inheriting from the
shape-biased stream would sit at its value; one reading the texture-biased
stream would sit below parity. It does neither.

Texture-selective ventral representations are therefore not in conflict with
shape-biased behaviour. Three properties the field treats as travelling with
shape bias -- representational dimensionality, out-of-distribution robustness,
and trial-by-trial agreement with ten human observers -- do not travel with it,
which is what follows if shape bias is a property of a readout rather than of a
representation.

> TODO: tighten to ~150 words for submission. Currently ~330, down from 636.

---

## 1. Introduction

> DRAFT -- argument fixed, prose to be tightened.

Humans presented with an image whose shape and texture point to different
categories name the shape. Standard convolutional networks name the texture.
That dissociation, measured on a cue-conflict set that ten human observers also
viewed, is the most conspicuous behavioural gap between human and machine
object recognition, and closing it has become a standard target for
biologically motivated models.

A second body of evidence points the other way. Regions of the human ventral
stream -- the pathway most identified with object recognition -- carry
representations that are well described as texture-like rather than shape-like.
Both findings are robust and they appear to be in tension. If behaviour is
shape-biased and the ventral stream is texture-biased, then either the
behavioural and representational measures disagree about what shape bias is, or
**behaviour is not a readout of one representation.**

This paper takes the second horn and shows it suffices. The claim is not that
one measure is wrong. It is that a system containing a shape-biased and a
texture-biased component, whose behaviour reads out from both, will look
shape-biased behaviourally and texture-biased wherever the texture-biased
component happens to dominate -- and that in a model of the subcortical split,
the component that dominates the ventral end is the texture-biased one.

Three facts about the visual system motivate the model we use.

**(i) The primate visual system does not receive one image.** It receives a
luminance-dominated, temporally fast, spatially coarse magnocellular signal and
a chromatically opponent, spatially finer parvocellular one, separated before
cortex. Models of ventral-stream computation almost universally discard this,
taking RGB pixels as input.

**(ii) That input is not stationary over development.** Infant acuity and
contrast sensitivity are poor and improve over the first years; the visual
system learns first from a low-passed world. A *developmental visual diet* (DVD)
implements this as a transform on the achromatic channel.

**(iii) Shape bias is measured behaviourally and interpreted
representationally.** The field's standard instrument is a behavioural rate on
cue-conflict images, and a model that matches the human rate is routinely
described as having human-like representations. Whether a behavioural rate
licenses a representational description is exactly what is at stake here.

**Contributions.**

1. **Two streams with opposite biases, arising by different routes** -- the
   shape-biased component supplied by the input, the texture-biased component
   built by learning -- in an architecture whose streams are bit-identical at
   initialisation, so that every divergence is attributable to input by
   construction (§2.1).
2. **The texture-biased stream is the one that carries the ventral end** of the
   hierarchy, on per-stream voxelwise encoding models over eight NSD subjects,
   surviving six front-end controls and, unlike an RSA-style analysis of the
   same question, not reducible to a global stream preference (§2.2).
3. **Behaviour reads out from both and integration mixes rather than
   inherits**, so shape-biased behaviour coexists with a texture-biased
   subcomponent (§2.3). Together, 1-3 dissolve the paradox in a model.
4. **A four-cell causal decomposition of what produces the shape-biased
   component** -- achromatic carrier x developmental transform, an interaction
   rather than a property of either factor (§2.4) -- and **what preserves it
   through training**, where the diet fixes which side of parity training ends
   on (§2.5).
5. **Three dissociations**: over identical inputs, shape bias separates from
   effective dimensionality, from out-of-distribution robustness, and from
   human decision structure (§2.6). Properties the field treats as a package do
   not travel together, which is what a readout account predicts.
6. An honest accounting of what the account costs (§2.7).

---

## 2. Results

### 2.1 The two streams acquire opposite biases, by different routes


*Source: PAPER_PLAN §28.1, §28.5/§35, §12.*

The model splits its input once, before any convolution: stream 1 receives the
luminance channel carrying the developmental transform, stream 2 the
untransformed L-M chromatic channel (Methods §4.1). Read out separately with a
linear probe, the two streams end training on opposite sides of the
shape/texture boundary, and the same holds behaviourally:

| readout | magno stream | parvo stream | integrated |
|---|---|---|---|
| linear probe, shape-texture bias (0 = sign flip) | **+0.819** | **-0.518** | -- |
| cue-conflict shape bias (0.5 = parity) | **0.878** | **0.418** | **0.675** |

Means of 3 training seeds. The probe classifies every image; the cue-conflict
denominators are reported per cell in §2.4.

**The decomposition is asymmetric in how it arises.** Scoring the same probe on
an untrained network with the same front end separates the two routes:

| stream | untrained | trained | learning contributes |
|---|---|---|---|
| magno (luminance + transform) | **+0.796** | **+0.810** | **+0.014** |
| parvo (L-M) | -0.087 | **-0.482** | **-0.395** |

The magno stream's shape bias is essentially complete before any weight is
updated -- the low-pass and contrast-sensitivity transform does it -- while the
parvo stream begins barely biased and acquires its texture preference over
training. **The shape-biased component is supplied by the input; the
texture-biased component is built by learning.** We state this plainly rather
than let it be inferred: nothing in this paper claims that learning creates the
shape-biased component, and §2.4's 2x2 is a decomposition of the *input*, not
of what training adds.

*This comparison is single-model.* The untrained references carry no training
seed, and the trained column here is seed 0 (the 3-seed trained values are the
table above). The route asymmetry is large relative to the trained arm's seed
range of 0.014, but it is reported at one seed per cell and should be read that
way.

**Ordinary training destroys the bias the input supplies; the transform is what
prevents that.** Removing the transform, the untrained magno bias falls from
+0.796 to +0.417 -- and training then drives it to **-0.219**. The unblurred
luminance stream does not merely fail to become shape-biased; it becomes
*texture*-biased. §2.5 finds the same pattern on an independent instrument,
where diet-trained networks end magno-dominant from front ends spanning 3.26x
while every no-diet network collapses to parity.

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

### 2.2 The texture-biased stream explains the ventral end of the hierarchy


*Source: PAPER_PLAN §43.1, §17, §48, §29, §66.*

If the two streams carry opposite biases, the paradox turns on which of them the
ventral hierarchy is explained by. We fit voxelwise banded-ridge encoding models
per stream on NSD (8 subjects, b3 betas, `nsdgeneral`, test set = the shared 515
images), and use them as measurement instruments twice: first to confirm that
the two streams' cortical mappings are oppositely organised, then to locate
where along the hierarchy each does its explanatory work.


**(a) In-silico cue conflict.** The encoding models are fitted on natural scenes
only; no participant ever viewed a cue-conflict image. Passing the 1,200
cue-conflict images through the *fitted* models therefore yields a genuine
model-based prediction — a per-voxel predicted shape-bias index — and the
magno−parvo difference in that index is **+1.534 with the diet and +0.110
without** (n=8, p=.0078, 8/8 subjects).

The raw statistic separates the arms even more sharply than the normalised one,
by sign rather than by degree: **+0.616 with the diet against −0.033 without**
(and +0.658 / −0.039, +0.632 / −0.053 in seeds 1 and 2, p=.0078 and 8/8
throughout). Without the diet the predicted cortical response to a cue-conflict
image is, if anything, marginally *more* texture- than shape-organised.

The effect is carried entirely by the magno stream (+0.791 vs **−0.638** —
without the diet the magno stream is *texture*-dominated), with the parvo streams
indistinguishable (−0.743 vs −0.748). This confirms the representational result
of §2.4 on independent machinery, through the brain data, using no probe and no
readout.



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
checkpoints. (An earlier version of this sentence gave 28-fold, against a §2.5
seed range of 0.904 that was itself a final-checkpoint artefact; see §2.5 and
PLAN §71.10. The cortical instrument is still the tighter of the two, but not by
the margin previously claimed, and we no longer attribute §2.5's spread to the
participation-ratio measure — it belonged to the checkpoint.) The
raw statistic agrees on every seed, so the two-statistics rule is satisfied three
times over.


**(b) The ventral gradient of stream division of labour.** The same fitted
models answer a second, orthogonal question: not *what* each stream's cortical
mapping is sensitive to, but *where along the ventral hierarchy* each stream does
its explanatory work. Partitioning every voxel's explained variance into the part
only the magno stream accounts for (Uₘ) and the part only the parvo stream does
(Uₚ), and summarising each retinotopic ROI by that contrast, gives a
stream-preference profile from V1v to hV4.

The profile **declines steeply with the diet and is nearly flat without it** —
early visual cortex is relatively more magno-explained and hV4 relatively more
parvo-explained, the direction a luminance/low-SF versus chromatic split
predicts. On all three seeds and both statistics (§4.7):

| statistic | seed | diet | no diet | p | subjects |
|---|---|---|---|---|---|
| normalised | 0 | **−0.862** | −0.101 | .0078 | 8/8 |
| normalised | 1 | **−0.802** | −0.132 | .0078 | 8/8 |
| normalised | 2 | **−0.781** | −0.073 | .0078 | 8/8 |
| raw | 0 | **−0.176** | −0.034 | .0078 | 8/8 |
| raw | 1 | **−0.151** | −0.041 | .0078 | 8/8 |
| raw | 2 | **−0.160** | −0.030 | .0078 | 8/8 |

**Both statistics reach the exact Wilcoxon floor with complete seed separation in
every seed** — the two-statistics rule of §8 satisfied six times over.
Monotonicity tracks it (Spearman ρ against hierarchical rank −0.964 with the diet
against −0.502 without) and the effect sizes differ by an order of magnitude
(dz −11.70 vs −1.06).

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
- **The chromatic side is specific to L−M, not to chromaticity as such.** The
  ventral end of this gradient rests on the parvo stream's unique variance at
  hV4, so we asked whether *any* cone-opponent axis would deliver it.
  Substituting the koniocellular axis S−(L+M) for L−M in stream 2, holding the
  luminance stream bit-identical — the magno-only fit is unchanged to
  0.000e+00 on every voxel of every subject, so the contrast is exactly
  R²_joint(konio) − R²_joint(parvo) with no magno-side estimation error — makes
  the ventral prediction **worse**, and worse where colour lives: hV4
  **−0.029**, VO **−0.024**, the pre-declared colour set **−0.026**, each at
  p=.0078 with **0/8** subjects positive and both statistics agreeing in sign.
  The deficit is concentrated rather than global — colour set minus the other 17
  ROIs is **−0.014** (p=.0078, 0/8) — and that specificity contrast is
  **unchanged at −0.014 when the konio channel instead keeps its native 2.05×
  contrast advantage**, a manipulation that lifts konio's mean over those other
  ROIs from −0.012 to +0.008 and lets it win early visual cortex outright (V1d
  +0.077, 8/8). The colour-specific deficit therefore does not lie on the
  contrast axis, which is the first axis a sceptic would reach for. Both
  cone-opponent axes have now been tested against colour-biased cortex and
  neither predicts it — L−M's own failure there is reported under *Claims
  deliberately not made* — so what the parvo stream contributes at the ventral
  end of this gradient is specific to L−M without being a claim about colour
  selectivity.
  *Two limits.* Pooled VO reaches the floor in the contrast-matched cell only
  (p=.11 unmatched, where hV4 and the specificity contrast still hold at .0078),
  and these fits are **untrained and single-seed** (Methods): they test the
  *input* as a model of colour cortex through the architecture prior — not a
  trained koniocellular stream, and no third stream was added.

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

### 2.3 Behaviour reads out from both streams, and the mixture is shape-biased


*Source: PAPER_PLAN §18.2, §52.3, §20.3, §28.1.*

The two streams are oppositely biased (§2.1) and the texture-biased one carries
the ventral end (§2.2). What remains is what the behavioural readout does with
them.


**Behaviour, against human observers.** The integrated readout's
cue-conflict shape bias is **0.675 ±0.025** with the diet against **0.416
±0.020** without, across three seeds each. The thesis checkpoint reproduces at
0.697 against its reported 0.677, inside our bootstrap CI (§18.2).

§2.6.3 qualifies everything in this section: the agreement below is one of
aggregate *rates*, and does not survive at the trial level.

On the identical measure and stimuli, Geirhos et al.'s ten human observers give
**0.958 ±0.019** (range 0.919–0.987), and the CNNs from that release give
0.429 (AlexNet), 0.312 (GoogLeNet) and 0.172 (VGG-16) — values our pipeline
reproduces, which is how we check our arithmetic against theirs. The
diet-trained **magno stream reaches 0.878**, closing most of the distance from
the CNN regime to human observers; the no-diet model stays inside that regime.
**No match is claimed:** 0.878 is below the lowest individual human, and
integration costs 0.203 of shape bias (0.878 → 0.675), which is 72% of the
remaining gap. See §2.7 and §52.3.


**Integration mixes; it does not inherit.** The integrated readout's
cue-conflict bias is **0.675**, against **0.878** for the magno stream alone and
**0.418** for the parvo stream alone. Integration therefore *costs* 0.203 of
shape bias -- 72% of the remaining distance to the human mean -- and that cost
is the evidence. A readout that simply inherited from the shape-biased stream
would sit at 0.878; one reading out the texture-biased stream would sit below
parity, near 0.418. It does neither. The readout combines two components with
opposite biases and lands shape-biased.

This is the mechanism we propose for the paradox, and we state its scope
exactly: **shape-biased behaviour does not require shape-biased
representations**, because a behavioural rate is a property of a readout over
components, not a report of any one of them. It is an existence proof with a
cortically grounded demonstration -- in this model, the component that dominates
the ventral end is the texture-biased one, and behaviour is nonetheless
shape-biased. It is not a measurement of human cortex, and §2.6.3 bounds how far
the behavioural correspondence extends.

### 2.4 What produces the shape-biased component


*Source: PAPER_PLAN §25, §13, §18, §21.*

§2.1 showed the shape-biased component is supplied by the input rather than by
learning. This section asks *what about the input* supplies it. The natural
reading -- that shape bias belongs to the achromatic magnocellular channel --
is wrong.


We decomposed the front end into two orthogonal factors: **which carrier** a
stream receives (achromatic luminance vs chromatic L−M) and **whether the
developmental transform is applied** to it. This defines four cells. Two existed
already; we built the other two (`swap`, which applies the transform to the
chromatic channel, and `parvo_highsf`, which hands the second stream an
untransformed high-acuity achromatic carrier alongside its chromatic one).

Linear probe, stream shape−texture bias (zero is the sign flip). Means of
**3 training seeds**, with the seed range in brackets:

| | no transform | with transform |
|---|---|---|
| **chromatic carrier (L−M)** | −0.504 [0.035] | −0.145 [0.043] |
| **achromatic carrier (lum)** | **−0.271** [0.023] | **+0.819** [0.014] |

Cells are read out on stream 2 in three of four cases and on the diet's magno in
the fourth, so the input factors vary while the stream role is held fixed
(§25.1): chromatic/no-transform is the no-diet parvo, achromatic/no-transform is
`parvo_highsf`'s stream 2, chromatic/transform is `swap`'s parvo.

Behavioural cue-conflict shape bias on the same four cells replicates it
independently: 0.364 [0.080] / 0.495 [0.038] / 0.396 [0.012] / **0.878** [0.022].
(`parvo_highsf` seed 0 is excluded from the behavioural row only — it decided
139/150 images when the convergence guard ran, so its denominator is not
comparable; the probe classifies every image and keeps all three seeds.)

The probe separates the conjunction cell from all three others by **+0.936**
with no seed overlap, and the behaviour by **+0.353**. On the behavioural row
the chromatic-transform cell straddles 0.5 across seeds (0.478–0.516), which is
the denominator caveat below rather than a texture preference: that cell decides
only 165 of 1,200 images and returns something near a coin flip by construction.

*A note on which instruments are seed-stable.* The conjunction cell's probe bias
varies by **0.014** across seeds — the tightest number in this paper. The same
three checkpoints vary by **0.120** on the magno/parvo dimensionality ratio of
§2.5 when both are scored at the best saved checkpoint. §2.5 still claims a side
rather than a level, but because its within-arm spread exceeds its between-arm
gap (§2.5), not because its instrument is unstable — an earlier version of this
note gave that spread as 0.904 and drew the wrong conclusion from it
(PLAN §71.10).

**Both single factors move a stream substantially toward neutral and neither
crosses zero. Only the conjunction crosses, and it goes all the way.** This is an
interaction, measured on all four cells rather than inferred from two, and it is
the strongest causal statement in the paper about *what* produces the shape bias:
the developmental low-pass transform applied specifically to an achromatic
channel.

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
controls (§2.6.2, §2.7), so the conjunction effect could in principle be an
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

| | diet (0.556) | no diet (0.70) | **accuracy-matched no diet (0.548)** |
|---|---|---|---|
| probe, magno bias | **+0.806** | −0.226 | **−0.148** |
| cue-conflict, magno-only | **0.903** | 0.421 | **0.455** |

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


### 2.5 What keeps the shape-biased component through training


*Source: PAPER_PLAN §45, §49, §50.*

§2.1 showed that ordinary training destroys the shape bias the input supplies
(+0.417 untrained -> -0.219 trained) while training under the developmental
transform preserves it (+0.796 -> +0.810). That is a claim about what learning
converges to, and this section tests it on an independent instrument, where a
front end built to point the wrong way can falsify it.

The four front ends of §2.4 hand a network very different stream asymmetries
before any learning -- the untrained magno/parvo dimensionality ratio spans 0.59
to 1.91 across them (quantified in §2.6.1). What does training do with them?


The first-order answer, from the two conditions available previously, was that
the diet makes training *keep* the asymmetry the input supplies while no-diet
training destroys it. That description is arithmetically correct and cannot be
right, because in both of those conditions the input asymmetry pointed the same
way — so *keep* and *install* predicted the same outcome.

`parvo_highsf` separates them. Its front end is inverted (0.586, below 1.0 in 8/8
subjects). We pre-registered the prediction that a diet-trained network on that
front end would stay inverted. **It does not.**

| arm | diet applied to | model | front end | after training |
|---|---|---|---|---|
| diet | luminance | `dkl_bcs_seed0` | 1.402 ±0.089 | **1.703** ±0.059 |
| diet | luminance | `dkl_bcs_parvo_highsf_seed0` | **0.586** ±0.013 | **1.640** ±0.030 |
| diet | **red−green** | `dkl_bcs_swap_seed0` | **1.909** ±0.125 | **2.327** ±0.085 |
| no diet | — | `dkl_base_seed0` | 1.634 ±0.118 | 0.947 ±0.035 |
| no diet | — | `dkl_base_seed1` | 1.634 ±0.118 | 0.988 ±0.040 |

**Seed replication (n=8 subjects x 3 seeds x 4 conditions).** Array 26413641
extracted seeds 1-2 for all four conditions. Every seed is scored at its **best
saved checkpoint** (PLAN §71.8, §71.10), so no arm is handicapped by the
late-training degradation of §2.6.2; for the six seeds whose best checkpoint is the
final one this is the same weight file, and those six reproduce to four decimals:

| arm | front end | seed 0 | seed 1 | seed 2 | seed range |
|---|---|---|---|---|---|
| diet / luminance | 1.402 | 1.703 | 1.722 | 1.602 | 0.120 |
| diet / parvo_highsf | **0.586** | 1.640 | 1.497 | 1.664 | 0.166 |
| diet / red−green | 1.909 | 2.327 | 1.541 | 1.995 | 0.787 |
| no diet | 1.634 | 0.947 | 0.988 | 0.914 | 0.074 |

All nine diet seeds are above parity in **8/8 subjects**. All three no-diet seeds
sit at or below it. Seed-level separation between the diet arms and no-diet is
**complete** for all three arms (p=0.100, the exact two-sided 3x3 floor); the
narrowest gap, diet/luminance against no diet, is **+0.614**.

We nonetheless report no level, no set point, and no compression factor: the
within-arm seed range reaches 0.79, which exceeds the 0.075 gap between the two
diet-on-luminance arm means, so any statement of the form "training converges to
X" remains unsupported at n=3. What replicates is the side. *(This retires two
earlier readings of our own data — a "convergence to within 2%" between the two
diet-on-luminance arms, whose 0.037 gap is still smaller than that arm's 0.120
seed noise, and a channel-specific set point, which n=3 seeds cannot separate:
PLAN §58.3.)*

**A correction to the seed spread we previously reported.** An earlier version of
this table scored every seed at its final checkpoint and gave the diet/luminance
seed range as 0.904 and the `swap` range as 2.388. Both were inflated by the
late-training degradation documented in §2.6.2: on weights selected by validation
accuracy the two ranges fall to **0.120 and 0.787**, a 7.6× and 3.0× tightening,
while the two arms whose best checkpoint *is* the final one do not move at all.
The same correction appeared independently on accuracy (§2.6.2, seed range
0.180 → 0.012). The diet arm is therefore **not** markedly less reproducible than
the control on either instrument; it was being scored on degraded weights
(PLAN §71.10).

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

The inverted case flips completely: 8/8 subjects below parity before training,
**0/8 after**, ending at 1.640 — *above* a model handed 1.402. So the input's
asymmetry is not what training carries forward.

But the diet does not drive training to a single value either. Moving the
identical transform onto the chromatic channel (`swap`) lands at **2.327**,
differing from both diet-on-luminance arms at p=.0078, 8/8 (per-subject
2.09–2.38). That arm is the best-controlled of the four — provenance-exact
reference, and unlike `parvo_highsf` it is architecture-matched — so it cannot
be set aside.

What is invariant is the **side of parity**. All three diet arms end
magno-dominant (≥1.60) from front ends spanning **3.26×**; both no-diet arms end
at parity (0.947, 0.988) from the most magno-dominant front end in the set. All
**six** diet × no-diet contrasts on the trained ratio: p=.0078, 8/8 subjects.

> **The claim.** A developmental visual diet does not preserve the stream
> asymmetry its front end supplies, and does not converge on one universal
> value. What it fixes is **which side of parity training ends on** — every
> diet-trained model ends magno-dominant regardless of a 3.26× spread in what it
> was handed, and every no-diet model collapses to parity from a strongly
> magno-dominant front end. The **level** reached is set by *which channel the
> diet is applied to* (~1.62 on luminance, 2.23 on red−green), not by the input
> asymmetry and not by the schedule alone.

This is what rules out the filtering account. The diet is not supplying a better
input representation: the transform *removes* bandwidth rather than adding it
(§2.6.1), and the no-diet front end hands over more asymmetry, not less. It is a
constraint on the solution learning converges to.

*The references are exact for the two decisive arms.* `untrained_parvo_highsf`
and `untrained_swap` each carry provenance identical to their trained model on
every recorded key except the checkpoint, so those are matched front-end
pairings rather than approximate ones (§49.2, §50.1).


### 2.6 Three properties that do not follow from shape bias

If shape bias is a property of a readout over components rather than of a
representation, then quantities the field treats as travelling with it need not.
Over identical inputs, three do not.

#### 2.6.1 Effective dimensionality is set by carrier bandwidth

*Source: PAPER_PLAN §47.*


Running the same four front ends through a participation-ratio analysis
(effective dimensionality of each stream's features on 1,000 NSD images per
subject, n=8) gives a monotone ordering:

| front end | stream 2 receives | Gabor bank | untrained net |
|---|---|---|---|
| `parvo_highsf` | **lum+rg**, no transform | **0.56** ±0.02 | **0.59** ±0.01 |
| `blur_cs` (diet) | rg, no transform | 1.38 ±0.09 | 1.40 ±0.09 |
| `noblur` (no diet) | rg, no transform | 2.23 ±0.13 | 1.63 ±0.12 |
| `swap` | rg + **transform** | **3.37** ±0.16 | **1.91** ±0.12 |

(magno/parvo ratio; ordering pre-registered before measurement, confirmed on both
references, 8/8 subjects, every adjacent pair p=.0078.)

Two things follow.

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


#### 2.6.2 Shape bias does not confer out-of-distribution robustness

*Source: PAPER_PLAN §53 (pre-registered), §56. Reported in the main text, not
the supplement, because a reader who accepts §2.4 will otherwise assume the
robustness follows.*


The standard reading of Geirhos et al. is that shape bias buys out-of-
distribution robustness. We pre-registered that prediction and tested it:
corruptions applied to the image *before* the front end, scored in the models'
own 112-way label space, **3 seeds per condition** on all four cells of §2.4's
2×2.

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

The **only** corruption on which the diet is more robust is low-pass — which we
declared trained-in before running, because the diet *is* a low-pass blur. The
experiment finds robustness exactly where robustness was trained in and nowhere
else, which is the strongest available evidence that the four nulls are real
rather than a broken measurement. Note that the diet is *not* more robust to
contrast reduction even though its front end includes a contrast-sensitivity
cut, so "trained-in" does not automatically confer robustness either.

*Checkpoint policy.* An earlier version scored this table on final checkpoints.
Moving to best saved checkpoints **strengthens** the dissociation on the three
pre-registered H1 corruptions — uniform noise goes from overlapping to
completely separated — while phase-scrambling moves the other way, from
separated to overlapping. The count of separated corruptions is four of five
under both policies; only which one is excluded changes. The pre-registered
crossover prediction (H2, that the diet should degrade *more* under
phase-scrambling) was supported only on the degraded weights; PLAN §54.2 had
already refused to report it as supported on the grounds that its apparent
support was vacuous, and that refusal is now vindicated for a second,
independent reason.

**A correction to how these models were selected.** Every result in this paper
was computed on the final training checkpoint. The per-epoch validation logs
show that is the wrong checkpoint for the diet arm, and only for the diet arm:
the diet models reach a per-epoch maximum of 0.592 (seed range 0.008) and end at
**0.445** (seed range 0.180), losing −0.060, −0.246 and −0.136 from their own
maxima, while the no-diet models lose −0.008, −0.013 and −0.003. The training
procedure (§4.2) has no validation-based model selection.

Weights were written every 25 epochs, so those per-epoch maxima are not
recoverable; the relevant comparison is the **best saved checkpoint** per seed
(diet: epochs 125, 75, 50; no-diet: epoch 150 for all three). On that basis the
diet arm reaches **0.556** (range 0.012) against the control's **0.701** (range
0.007).

Two consequences. The accuracy cost is **20.7%** relative, not the 37% a
final-checkpoint comparison suggests — and not the 16.5% that the unsaved
per-epoch maxima would imply, which we report only to characterise the schedule,
never as the cost of a model anyone could evaluate. And the diet arm is **not**
markedly less stable across seeds: at its best saved checkpoint it agrees to
0.012 against the control's 0.007, against 0.180 at the final checkpoint. The
apparent instability is an artefact of taking the last checkpoint. What is real,
and is a separate finding, is that **the diet destabilises late training**: the
diet arm degrades after its peak, the control does not.

Re-running cue-conflict on each diet seed's best saved checkpoint gives a shape
bias of **0.903** (range 0.028) against 0.878 at the final checkpoint — slightly
*higher* on the non-degraded models, even though those models are 0.111 more
accurate. The §2.4 probe cell is likewise insensitive to the choice: magno
+0.806 (range 0.013) against the published +0.819 (range 0.014), with all three
seeds on the predicted side and the magno−parvo seed range *tightening* from
0.081 to 0.034. Degradation does not produce the effect reported in §2.4; if
anything it dilutes it (PLAN §67, §68, §71).

> **The claim.** Shape bias and OOD robustness are not intrinsically coupled.
> This does not contradict Geirhos et al., who showed that *Stylized-ImageNet
> training* produces models that are both shape-biased and robust. It shows the
> coupling is a property of that route, not of shape bias: a developmental route
> to comparable shape bias *in rate* (probe +0.819, behaviour 0.878 against a
> human 0.958 — though §2.6.3 shows the rate match does not extend to trial-level
> agreement)
> delivers **none** of the robustness. "Increase shape bias to get robustness"
> does not generalise across the manipulations that produce it.

This is the paper's second dissociation of properties the field treats as
coupled: §2.6.1 separates shape bias from representational dimensionality over
identical inputs, and the present section separates it from robustness.



#### 2.6.3 Matching the human shape-bias rate is not matching human decisions

*Source: PAPER_PLAN §57 (pre-registered), §59. Reported in the main text, not
the supplement, because it bounds this paper's own behavioural claim.*


A shape-bias *rate* near the human one is weak evidence for human-likeness: two
decision-makers with the same marginal rate can agree on no individual image
beyond what those rates force. Geirhos et al. (2020) measure the difference with
**error consistency** — Cohen's kappa on trial-by-trial agreement, which
subtracts exactly that forced agreement. We applied it to the 10 human observers
who viewed these identical stimuli.

Human–human ceiling: **kappa = 0.234** (45 observer pairs, sd 0.095).

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
human-consistent than its control. It is not.** It is numerically less so on the
stream readout, and no model of ours exceeds AlexNet. Our shape-biased stream
carries **twice AlexNet's shape bias and no more agreement with human
decisions**.

Across all six models the shape-bias rate spans 0.17 to 0.89 — a 5× range —
while human agreement stays flat at 7–18% of ceiling. **Shape bias, the field's
most-cited human-alignment metric for vision models, does not track human
decision structure.** That is this paper's third dissociation and the only one
that concerns human behaviour directly.

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

### 2.7 What the account costs


*Source: PAPER_PLAN §43.3, §48.2, §20.2, §20.3.*


Stated here rather than in a limitations footnote, because a reviewer finds these
immediately and because they bound the claim.

**Prediction accuracy.** The no-diet model predicts BOLD **better**: median
R²/NC **0.464 without the diet against 0.405 with it**, p=.0078 on both
statistics, 8/8 subjects — a ~13% cost in explained variance. The diet's case
rests on stream differentiation and on the ventral gradient, **not** on
prediction accuracy, and we do not argue otherwise.

**Two of the four front-end controls are degenerate on the cortical instrument.**
In `swap` the parvo stream contributes unique variance Up = 0.006; in
`parvo_highsf` the magno stream contributes Um = 0.007. Each leaves one stream
contributing essentially nothing unique. Because the gradient statistic is a
ratio, a condition with one stream at ~0 has little dynamic range in which a
gradient could appear. So the correct statement of §2.2(b)'s control result is
**not** "the controls abolish the gradient" — it is that only the
diet-on-luminance configuration yields two streams that *both* carry substantial
unique cortical variance **and** a hierarchical gradient between them. The
controls do not relocate the diet within a fixed two-stream decomposition; they
destroy the decomposition. That is narrower than a clean control result and it is
what the data support.

**The human comparison is directional, and now quantified.** Human observer data
is in hand (§52), so the gap is measured rather than gestured at: the best stream
sits **0.080 below the human mean** and below the lowest individual human, and
the integrated readout sits 0.283 below. The paper therefore states that the diet
moves the network most of the way from the texture-biased CNN regime toward human
observers and that the no-diet model stays in that regime, **with the numbers
given**. It does not claim a match, does not use "human-like", and does not
report any stream as falling in the human range.

## 3. Discussion

### The paradox, and what dissolves it

Human behaviour on cue-conflict images is shape-biased; human ventral-stream
representations have been described as texture-like. Read as competing claims
about a single representation, these cannot both be right. Read as claims about
a *readout over components*, they are not in competition at all.

This paper builds a system in which both are true simultaneously and shows what
makes them compatible. The luminance stream is shape-biased (+0.819) and the
chromatic stream is texture-biased (-0.518). The texture-biased one is the one
whose unique variance carries the ventral end of the hierarchy: the
stream-preference profile falls from V1v to hV4 by -0.862 with the developmental
front end and by -0.101 without, monotonically with hierarchical rank, on three
seeds and both statistics, and not as a global preference. And the behavioural
readout, which sees both, reaches 0.675 -- shape-biased, but 0.203 *below* the
shape-biased stream alone. Nothing has to be wrong for a system to look
texture-biased in ventral cortex and shape-biased in behaviour. It only has to
have more than one component and a readout that mixes them.

We are precise about what this licenses. It is an existence proof plus a
cortically grounded demonstration: the paradox *need not* be a paradox, and here
is a mechanism that reproduces it in a model fitted to human cortex. It is not a
measurement of the human readout. A single-stream account in which texture-like
ventral representations feed shape-biased behaviour directly is not excluded by
these data.

### Shape bias is a readout property, not a representational one

The field's standard instrument is a behavioural rate on cue-conflict images,
and a model matching the human rate is routinely described as having human-like
representations. Our system shows that inference is invalid twice over. The same
network contains components at +0.819 and -0.518 and behaves at 0.675, so the
rate does not identify a representation. And §2.6.3 shows the rate does not
identify a *decision process* either: across six models spanning a 5x range of
shape bias, trial-by-trial agreement with ten human observers stays flat at
7-18% of the human-human ceiling, with our shape-biased stream no better than
AlexNet. We pre-registered the opposite prediction.

The methodological consequence is direct. Shape bias should be reported as what
it is -- a readout statistic -- alongside error consistency wherever human trial
data exist, and a shape-bias rate should not be used on its own to license a
claim about a model's representations.

### Against channel essentialism

The two-condition literature invites the reading that the magnocellular channel
*is* the shape pathway. §2.4 and §2.6.1 jointly rule that out. Shape bias
requires a carrier x transform conjunction: an achromatic carrier alone does not
produce it, the transform alone does not produce it, and only their combination
crosses the boundary -- by +0.936 with no seed overlap. Effective dimensionality,
meanwhile, needs only bandwidth, and reverses when the carrier moves.

Both properties are relational -- facts about what a stream receives, not about
what it is called. This matters beyond our architecture, because "magno-like"
and "parvo-like" are routinely used as if they named intrinsic computational
roles. In our model the streams are bit-identical at initialisation, so the
labels can name nothing but the input.

### What a developmental constraint actually does

The intuitive account of developmental low-pass input is informational: a
learner given coarse input first is protected from high-frequency texture, and
so learns shape. Our results do not support that account. The transform
*removes* bandwidth rather than adding structure, and the shape bias it produces
is already present in an untrained network (+0.796), so it is not something
learning acquires from a better input.

What the transform does is decide what learning *does not destroy*. Without it,
training drives the luminance stream from +0.417 to -0.219; with it, +0.796 is
still +0.810 after 150 epochs. The same signature appears on a second instrument
in §2.5: networks trained under the diet end magno-dominant from front ends
spanning 3.26x, including one that begins parvo-dominant in 8/8 subjects, while
every no-diet network collapses to parity from the most magno-dominant front end
in the set. The constraint acts on what learning converges to, not on what the
input supplies.

We are careful about how far this generalises. What replicates across seeds is
the *side* of parity, not a level. Across three seeds a single condition's
trained ratio ranges over 0.79 -- wider than the gap between conditions -- and
two readings we ourselves previously advanced, a convergence "to within 2%" and
a channel-specific set point, do not survive that spread.

### Three dissociations, one lesson

Shape bias, representational dimensionality, out-of-distribution robustness and
human-likeness are routinely discussed as a package. Over identical inputs, this
paper separates them three times, and a readout account says why they should
separate: none of the three is a property of a readout.

Dimensionality comes apart in the direction opposite to intuition -- §2.6.1's
`parvo_highsf` cell is more dimensional *and* more texture-biased. Robustness
comes apart in a way sharpened by its own control: the diet-trained network is
less robust than every comparison on four of the five corruption families it was
not trained on, and more robust on exactly the one family the diet installs.
Geirhos et al.'s stylisation route yields shape bias together with robustness;
our developmental route yields comparable shape bias in rate and none of it, so
the coupling belongs to the route rather than to shape bias. And human agreement
comes apart most cleanly of all.

### What generalises, and what does not

The readout account makes a testable prediction beyond this architecture: in any
system whose behaviour is measured by a cue-conflict rate, that rate should be
recoverable from a weighted combination of component biases, and manipulating
the weighting should move the rate without moving any component. Our
stream-selective readouts are the crude version of that manipulation; a
learnable or graded weighting is the natural next test.

What does not generalise is any recommendation to use a developmental diet as a
training method. It makes models less accurate (0.556 vs 0.701 at the best saved
checkpoint), less stable late in training, and less robust on four of the five
corruptions it was not trained on. Its interest is as a model of development.

### Limits

The cortical result (§2.2) is the load-bearing one and rests on eight subjects
from a single dataset. The `parvo_highsf` condition varies architecture
alongside the front end. Our diet is a fixed 48-month front end rather than a
trajectory. Human comparisons are against published observers rather than an
experiment of our own, and are comparisons of aggregate rate. The correspondence
between our chromatic stream and texture-selective ventral cortex is by
explanatory role, not by content. See *Limitations* for the full accounting.

---

## 4. Methods

> Parameters below are read from the code and run scripts, not from memory:
> `src/training/standalone_trainer.py`, `slurm/train_array.sh`,
> `src/analysis/encoding/{run_encoding,dkl_frontend}.py`,
> `src/analysis/behavior/{cue_conflict,stream_shape_texture}.py`.

### 4.1 Front end

RGB is converted to DKL through RGB→XYZ (D65)→LMS (Hunt–Pointer–Estévez)→DKL, at
256×256. **Stream 1 (magno)** receives the luminance channel under two
transforms: an isotropic Gaussian blur (σ = 4 px, kernel 8σ, reflect padding) and
a scale-free contrast-sensitivity cut evaluated at a **fixed age of 48 months**
(`cs_logspan_start` = 5×10⁻³, decomposition mode "keep"). **Stream 2 (parvo)**
receives the L−M chromatic channel untransformed.

Age is fixed rather than scheduled. This is deliberate — it separates the
transform from the trajectory — and its consequence is examined in §2.5 against
a schedule-trained checkpoint.

Four front-end conditions are used throughout:

| condition | stream 1 | stream 2 |
|---|---|---|
| `dkl_bcs` (diet) | luminance + blur + CS cut | L−M |
| `dkl_base` (no diet) | luminance, untransformed | L−M |
| `dkl_bcs_swap` | luminance, untransformed | L−M **+ blur + CS cut** |
| `dkl_bcs_parvo_highsf` | luminance + blur + CS cut | L−M **+ untransformed luminance** (2 ch) |

`parvo_highsf` gives stream 2 two input channels rather than one; this
architectural difference is measured rather than argued away (§2.4, capacity
control).

### 4.2 Architecture and training

Two ResNet-18 backbones, one per stream, with a joint classification head.
MiniEcoset at 256 px, 112 classes, pre-converted to DKL (`uint8`, with per-channel
min/max stored so that NSD stimuli are normalised identically at analysis time).

150 epochs, batch 256, Adam (lr 1×10⁻⁴, β₁ = 0.9), cross-entropy. Augmentation is
the thesis DVD-B set applied on GPU via kornia: horizontal flip (p = 0.25),
rotation ±15° (p = 0.25), colour jitter (brightness 0.4, p = 0.5), random
equalise (p = 0.5), perspective 0.3 (p = 0.5), sharpness 0.5 (p = 0.5). Random
grayscale is applied for RGB inputs only and is skipped for DKL. **Three seeds
per condition** (0, 1, 2); all twelve checkpoints are used in the seed analyses
of §2.4 and §2.5.

### 4.3 Behaviour and probe

**Cue conflict.** The Geirhos style-transfer set, restricted to the 16 categories
present in MiniEcoset and to images whose shape and texture categories differ
(1,200 images). Model logits are restricted to those 16 columns and the argmax is
scored as a shape or texture response; images decided as neither are excluded
from the denominator, and that count is reported per cell because it varies ~4×.
Streams are read out individually (`magno_only`, `parvo_only`) and jointly
(`full`). Bootstrap 95% CIs over images.

**Human observers.** The ten observers of the same release, 1,280 trials each,
scored by the identical rule. Our recomputation reproduces the published shape
bias (0.958) and the published AlexNet/GoogLeNet/VGG values.

**Error consistency** (§2.6.3). Cohen's kappa on the binary shape/texture choice
over images both parties decided: κ = (c_obs − c_exp)/(1 − c_exp) with
c_exp = p_m·p_h + (1−p_m)(1−p_h). The reference is the human–human κ over all 45
observer pairs, and every model is additionally reported as a percentage of it.

**Linear probe.** Stream features reduced to 256 PCA components (fit per stream,
so both streams are probed at equal dimensionality), then multinomial logistic
regression (C = 1.0, max_iter = 2000) under 5-fold stratified CV, predicting
shape and texture labels separately. Bias = shape accuracy − texture accuracy;
zero is the sign flip. The same CV partition is used for both streams so the
difference is a within-fold contrast.

### 4.4 Effective dimensionality

Participation ratio, (Σλ)²/Σλ², on the stream feature matrix for 1,000 NSD images
per subject. Computed Gram-side (n_images × n_images) since features outnumber
images. Reported as the magno/parvo ratio, with 1.0 — parity — as the reference.
Subjects are the replication unit within a seed; seeds are reported separately
and never pooled (§2.5).

### 4.5 Encoding

NSD `betas_fithrf_GLMdenoise_RR` (b3), converted to percent signal change and
z-scored per session per voxel, `nsdgeneral` mask. Splits are by 73k image id
with the shared-515 images held out as test, so no image appears in both.

Stream features are extracted for each subject's stimuli, reduced by PCA to 1,000
components per stream (fit on training images only, never refit on test), and
entered as two feature bands into a banded ridge (himalaya), 3-fold GroupKFold by
image id, with per-voxel alphas and per-voxel band weights. Noise ceiling from
NSD `ncsnr`; voxels below NC 0.1 are excluded, and any ROI cell with fewer than
20 surviving voxels is reported as missing rather than estimated.

Stream-wise variance partitioning gives the per-voxel shape-bias index (VSB) used
in §2.2, computed by pushing the cue-conflict set through each fitted encoding
model.

*Koniocellular control (§2.2b).* The DKL front end already computes all three
opponent axes, so the konio arm is a channel slice, not a new model: stream 2
receives S−(L+M) instead of L−M, with the same untrained network at seed 0, the
same luminance stream, the same architecture, splits and mask. Two cells were
pre-registered. In the **contrast-matched** cell S−(L+M) is scaled to L−M's
per-image RMS contrast, so the only remaining difference is which chromatic axis
stream 2 carries; this is the primary arm. In the **unmatched** cell the channel
keeps its native contrast, which is 2.05× L−M's, so a loss there is decisive
while a win alone would prove nothing. Because the magno feature matrix is
bit-identical across arms and the magno band is fitted alone, `r2_magno` must
cancel; we pre-registered this as a gate at max|Δr2_magno| < 1e-4 and it passed
bit-exact (0.000e+00) on every voxel of all 8 subjects in both cells. ROI
statistics use the Kastner2015 atlas, with the colour set (hV4, VO1, VO2) and
the specificity contrast against the remaining ROIs declared before the run.
No eccentricity-residualised VO test is reported: `prf-eccrois` labels 0% of VO2
and 9–23% of VO1 voxels, so it is not computable on this atlas.

### 4.6 Out-of-distribution robustness

Corruptions are applied to RGB **before** the front end, so each model sees the
corruption through its own front end, and are scored in the models' native 112-way
label space to avoid stacking a domain shift on top of the corruption. Five
families were declared untrained-on before the run (uniform noise, high-pass,
phase scrambling, rotation, false colour) and two as trained-in positive controls
(low-pass, contrast), since the diet *is* a low-pass and contrast manipulation.
Reported as relative accuracy AUC across severities. The RGB→DKL path was gated
on an exact reproduction of the training h5 before any number was recorded.

### 4.7 Statistics

Within-subject comparisons use the paired exact Wilcoxon signed-rank test, n = 8
(exact two-sided floor p = .0078). Seed-level comparisons use the exact
Mann–Whitney U, n = 3 v 3 (exact two-sided floor p = 0.100, reached only on
complete separation); seeds are never pooled with subjects, and a seed-level claim
requires the within-seed subject test to hold in all three seeds independently.

**Two-statistics rule.** Every claim is made only where the normalised and raw
statistics agree; disagreements are reported as unresolved rather than resolved
in the favourable direction.

**Pre-registration.** The front-end 2×2 (§2.4), the inverted-front-end prediction
(§2.5), the OOD families (§2.6.2) and the error-consistency hypothesis (§2.6.3) were
each written down before the corresponding analysis was run. Two of the four
predictions failed and are reported as failures in the sections above.

---

## Limitations

1. **The diet costs, on every engineering axis we measured.** ~13% of explained
   BOLD variance (§2.7); 20.7% relative classification accuracy at the best
   saved checkpoint (§2.6.2); late-training stability (§2.6.2); and robustness on
   four of the five corruptions it was not trained on (§2.6.2). The paper makes no
   claim that
   the diet improves a model. **An earlier version of this manuscript reported a
   37% accuracy cost and markedly higher seed variance; both were artefacts of
   using the final rather than the best checkpoint, and are corrected here
   (PLAN §67.1). A subsequent version reported the cost as 16.5%, taken from
   per-epoch validation maxima at epochs whose weights were never saved; the
   figure quoted here is the best checkpoint that exists and can be
   re-evaluated (PLAN §71).**
2. **Two front-end controls are degenerate on the cortical instrument** (§2.7),
   which narrows the control claim.
3. **`parvo_highsf` varies architecture alongside the front end** — its second
   stream has two input channels rather than one, so input and first-layer
   fan-in change together and no existing model separates them. The capacity
   confound that would fake the result is ruled out (§2.4), but the comparison
   is not architecture-matched. This qualifies §2.5's decisive case and must
   travel with it.
4. **There is no set point, and the trained level is not reproducible across
   seeds.** Earlier drafts of this work reported the two diet-on-luminance arms
   as converging to within 2% (a gap of 0.037), and then that the level was
   channel-specific (red−green at 2.327 against luminance at 1.703). The seed
   replication retires both: the within-arm seed range on the luminance arm is
   **0.120**, still larger than that 0.037 gap, and the red−green and luminance
   ranges overlap (1.541–2.327 vs 1.602–1.722), so n=3 seeds cannot establish
   channel-specificity either. The paper therefore reports **no level, no set
   point and no compression factor anywhere** — only the side of parity, which
   replicates 12/12 seeds with complete separation (§2.5, PLAN §58.3).
5. **The independent thesis checkpoint lands at 1.158** — below every one of our
   diet seeds (1.497–2.327) but on the same side of parity, in 8/8 subjects. It
   is a different codebase and a real age schedule rather than our fixed age 48,
   feature-extracted through a schedule that is not its own, so it is the least
   comparable arm. It is reported as a convergent check on the *side* claim, and
   as the paper's only evidence that a genuine developmental trajectory behaves
   like our static front end (§2.5, PLAN §58.5).
6. **n=8 is subjects-as-image-samples for the PR analyses** — participation ratio
   is computed on each subject's cached image set, so the paired test speaks to
   stability across image samples, not to population inference. The encoding and
   behavioural results are not affected.
7. **The human comparison is directional only.** The best stream (0.878) is
   below the lowest individual human observer (0.919), and the integrated readout
   (0.675) is far below. Human data is in hand and cited (§52); what is not
   available is any basis for a matching claim.
8. **Single-trainer accuracy discrepancy, unexplained.** The thesis checkpoint's
   *behavioural* number reproduces (§18.2) but its *accuracy* does not; the
   discrepancy is isolated to the classification evaluation path, not the
   checkpoint or front end (PLAN §13.2).


9. **The mapping to ventral cortex is by role, not by content.** Our
   texture-biased stream is chromatic (L-M); the texture-like representations
   reported in human ventral cortex are not specifically chromatic. What §2.2
   establishes is that the texture-biased *component of this model* is the one
   whose unique variance carries the ventral end, not that it carries the same
   features. The paradox argument needs only the former; nothing here asserts
   the latter.
10. **The ventral claim is relational, not absolute.** hV4's absolute parvo
   preference on the 20-ROI profile is weak (-0.012, 2/8 subjects). The
   load-bearing statistic is the *gradient* -- hV4 minus V1v, and Spearman rho
   against hierarchical rank -- so the claim is that the parvo stream's
   explanatory share rises toward the ventral end, never that hV4 is a parvo
   area.
11. **Learning contributes +0.014 to the magno stream's shape bias.** The
   shape-biased component is a property of the front end, established before
   training (§2.1). Every statement in this paper about what training does
   concerns whether that bias *survives*, not whether it is created. The
   untrained references are single models with no training seed.
12. **Whether a two-stream architecture specialises on its own is untestable
   here.** The two trunks are bit-identical at initialisation, so the answer we
   can support is "not this one, by construction". Testing the general question
   needs symmetry broken at initialisation -- a per-stream seed -- which is a
   new set of training runs (3 seeds x ~2 days GPU) and is not reported.
13. **`rgb_tsn`'s channels are DKL lum/rg/by, not R/G/B**, and this condition
   alone was trained with `RandomGrayscale(p=0.5)`.


## Claims deliberately not made

Recorded so the omissions are visible rather than silent.

- **Where in visual cortex the diet's shape-bias effect lands.** We tested
  whether the per-voxel predicted shape-bias profile carries a diet-specific
  hV4/LO step. Under the full control set it does not (PLAN §43.2, §44.1). We
  therefore make no claim about cortical localisation of the shape bias, and the
  analysis is reported in supplementary material rather than the main text. Note
  this is a *different* measure from §2.2(b)'s stream-bias gradient, which is
  diet-specific and survives its controls.
- **That a koniocellular stream is or is not worth building.** The konio control
  of §2.2(b) swaps which chromatic axis stream 2 carries; it does not add a third
  stream. The architecture that would justify this network's name — a thin
  quarter-width K stream with dual projection (K→parvo after block 1, late skip
  to magno; Sincich 2004) — is not implemented and not tested. What the
  result refutes is the premise that motivated it, namely that S−(L+M) is the
  chromatic axis the failed colour prediction was missing. It is also an
  *untrained*, single-seed comparison, so nothing follows about a trained konio
  stream, and no blur was applied to the K channel: the measured radial spectrum
  of this pipeline's S−(L+M) is nearly L−M's in shape (0.614 vs 0.570 of power in
  the lowest octave), so blurring would impose the low-acuity assumption rather
  than reproduce it. Whether a three-band model beats a two-band one is a
  separate question needing 3-way variance partitioning, which our banded ridge
  does not implement.
- **That integration produces human-like shape bias** (§2.7).
- **That the diet improves neural predictivity** — it does not (§2.7).
- **Any dorsal/ventral stream mapping.** The BA framing was
  "magno→dorsal, parvo→ventral". We tested it rather than dropping it on
  argument. The pre-declared dorsal−ventral interaction on the `streams` atlas
  is +0.0245 (p_perm = .0079, 8/8 subjects) on the diet model — and
  **+0.0122, p = .195 without the low-pass front end**, so it does not survive
  its own control and is withdrawn (PLAN §8.2, §8.3).
- **hMT+ specificity.** Kastner2015 supplies TO1/TO2 (hMT+), so the dorsal test
  is possible on static images after all. At n=7 hMT+−VO looked real (p = .047);
  **at n=8 the two statistics disagree (p = .195 normalised, p = .039 raw)**, so
  the §8 two-statistics rule refuses it, and it is already dead in the blur
  control. An earlier draft of our own analysis promoted hMT+ from "limitation"
  to "result"; that promotion was a false positive from an incomplete subject
  set and is retracted (PLAN §16.1).
- **The eccentricity headline.** "Parvo dominates the fovea, magno the
  periphery" was a candidate headline. The foveal band's sign **flips** when the
  blur is removed (−0.057, 0/8 → +0.051, 7/8), so it is an artefact of the front
  end, not a biological result (PLAN §8.3). What survives is only that magno's
  *absolute* unique-variance advantage grows with eccentricity in both models
  (raw slope +0.047 vs +0.041, 8/8 both, difference n.s.).

  **Why these three failed together, and what it implies for the survivor.**
  Without the low-pass front end essentially *every* ROI is magno-preferring.
  The blur manufactures apparent regional specificity by handicapping the magno
  stream until the parvo stream can win somewhere — the same general-factor
  pathology that defeated the thesis's RSA, reappearing in encoding form. Any
  regional claim on these models must therefore be shown to survive removal of
  the blur. §2.2(b)'s ventral gradient is reported *because* it does. Run at the
  input level, where the blur can actually be removed — the untrained network on
  the diet front end against the same network on the no-blur front end — the
  gradient gets **larger** without the blur on the raw statistic (−0.118 →
  −0.208) and is unchanged on the normalised one (−0.471 → −0.488, difference
  p = .383). It is also not eccentricity: the eccentricity-matched contrast is
  −0.736 (p = .008, 0/8, dz −5.72). The blur is thus not what manufactures this
  particular gradient, which is exactly what could not be said of the three
  results above.
- **A clean dorsal/ventral organisation of the 20-ROI profile.** The profile
  across Kastner2015's 20 ROIs is highly reproducible (leave-one-subject-out
  Spearman ρ = +0.794, range .589–.896) and strongly graded — V1 > V2 > V3 >
  rest, with **hV4 the only parvo-preferring area** — but it ranks by
  hierarchical level and eccentricity, *not* by a dorsal/ventral split. We
  report the gradient (§2.2b) and make no anatomical-stream claim.
- **That the chromatic stream predicts colour-biased cortex.** The natural
  prediction from a parvo/chromatic stream is that it should win in VO. It does
  not: pooled VO is **+0.0045 (p = .46)** — if anything magno-preferring — on the
  same 20-ROI profile, and hV4's parvo preference (−0.012, 2/8) is weak. This is
  a failed prediction and we report it as one. It is also what makes the konio
  control of §2.2(b) interpretable: because L−M already fails at VO, the two
  live readings were "the chromatic stream does not predict colour cortex" and
  "L−M is the wrong chromatic axis". Swapping in S−(L+M) settles it as the
  first — the substitute is *reliably and specifically worse* in exactly those
  ROIs — so both opponent axes have now been tested against colour-biased cortex
  and neither predicts it. What §2.2(b) claims is a hierarchical gradient in
  which stream does the explanatory work, not that either stream maps onto a
  colour-selective region (PLAN §7.1, §70.4).


- **That human cortex uses this mechanism.** §2.3 is an existence proof with a
  cortically grounded demonstration: a system can be behaviourally shape-biased
  while its ventral-explaining component is texture-biased, and this one is. We
  do not claim that the human readout works this way, and the single-stream
  alternative -- texture-like ventral representations feeding shape-biased
  behaviour directly -- is not excluded by these data. The claim is that the
  reported tension is not a contradiction, not that we have measured its
  resolution in humans.
- **That two-stream architecture causes the specialisation.** The streams are
  bit-identical at initialisation and stay so under identical input, so the
  architecture-only comparison is an algebraic identity, not a test
  (Limitations 12). What the construction buys is attribution: every measured
  stream difference is attributable to the input.


## Open items before submission

- [x] **Geirhos human observer data obtained** (2026-09-02) —
      `/scratch/byrong/VST_data/geirhos_human_data/cue_conflict`, computed by
      `tmp_analysis/geirhos_humans.py`, recorded in PLAN §52. Human mean 0.958;
      our CNN recomputation reproduces the published values. Every human
      reference stays directional; "human-like" stays out.
- [ ] Cite the Geirhos **data release** as well as the ICLR paper. **Licence
      not yet confirmed:** only the per-observer CSVs were fetched into
      `/scratch/byrong/VST_data/geirhos_human_data/`, and no LICENSE or README
      came with them, so the terms must be checked at the source repository
      before submission. Do not assert a licence in the manuscript until then.
- [x] **Human benchmark is in the abstract** (2026-09-02), but as a *rate*
      comparison carrying its own qualification — the whole-model figure (0.692)
      is given alongside the stream figure (0.878), and the error-consistency
      null (§2.6.3) is stated in the same paragraph.
- [x] **Figure 1 built** — `results/figures/fig1_convergence.{png,pdf}`, caption
      in `fig1_convergence_caption.md`, built by `tmp_analysis/fig1{,_data}.py`
      (PLAN §51). Panel (c) is the set point as a front-end→trained slopegraph,
      superseding §46.3's retention spec per §49.
- [x] **Ventral-gradient figure renumbered** (2026-09-03) —
      `fig1_diet_gradient.{png,pdf}` → `fig2_ventral_gradient.{png,pdf}`,
      caption → `fig2_ventral_gradient_caption.md`. It is §2.2(b)'s evidence;
      Figure 1 is the four-panel convergence figure (PLAN §51.4).
- [x] **Venue: NHB** (decided 2026-09-02, re-confirmed 2026-09-06 after the
      reframe). NeurIPS/NMI dropped — there is no ML payoff to report, only
      costs on accuracy, stability and robustness (PLAN §56.6). Under the
      *first* draft's framing the claim was about what makes a CNN
      shape-biased, and an internal referee simulation returned reject-with-
      transfer on scope (`docs/NHB_REVIEW_2026-09-06.md`, Relevance 1.7). The
      reframe answers the originating thesis's Research Question 1 — why human
      behaviour is shape-biased when ventral representations are texture-like —
      which is a question about human vision, and restores the NHB case
      (`docs/FRAME_PROPOSAL_2026-09-06.md`). Fallbacks if scope is still
      judged too narrow: eLife, Nature Communications.
- [ ] **Blockers carried over from the referee simulation**, none needing new
      compute: no References section at all; `±` undefined throughout though it
      carries the §2.5 headline table; ~44 p-values with no multiplicity
      statement; the three §2.6 dissociations are nulls with no equivalence
      bound or Bayes factor; no ethics / data / code / materials statements for
      either human dataset. See `docs/NHB_REVIEW_2026-09-06.md` §"Specific
      Improvement Suggestions" for the ranked list.
- [ ] **Length.** Main text is ~9,300 words against NHB's ≈4,000–6,000, up from
      ~7,950 because the reframe adds its own framing before anything was cut.
      The six self-correction passages in Results are project history rather
      than results and belong in Supplementary (~1,500–2,000 words).
- [ ] **Add §72's single-stream control** — still absent. It answers "why two
      streams at all?", which under the new frame is the first question a
      referee asks. Split buys 0.088 shape bias, costs 0.043 accuracy, at half
      the parameters, complete separation under both checkpoint policies.
- [x] **Methods expanded** (2026-09-03) — §4.1–4.7, every parameter read from
      the code and run scripts rather than from memory; 3 seeds per condition
      stated, and the subject vs seed inferential axes separated in §4.7.
- [ ] Supplementary: the VSB cortical profile with its negatives intact; the
      capacity ladder (PLAN §42) with its non-monotonicity caveat.
- [x] **Cortical seed replication complete** (2026-09-04) — array 26416428,
      32/32 COMPLETED. §2.2 replicates across three seeds with both statistics
      agreeing (diet seed range 0.032). **The paper now has no n=1 claim in its
      main results** (PLAN §66).
