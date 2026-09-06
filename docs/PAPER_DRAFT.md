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

Shown an object whose shape and texture disagree, humans name the shape;
convolutional networks name the texture (Geirhos et al., 2018a). Set against
this is evidence pointing the other way: representations along the ventral
stream, the pathway most identified with object recognition, are well described
by texture statistics (Jagadeesh & Gardner, 2022; Okazawa et al., 2015). We show
the two facts are compatible, and give a mechanism.

In a two-stream network receiving approximations of the magnocellular and
parvocellular signals reaching cortex, the streams end training with opposite
shape–texture biases, reached by different routes: the luminance stream's shape
bias is supplied by its input statistics, while the chromatic stream's texture
bias is built by learning. Fitted to eight subjects of the Natural Scenes
Dataset, it is the *texture-biased* stream whose unique variance carries the
ventral end of the hierarchy. Behaviour reads out from both, and the mixture is
shape-biased. Reweighting that readout over frozen representations moves shape
bias from 0.40 to 0.90, spanning the range the field's models occupy without
either component changing.

Shape-biased behaviour therefore need not imply shape-biased representations: a
behavioural rate is a property of a readout over components, not a report of any
one of them. Consistent with this, three properties the field treats as
travelling with shape bias — representational dimensionality, out-of-distribution
robustness, and trial-by-trial agreement with human observers — do not. The
correspondence we draw to cortex is one of explanatory role, not of content.

---

## 1. Introduction

> DRAFT -- argument fixed, prose to be tightened.

Humans presented with an image whose shape and texture point to different
categories name the shape. Standard convolutional networks name the texture.
That dissociation, measured on a cue-conflict set that ten human observers also
viewed (Geirhos et al., 2018a), is the most conspicuous behavioural gap between
human and machine object recognition, and closing it has become a standard
target for biologically motivated models (Hermann et al., 2020; Jang & Tong,
2021; Lu et al., 2026).

A second body of evidence points the other way. Along the ventral stream -- the
pathway most identified with object recognition -- object representations are
well described by texture statistics. In humans, category-selective cortex fails
to discriminate natural images from feature-matched scrambles that destroy
spatial arrangement while preserving texture statistics, yet still discriminates
between categories (Jagadeesh & Gardner, 2022). In macaque, V4 responses to
natural textures are predicted by their image statistics (Okazawa et al., 2015),
and in IT both shape and texture are represented but texture is the more
accurately decodable of the two (Jagadeesh & Livingstone, 2024); material and
surface appearance recruit an extended ventral network (Komatsu & Goda, 2018).

*We state the scope of this premise precisely, because the paper rests on it.*
The human evidence is one study using one image-synthesis manipulation in
category-selective regions; the remainder is macaque, and "texture-like" in each
case means that a texture-statistics description suffices for the measured
responses, not that shape is absent. The tension we address is between that body
of work and the behavioural result, and our argument needs only that the ventral
description is *substantially* texture-like -- not that ventral cortex is
texture and nothing else.

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

The raw statistic agrees: **+0.616 with the diet against −0.033 without** (and
+0.658 / −0.039, +0.632 / −0.053 in seeds 1 and 2, p=.0078 and 8/8 throughout).
**We draw no inference from the sign of the no-diet values.** An earlier version
described the raw statistic as separating the arms "by sign rather than by
degree" and read the no-diet arm as marginally texture-organised; both are
withdrawn. Those values sit near zero, the quantity's behaviour near zero has not
been characterised, and the claim this section makes — that the arms separate,
at the exact Wilcoxon floor in 8/8 subjects on both statistics — does not need
them to fall on any particular side of it.

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
checkpoints. (An earlier version of this comparison gave 28-fold, using a §2.5
seed range that was itself a final-checkpoint artefact; Supplementary Note S1.)
The raw statistic agrees on every seed, so the two-statistics rule is satisfied three
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

**This gradient was not pre-specified, and it is the survivor of a family.** We
state that here rather than leave it to be assembled from the *Claims
deliberately not made* section. Seven regional hypotheses were examined on these
same fitted encoding models. **Six failed and are reported as failures:** a
diet-specific hV4/LO step in the per-voxel predicted shape-bias profile; the
pre-declared dorsal−ventral interaction on the `streams` atlas (which does not
survive removal of the low-pass front end, p = .195); hMT+ specificity (the two
statistics disagree at n=8, so the §8 rule refuses it); the foveal/peripheral
eccentricity contrast (its sign flips when the blur is removed); a dorsal/ventral
organisation of the 20-ROI profile (it ranks by hierarchical level and
eccentricity instead); and a parvo preference in colour-biased cortex (+0.0045,
p = .46, bounded by equivalence to within ±0.017). The V1v→hV4 gradient is the
seventh, and the one that survived.

**The controls below address confounding, not selection.** Blur removal,
monotonicity, the cross-codebase replication and the global-preference contrast
each rule out a way the gradient could be an artefact; none of them speaks to the
fact that it was chosen after six siblings failed. Readers should weigh it as
what it is — one surviving regional result out of seven, with the failures
visible — and the strongest thing we can say for it is not a p-value but that it
is the only one of the seven that survives removal of the very front-end
manipulation that produces the apparent regional structure in the first place
(see the closing paragraph of *Claims deliberately not made*, where that
diagnosis is set out for the six).

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


**Integration mixes rather than inherits.** The integrated readout's
cue-conflict bias is **0.675**, against **0.878** for the magno stream alone and
**0.418** for the parvo stream alone. Integration therefore *costs* 0.203 of
shape bias — 72% of the remaining distance to the human mean.

*We are careful about what that single number can show.* The readout is a fixed
linear sum of the two streams' features at w = 0.5, and w is a hyperparameter,
not a learned quantity (§4.3). Given two oppositely biased components, an
intermediate result is close to arithmetically forced, so **the intermediate
value is not by itself a prediction that could have failed** and we do not
present it as one.

**The readout weight is the control variable, and it moves behaviour across the
entire range.** What is *not* forced is the shape of the dependence, and that is
measurable, because `y₁` and `y₂` do not depend on `w`: sweeping `w` moves the
readout while holding both components **exactly** fixed. This is the manipulation
the readout account calls for. Cue-conflict shape bias at 21 values of `w`,
3 seeds, both checkpoint policies:

| `w` (weight on the magno stream) | 0.00 | 0.25 | 0.50 | 0.75 | 1.00 |
|---|---|---|---|---|---|
| final checkpoints | 0.419 | 0.508 | **0.675** | 0.827 | 0.879 |
| best saved checkpoints | 0.399 | 0.508 | **0.655** | 0.807 | 0.904 |

**Shape bias runs from below parity to 0.90 as the readout is reweighted over
components that never change.** The curve is monotone non-decreasing across all
21 values in 3/3 seeds at the best saved checkpoint (2/3 at the final
checkpoint, where one seed has a single local inversion), and the total span is
**+0.46 to +0.52** of shape bias per seed. The same network, the same two
representations, anywhere from texture-biased to more shape-biased than any
model in §2.6.3's table — set only by how the readout weights them.

That is the paper's claim made as a measurement rather than an argument. A
behavioural shape-bias rate is a property of a readout over components; move the
readout and the rate moves, with the representations held fixed.

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

**Nothing here fits `w` to humans.** No value of the weighting was estimated
from human data, and the sweep does not identify which mixture a human readout
uses — only that a mixing readout over these components spans the entire
behavioural range, and that the system's shape bias is therefore a fact about
the readout rather than about either representation.

**The split is doing the work: without it the behaviour is worse.** The
immediate objection to a two-component account is that the components may be
decorative -- that one trunk carrying both channels, with the identical diet,
would behave the same. We trained that control (`dkl_bcs_single`, 3 seeds,
one ResNet-18 trunk taking both DKL channels stacked, same diet, same schedule):

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

> magno stream alone **0.903**  >  two-stream integrated **0.655**  >  single
> stream **0.567**,

which locates the cost precisely: integration costs 0.248 relative to the best
single component, but having two components to integrate is still worth 0.088
over having one. So integration is not what destroys the shape bias — a single
stream carrying the same information does worse than the integrated pair. The
remedy for the integration cost is not to drop the split.

*Two limits.* This tests the split for the **diet-on-luminance arm only** and
must not be generalised to `swap` or `parvo_highsf`. And the capacity confound
is retired for the accuracy comparison (the smaller model wins) but **not** for
the shape-bias comparison: a parameter-matched single stream (`resnet34`,
~21.3M) might close the 0.088 gap. That arm was pre-registered and has not been
run; it is 3 seeds × ~26 h of GPU and its absence is a decision, not an
oversight (PLAN §72.6).

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
varies by **0.014** across seeds — the tightest number in this paper — against
**0.120** for §2.5's magno/parvo dimensionality ratio on the same three
checkpoints, both scored at the best saved checkpoint. §2.5 claims a side rather
than a level because its within-arm spread exceeds its between-arm gap, not
because its instrument is unstable (Supplementary Note S1).

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
| diet | luminance | `dkl_bcs_seed0` | 1.402 ±0.095 | **1.703** ±0.059 |
| diet | luminance | `dkl_bcs_parvo_highsf_seed0` | **0.586** ±0.014 | **1.640** ±0.032 |
| diet | **red−green** | `dkl_bcs_swap_seed0` | **1.909** ±0.133 | **2.327** ±0.085 |
| no diet | — | `dkl_base_seed0` | 1.634 ±0.126 | 0.947 ±0.038 |
| no diet | — | `dkl_base_seed1` | 1.634 ±0.126 | 0.988 ±0.043 |

*`±` here and in §2.6.1 is the sample standard deviation across the eight
subjects (ddof = 1), unified 2026-09-06. Seventeen cells across the two tables
previously mixed two estimators; fifteen have been restated and two were already
ddof = 1. **No mean, ordering, separation or claim changes** — every dispersion
that moved moved by the same √(8/7) = 1.069 — so a reader comparing against an
earlier version will see seventeen numbers change and no result change.*

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

**The seed ranges above are best-checkpoint values.** Scored at final
checkpoints they would read 0.904 and 2.388, both inflated by the late-training
degradation of §2.6.2; on validation-selected weights they fall to 0.120 and
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
Equivalence: the difference is **within ±0.033** (Δ = +0.0092, 90% CI
[−0.0145, +0.0330], TOST p = .0287), below the smallest OOD effect this paper
claims (rotation, 0.041). This is the weakest of the paper's four equivalence
results and is labelled as such: with 3 seeds per arm it rests on a Welch
interval at **2.3 degrees of freedom**, and PLAN §77.4 declared in advance that
no *nonparametric* equivalence result is obtainable at 3v3 at any bound.

The **only** corruption on which the diet is more robust is low-pass — which we
declared trained-in before running, because the diet *is* a low-pass blur. The
experiment finds robustness exactly where robustness was trained in and nowhere
else, which is the strongest available evidence that the four nulls are real
rather than a broken measurement. Note that the diet is *not* more robust to
contrast reduction even though its front end includes a contrast-sensitivity
cut, so "trained-in" does not automatically confer robustness either.

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

Across all seven models the shape-bias rate spans 0.17 to 0.89 — a 5× range —
while trial-level agreement with humans never exceeds **18%** of the
human–human ceiling for any of them. **Shape bias, the field's
most-cited human-alignment metric for vision models, does not track human
decision structure.** That is this paper's third dissociation and the only one
that concerns human behaviour directly.

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
In `swap` the parvo stream contributes unique variance Uₚ = 0.006; in
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
identify a *decision process* either: across seven models spanning a 5x range of
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
the weighting should move the rate without moving any component. We test that
directly in §2.3 — reweighting the readout over frozen components moves shape
bias from 0.40 to 0.90, monotonically, spanning the entire range the field's
models occupy. What remains open is the *learned* version: a weighting fitted to
data rather than swept, and whether any principled objective selects the human
operating point. Nothing here selects it.

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
the thesis DVD-B set applied on GPU via kornia, each with its application
probability *p*_aug: horizontal flip (*p*_aug 0.25), rotation ±15° (*p*_aug
0.25), colour jitter (brightness 0.4, *p*_aug 0.5), random equalise (*p*_aug
0.5), perspective 0.3 (*p*_aug 0.5), sharpness 0.5 (*p*_aug 0.5). Random
grayscale is applied for RGB inputs only and is skipped for DKL. **Three seeds
per condition** (0, 1, 2); all twelve checkpoints are used in the seed analyses
of §2.4 and §2.5.

### 4.3 Behaviour and probe

**Cue conflict.** The Geirhos style-transfer set, restricted to the 16 categories
present in MiniEcoset and to images whose shape and texture categories differ
(1,200 images). Model logits are restricted to those 16 columns and the argmax is
scored as a shape or texture response; images decided as neither are excluded
from the denominator, and that count is reported per cell because it varies ~4×.
Bootstrap 95% CIs over images.

**The readout, stated explicitly.** The classifier is linear and is applied to a
weighted sum of the two streams' penultimate features,

> `full = classifier(w·y₁ + (1−w)·y₂)`,  `magno_only = classifier(w·y₁)`,
> `parvo_only = classifier((1−w)·y₂)`,

with `y₁`, `y₂` the two trunks' outputs. **The fusion weight `w` is a
hyperparameter, not a learned parameter: it is fixed at 0.5 for every model
reported here** (`TwoStreamNetwork.stream1_weight`; a separate `mheavy`
condition at w = 0.9 exists and is not used in this paper). The two
stream-restricted readouts are therefore the same expression with one term
zeroed, and they pass through the same classifier and the same bias term as
`full`, which is why they are directly comparable to it. Cross-stream mixing
(`stream_concat`) is **off** in every condition reported here, so `y₁` and `y₂`
carry no information from each other. §2.3 reports shape bias as a function of
`w`; because `y₁` and `y₂` do not depend on `w`, that sweep moves the readout
while holding both components exactly fixed.

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

**What "pre-registered" means here, exactly.** These predictions were written
into a version-controlled project plan (`docs/PAPER_PLAN.md`) and committed
before the corresponding analysis was run, so the ordering is verifiable from
the git history and the relevant commit hashes are given in the Supplementary.
This is *internal pre-specification*, not registration with a public registry:
there is no OSF or AsPredicted entry, no third party held the plan, and nothing
prevented the plan itself from being revised. We use the term in that restricted
sense throughout and claim nothing stronger.

**Uncertainty notation.** Throughout, `±` denotes a **standard deviation**, over
the unit named in each case: across the three training seeds for behavioural and
probe quantities, across the ten observers for the human figure (0.958 ±0.019),
and across the eight NSD subjects for the participation-ratio and
stream-preference tables. Ranges, where given, are stated as explicit intervals
("range 0.919–0.987", "seed range 0.120") and are never written as `±`.
Cue-conflict values additionally carry a **bootstrap 95% CI over cue-conflict
images** (§4.3), which is labelled as a CI wherever it appears.

*One inconsistency, found and resolved.* The manuscript previously carried
**three** groups of cells rather than one: behavioural and probe tables at the
sample SD (ddof = 1); most participation-ratio and stream-preference cells at
NumPy's default population SD (ddof = 0); and two cells of §2.5's "after
training" column — `dkl_bcs_seed0` (±0.059) and `dkl_bcs_swap_seed0` (±0.085) —
**already** at ddof = 1, because they were re-scored at their best saved
checkpoints through a different routine than the rest of the column. All
seventeen cells were re-derived from the feature cache, each checked against its
currently printed value before any replacement was emitted, and the fifteen
ddof = 0 cells restated at ddof = 1; the two already-correct cells were left
alone, since rescaling them would have printed 0.063 and 0.091. A single
estimator now carries the symbol. No mean, ordering, separation or claim is
affected.

**Multiplicity, and why no correction can license this design.** This paper
reports 28 p-values (an earlier count of 36 wrongly included seven kornia
augmentation probabilities and one unique-variance value; those are now written
*p*_aug and Uₚ).

The honest statement about multiplicity here is a negative one. Nearly every
inferential claim rests on the paired exact Wilcoxon signed-rank test at n = 8,
whose **exact two-sided floor is p = 2/2⁸ = .0078125** — the value returned by a
*perfect* 8/8 split, and unimprovable no matter how large the effect (our
largest, dz = −11.70, returns exactly the same .0078125 as a marginal 8/8).
Holm–Bonferroni over a family of *m* tests requires the smallest p to clear
.05/*m*. Since .0078125 > .05/*m* for every *m* > 6, **no family larger than six
tests can have a single rejection at α = .05**, and for our 18-test
pre-specified confirmatory family Holm rejects nothing at all: every adjusted
value is min(1, 18 × .0078125) = **.141**. Benjamini–Hochberg returns .141 too.
This is a resolution limit of an exact test at n = 8, not a choice-of-procedure
problem, and it cannot be escaped by selecting a smaller family after seeing
which claims need to survive.

We therefore do not claim multiplicity-corrected significance anywhere, and no
claim in this paper should be read as carrying it. What the claims rest on
instead is stated with each of them and is not a p-value: **complete separation
across all eight subjects (8/8 or 0/8), agreement of the raw and
noise-ceiling-normalised statistics (§8), independent replication across three
seeds, and effect size.** The p-values are reported because they are
conventional and because their floor is informative about the design, not
because they discriminate among our results — at n = 8 they cannot.

The exploratory ROI screen behind §2.2(b) is treated the same way and is the
reason four regional analyses appear under *Claims deliberately not made* as
failures rather than as corrected significances. Equivalence tests (§78) are
one-sided with bounds fixed in advance and each bounds a separate pre-declared
null; they are reported with their achieved bounds and their power caveats
rather than corrected.

---

## Reporting and availability statements

**Ethics.** This study analysed two existing human datasets and collected no new
human data. Neuroimaging data are from the Natural Scenes Dataset (Allen et al.,
2022), collected under approval of the University of Minnesota institutional
review board with written informed consent from all eight participants; see that
publication for the full protocol. Behavioural data are the ten observers of
Geirhos et al. (2018a, 2021), collected under the approval described in those
papers. Our secondary analysis of both datasets required no additional approval
under our institution's policy for de-identified, publicly released data. No
participant-identifying information was accessed at any point.

**Sample and constraints on generality.** The behavioural reference is ten adult
observers tested under controlled laboratory viewing in a German university
laboratory; the neuroimaging sample is eight adults in the United States. Neither
sample's demographics beyond adulthood are reported in the source publications at
a level we can restate, and we do not restate them. Both are small, Western,
convenience-recruited samples of adults with normal or corrected-to-normal
vision. Claims in this paper about "human observers" should be read as claims
about these samples; nothing here establishes cross-cultural, developmental or
clinical generality, and the developmental motivation of the front end is a
motivation for the model, not a claim about any developmental population we
measured.

**Data availability.** The Natural Scenes Dataset is available at
`naturalscenesdataset.org` under its own data-use agreement, which requires
registration; we used the `betas_fithrf_GLMdenoise_RR` (b3) preparation and the
`nsdgeneral` mask. The Geirhos cue-conflict stimuli and human observer responses
are distributed at `github.com/rgeirhos/texture-vs-shape` and
`github.com/bethgelab/model-vs-human`. MiniEcoset is at `osf.io/msna2` (Thorat &
Kietzmann, 2024). Derived results tables and the fitted encoding-model summaries
needed to reproduce every figure will be deposited with a DOI on acceptance.

**Code availability.** Training, front-end, encoding, behavioural and analysis
code will be released with a DOI and an environment manifest on acceptance. The
modules that produce the numbers reported here are named by path in §4.1–§4.7.

**Materials.** The cue-conflict stimulus set is the public Geirhos release, used
unmodified apart from the 16-category and shape≠texture restrictions described
in §4.3. **The licence of that release is being confirmed with the authors and
will be stated before submission; we make no licence assertion here.**

**Competing interests.** The authors declare no competing interests.

---

## References

Alain, G., & Bengio, Y. (2016). Understanding intermediate layers using linear
classifier probes. *arXiv*. https://doi.org/10.48550/arXiv.1610.01644

Allen, E. J., St-Yves, G., Wu, Y., Breedlove, J. L., Prince, J. S., Dowdle,
L. T., Nau, M., Caron, B., Pestilli, F., Charest, I., Hutchinson, J. B.,
Naselaris, T., & Kay, K. (2022). A massive 7T fMRI dataset to bridge cognitive
neuroscience and artificial intelligence. *Nature Neuroscience*, 25(1), 116–126.
https://doi.org/10.1038/s41593-021-00962-x

Bakhtiari, S., Mineault, P., Lillicrap, T., Pack, C., & Richards, B. (2021). The
functional specialization of visual cortex emerges from training parallel
pathways with self-supervised predictive learning. *Advances in Neural
Information Processing Systems*, 34, 25164–25178.

Casagrande, V. A. (1994). A third parallel visual pathway to primate area V1.
*Trends in Neurosciences*, 17(7), 305–310.
https://doi.org/10.1016/0166-2236(94)90065-5

Cohen, J. (1960). A coefficient of agreement for nominal scales. *Educational and
Psychological Measurement*, 20(1), 37–46.
https://doi.org/10.1177/001316446002000104

Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., & Fei-Fei, L. (2009).
ImageNet: A large-scale hierarchical image database. *IEEE Conference on Computer
Vision and Pattern Recognition*, 248–255.
https://doi.org/10.1109/CVPR.2009.5206848

Derrington, A. M., & Lennie, P. (1984). Spatial and temporal contrast
sensitivities of neurones in lateral geniculate nucleus of macaque. *The Journal
of Physiology*, 357(1), 219–240. https://doi.org/10.1113/jphysiol.1984.sp015498

Derrington, A. M., Krauskopf, J., & Lennie, P. (1984). Chromatic mechanisms in
lateral geniculate nucleus of macaque. *The Journal of Physiology*, 357(1),
241–265. https://doi.org/10.1113/jphysiol.1984.sp015499

DiCarlo, J. J., Zoccolan, D., & Rust, N. C. (2012). How does the brain solve
visual object recognition? *Neuron*, 73(3), 415–434.
https://doi.org/10.1016/j.neuron.2012.01.010

Doerig, A., Sommers, R. P., Seeliger, K., Richards, B., Ismael, J., Lindsay,
G. W., Kording, K. P., Konkle, T., Van Gerven, M. A., Kriegeskorte, N., &
Kietzmann, T. C. (2023). The neuroconnectionist research programme. *Nature
Reviews Neuroscience*, 24(7), 431–450.
https://doi.org/10.1038/s41583-023-00705-w

Dupré la Tour, T., Eickenberg, M., Nunez-Elizalde, A. O., & Gallant, J. L.
(2022). Feature-space selection with banded ridge regression. *NeuroImage*, 264,
119728. https://doi.org/10.1016/j.neuroimage.2022.119728

Felleman, D. J., & Van Essen, D. C. (1991). Distributed hierarchical processing
in the primate cerebral cortex. *Cerebral Cortex*, 1(1), 1–47.
https://doi.org/10.1093/cercor/1.1.1-a

Ferrera, V. P., Nealey, T. A., & Maunsell, J. H. R. (1994). Responses in macaque
visual area V4 following inactivation of the parvocellular and magnocellular LGN
pathways. *Journal of Neuroscience*, 14(4), 2080–2088.
https://doi.org/10.1523/JNEUROSCI.14-04-02080.1994

Gao, P., Trautmann, E., Yu, B., Santhanam, G., Ryu, S., Shenoy, K., &
Ganguli, S. (2017). A theory of multineuronal dimensionality, dynamics and
measurement. *bioRxiv*. https://doi.org/10.1101/214262

Gatys, L. A., Ecker, A. S., & Bethge, M. (2016). Image style transfer using
convolutional neural networks. *IEEE Conference on Computer Vision and Pattern
Recognition*, 2414–2423. https://doi.org/10.1109/CVPR.2016.265

Geirhos, R., Rubisch, P., Michaelis, C., Bethge, M., Wichmann, F. A., & Brendel,
W. (2018a). ImageNet-trained CNNs are biased towards texture; increasing shape
bias improves accuracy and robustness. *International Conference on Learning
Representations* (ICLR 2019). https://openreview.net/forum?id=Bygh9j09KX

Geirhos, R., Temme, C. R. M., Rauber, J., Schütt, H. H., Bethge, M., &
Wichmann, F. A. (2018b). Generalisation in humans and deep neural networks.
*Advances in Neural Information Processing Systems*, 31.

Geirhos, R., Meding, K., & Wichmann, F. A. (2020). Beyond accuracy: Quantifying
trial-by-trial behaviour of CNNs and humans by measuring error consistency.
*Advances in Neural Information Processing Systems*, 33.
https://doi.org/10.48550/arXiv.2006.16736

Geirhos, R., Narayanappa, K., Mitzkus, B., Thieringer, T., Bethge, M.,
Wichmann, F. A., & Brendel, W. (2021). Partial success in closing the gap
between human and machine vision. *Advances in Neural Information Processing
Systems*, 34, 23885–23899. https://doi.org/10.48550/arXiv.2106.07411

Goodale, M. A., & Milner, A. D. (1992). Separate visual pathways for perception
and action. *Trends in Neurosciences*, 15(1), 20–25.
https://doi.org/10.1016/0166-2236(92)90344-8

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image
recognition. *IEEE Conference on Computer Vision and Pattern Recognition*,
770–778. https://doi.org/10.1109/CVPR.2016.90

Hermann, K., Chen, T., & Kornblith, S. (2020). The origins and prevalence of
texture bias in convolutional neural networks. *Advances in Neural Information
Processing Systems*, 33. https://doi.org/10.48550/arXiv.1911.09071

Hubel, D. H., & Wiesel, T. N. (1968). Receptive fields and functional
architecture of monkey striate cortex. *The Journal of Physiology*, 195(1),
215–243. https://doi.org/10.1113/jphysiol.1968.sp008455

Jagadeesh, A. V., & Gardner, J. L. (2022). Texture-like representation of objects
in human visual cortex. *Proceedings of the National Academy of Sciences*, 119,
e2115302119. https://doi.org/10.1073/pnas.2115302119

Jagadeesh, A. V., & Livingstone, M. S. (2024). Texture bias in primate ventral
visual cortex. *ICLR 2024 Workshop on Representational Alignment*.
https://openreview.net/forum?id=D6YvyvZ7P7

Jang, H., & Tong, F. (2021). Convolutional neural networks trained with a
developmental sequence of blurry to clear images reveal core differences between
face and object processing. *Journal of Vision*, 21(12), 6.
https://doi.org/10.1167/jov.21.12.6

Kaplan, E., & Shapley, R. M. (1982). X and Y cells in the lateral geniculate
nucleus of macaque monkeys. *The Journal of Physiology*, 330(1), 125–143.
https://doi.org/10.1113/jphysiol.1982.sp014333

Kingma, D. P., & Ba, J. (2017). Adam: A method for stochastic optimization.
*arXiv*. https://doi.org/10.48550/arXiv.1412.6980

Kobatake, E., & Tanaka, K. (1994). Neuronal selectivities to complex object
features in the ventral visual pathway of the macaque cerebral cortex. *Journal
of Neurophysiology*, 71(3), 856–867. https://doi.org/10.1152/jn.1994.71.3.856

Komatsu, H., & Goda, N. (2018). Neural mechanisms of material perception: Quest
on Shitsukan. *Neuroscience*, 392, 329–347.
https://doi.org/10.1016/j.neuroscience.2018.09.001

Kriegeskorte, N., Mur, M., & Bandettini, P. A. (2008). Representational
similarity analysis — connecting the branches of systems neuroscience.
*Frontiers in Systems Neuroscience*, 2, 4.
https://doi.org/10.3389/neuro.06.004.2008

Lakens, D. (2017). Equivalence tests: A practical primer for t tests,
correlations, and meta-analyses. *Social Psychological and Personality Science*,
8(4), 355–362. https://doi.org/10.1177/1948550617697177

Landau, B., Smith, L. B., & Jones, S. S. (1988). The importance of shape in early
lexical learning. *Cognitive Development*, 3(3), 299–321.
https://doi.org/10.1016/0885-2014(88)90014-7

Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár,
P., & Zitnick, C. L. (2014). Microsoft COCO: Common objects in context.
*European Conference on Computer Vision*, 740–755.
https://doi.org/10.1007/978-3-319-10602-1_48

Linsley, D., Feng, P., & Serre, T. (2025). Better artificial intelligence does
not mean better models of biology. *arXiv*.
https://doi.org/10.48550/arXiv.2504.16940

Livingstone, M. S., & Hubel, D. H. (1987). Psychophysical evidence for separate
channels for the perception of form, color, movement, and depth. *Journal of
Neuroscience*, 7(11), 3416–3468.
https://doi.org/10.1523/JNEUROSCI.07-11-03416.1987

Lu, Z., Thorat, S., Cichy, R. M., & Kietzmann, T. C. (2026). Adopting a human
developmental visual diet yields robust and shape-based AI vision. *Nature
Machine Intelligence*, 8(5), 735–748.
https://doi.org/10.1038/s42256-026-01228-6

Masri, R. A., Grünert, U., & Martin, P. R. (2020). Analysis of parvocellular and
magnocellular visual pathways in human retina. *Journal of Neuroscience*, 40(42),
8132–8148. https://doi.org/10.1523/JNEUROSCI.1671-20.2020

Mehrer, J., Spoerer, C. J., Jones, E. C., Kriegeskorte, N., & Kietzmann, T. C.
(2021). An ecologically motivated image dataset for deep learning yields better
models of human vision. *Proceedings of the National Academy of Sciences*,
118(8), e2011417118. https://doi.org/10.1073/pnas.2011417118

Mishkin, M., Ungerleider, L. G., & Macko, K. A. (1983). Object vision and spatial
vision: Two cortical pathways. *Trends in Neurosciences*, 6, 414–417.
https://doi.org/10.1016/0166-2236(83)90190-X

Nassi, J. J., & Callaway, E. M. (2009). Parallel processing strategies of the
primate visual system. *Nature Reviews Neuroscience*, 10(5), 360–372.
https://doi.org/10.1038/nrn2619

Okazawa, G., Tajima, S., & Komatsu, H. (2015). Image statistics underlying
natural texture selectivity of neurons in macaque V4. *Proceedings of the
National Academy of Sciences*, 112(4), E351–E360.
https://doi.org/10.1073/pnas.1415146112

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen,
T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Köpf, A., Yang, E.,
DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L.,
Bai, J., & Chintala, S. (2019). PyTorch: An imperative style, high-performance
deep learning library. *arXiv*. https://doi.org/10.48550/arXiv.1912.01703

Prince, J. S., Charest, I., Kurzawski, J. W., Pyles, J. A., Tarr, M. J., & Kay,
K. N. (2022). Improving the accuracy of single-trial fMRI response estimates
using GLMsingle. *eLife*, 11, e77599. https://doi.org/10.7554/eLife.77599

Schuirmann, D. J. (1987). A comparison of the two one-sided tests procedure and
the power approach for assessing the equivalence of average bioavailability.
*Journal of Pharmacokinetics and Biopharmaceutics*, 15(6), 657–680.
https://doi.org/10.1007/BF01068419

Sincich, L. C., Park, K. F., Wohlgemuth, M. J., & Horton, J. C. (2004). Bypassing
V1: A direct geniculate input to area MT. *Nature Neuroscience*, 7(10),
1123–1128. https://doi.org/10.1038/nn1318

Skottun, B. C., & Skoyles, J. R. (2011). On identifying magnocellular and
parvocellular responses on the basis of contrast-response functions.
*Schizophrenia Bulletin*, 37(1), 23–26. https://doi.org/10.1093/schbul/sbq114

Smith, A. T., Singh, K. D., Williams, A. L., & Greenlee, M. W. (2001).
Estimating receptive field size from fMRI data in human striate and extrastriate
visual cortex. *Cerebral Cortex*, 11(12), 1182–1190.
https://doi.org/10.1093/cercor/11.12.1182

Tanaka, K. (1996). Inferotemporal cortex and object vision. *Annual Review of
Neuroscience*, 19, 109–139.
https://doi.org/10.1146/annurev.ne.19.030196.000545

Thorat, S., & Kietzmann, T. C. (2024). MiniEcoset.
https://doi.org/10.17605/OSF.IO/MSNA2

Wandell, B. A., Dumoulin, S. O., & Brewer, A. A. (2007). Visual field maps in
human cortex. *Neuron*, 56(2), 366–383.
https://doi.org/10.1016/j.neuron.2007.10.012

Wang, L., Mruczek, R. E. B., Arcaro, M. J., & Kastner, S. (2015). Probabilistic
maps of visual topography in human cortex. *Cerebral Cortex*, 25(10), 3911–3931.
https://doi.org/10.1093/cercor/bhu277

Yamins, D. L. K., & DiCarlo, J. J. (2016). Using goal-driven deep learning models
to understand sensory cortex. *Nature Neuroscience*, 19(3), 356–365.
https://doi.org/10.1038/nn.4244

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
   alone was trained with `RandomGrayscale(p_aug=0.5)`.
14. **The readout weight is a hyperparameter, and nothing selects its value.**
   The integration in §2.3 is a linear sum of the two streams' features at a
   fixed w = 0.5 (§4.3). §2.3 sweeps w and shows shape bias spans 0.40–0.90 over
   frozen components, which is the readout account's central dependence measured
   rather than argued — but it cuts both ways: **the model's headline 0.675 is a
   consequence of an arbitrary hyperparameter**, not a property the system
   arrived at. No w was fitted to human data and nothing here estimates the
   weighting a human readout would use. The claim is that *a* mixing readout over
   oppositely biased components yields shape-biased behaviour, not that this is
   the human mixture.
15. **The species scope of the motivating premise is mixed.** One human study
   (Jagadeesh & Gardner, 2022) and a macaque literature (Okazawa et al., 2015;
   Jagadeesh & Livingstone, 2024; Komatsu & Goda, 2018) motivate the paradox.
   Our cortical measurements are human (NSD), our behavioural reference is
   human, and our model is not a model of macaque; but the premise is not
   carried by human data alone, and §1 states this rather than leaving it to be
   discovered.
16. **The single-stream control is not parameter-matched.** It has half the
   parameters of the two-stream model (§2.3). The confound runs against the
   accuracy result and in favour of the shape-bias result; the pre-registered
   `resnet34` matched arm has not been run (PLAN §72.6).


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
  not: pooled VO is **+0.0045 (p = .46)**, bounded by equivalence to **within
  ±0.017** (90% CI [−0.0082, +0.0171], TOST p = .0297; percentile bootstrap
  [−0.0065, +0.0142], 20k resamples) on the
  same 20-ROI profile, and hV4's parvo preference (−0.012, 2/8) is weak. This is
  a failed prediction and we report it as one. **We draw no inference from the
  sign of that +0.0045.** An earlier version of this entry read it as "if
  anything magno-preferring"; at p = .46 that was an interpretation a null does
  not support, and it is withdrawn. The quantity is bounded near zero and that
  is all we claim for it. *The equivalence result should not
  be over-read:* at n = 8 the effect detectable at 80% power is 0.0220, slightly
  **above** the ±0.0195 interval declared for this test, so equivalence is
  achieved because the point estimate sits near zero rather than because the
  design is highly sensitive (PLAN §78.3). It is also what makes the konio
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


---

## Supplementary Note S1 — corrections to our own earlier analyses

Every number in this paper has at some point been computed a different way, and
five passages of the Results previously carried the full narrative of those
revisions. They are project history rather than results, so they are collected
here; the substantive consequence of each remains in the main text at the point
where it bears on a claim. Nothing is withdrawn by being moved. The single cause
behind four of the five is the same: results were originally scored on the final
training checkpoint, and the diet arm — uniquely — degrades after its per-epoch
peak, so the final checkpoint measured degraded weights (PLAN §67, §68, §71).

### S1.1 — the 28-fold comparison (§2.2)

checkpoints. (An earlier version of this sentence gave 28-fold, against a §2.5
seed range of 0.904 that was itself a final-checkpoint artefact; see §2.5 and
PLAN §71.10. The cortical instrument is still the tighter of the two, but not by
the margin previously claimed, and we no longer attribute §2.5's spread to the
participation-ratio measure — it belonged to the checkpoint.) The
raw statistic

### S1.2 — which instruments are seed-stable (§2.4)

*A note on which instruments are seed-stable.* The conjunction cell's probe bias
varies by **0.014** across seeds — the tightest number in this paper. The same
three checkpoints vary by **0.120** on the magno/parvo dimensionality ratio of
§2.5 when both are scored at the best saved checkpoint. §2.5 still claims a side
rather than a level, but because its within-arm spread exceeds its between-arm
gap (§2.5), not because its instrument is unstable — an earlier version of this
note gave that spread as 0.904 and drew the wrong conclusion from it
(PLAN §71.10).

### S1.3 — the seed-spread correction (§2.5)

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

### S1.4 — checkpoint policy on the OOD table (§2.6.2)

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

### S1.5 — how the models were selected, and the 37% / 16.5% / 20.7% history (§2.6.2)

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


---

## Supplementary Note S2 — the koniocellular control

Reported here rather than in §2.2 because it is a control on a control: it tests
whether the *chromatic axis* carried by stream 2 is what delivers the ventral end
of the gradient, and it constrains a claim the paper does not make (see *Claims
deliberately not made*: that the chromatic stream predicts colour-biased cortex).
Methods in §4.5.

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
      **Re-run against the reframed draft** (`docs/NHB_REVIEW_2026-09-06_v2.md`,
      commit `d38dbdb`) returns **Major revision, 6.4/10**, up from
      reject-with-transfer at 6.0 — Relevance 1.7 → 2.3, and all three
      personas now send it out. Decomposed: the reframe contributes +0.24,
      the §77–§78 equivalence tests +0.18. Two hard overrides cap it at
      Major revision and **both are clerical** — undefined `±` and the
      missing ethics statements — so neither needs new data or compute.
- [x] **Blockers carried over from the referee simulation** — **all four
      cleared 2026-09-06.** References section written (52 entries, every one
      verified against the source rather than transcribed from the BA); `±`
      defined in §4.7 as the across-seed range, with the two labelled
      exceptions; the 36 p-values split into three declared families with the
      correction stated for each (§4.7); ethics, constraints-on-generality,
      data, code, materials and competing-interests statements added as a
      *Reporting and availability* section. The Geirhos licence remains the one
      open sub-item and is flagged in the materials statement itself. **The equivalence gap is closed** — a parallel session ran the
      TOST intervals (PLAN §77–§78, `results/equivalence/`) and all four nulls
      clear a pre-fixed bound; the achieved bounds are now quoted in §2.6.2,
      §2.6.3 and the VO entry, with the two caveats §78.3 requires. See `docs/NHB_REVIEW_2026-09-06.md` §"Specific
      Improvement Suggestions" for the ranked list.
- [x] **Cite the paradox's second horn — DONE 2026-09-06, and the check paid
      off.** The BA cited three sources for the texture-like premise
      (Jagadeesh & Livingstone 2024; Komatsu & Goda 2018; Okazawa et al. 2015)
      and **all three are macaque**, while the draft claimed "regions of the
      *human* ventral stream". Jagadeesh & Livingstone 2024 is additionally an
      **ICLR workshop paper**, not a journal article. The human result the
      premise actually needs is **Jagadeesh & Gardner (2022), PNAS 119,
      e2115302119** — human fMRI, category-selective cortex, feature-matched
      scrambles — which the BA never cited. The abstract and §1 now cite it as
      the human evidence, keep the macaque work as convergent support, and
      state the species scope explicitly (Limitation 15).
      *Superseded description of the original problem:*
      The claim that human ventral-stream representations are texture-like
      appears **eight times** ("have been described as", "well described as")
      with **no author, no year, no study anywhere in the manuscript**. Under
      the first framing this was one gap in a reference-free draft; under the
      reframe it is the load-bearing premise of the title, abstract,
      introduction and discussion. Name the studies, record region / species /
      stimulus class for each, and narrow the abstract's phrasing to what they
      actually support — the paradox survives being narrowed, not being
      unsourced. Jagadeesh & Livingstone is cited by the BA as 2024 and its
      year/venue is still unverified (v2 referee sim, M1).
- [x] **Fusion-weight sweep DONE 2026-09-06** (array 26439369, 6/6, `main`, no
      GPU; `tmp_analysis/fusion_sweep.py`, `results/fusion_sweep/`). 21 values of
      w × 3 seeds × 2 checkpoint policies. Shape bias runs **0.419 → 0.879**
      (final) and **0.399 → 0.904** (best saved) with both components frozen,
      monotone in 3/3 seeds at the best checkpoint. w = 0.5 reproduces the
      published per-seed values exactly, which is the anchor. §2.3 rewritten
      around it and its overclaim removed. *Superseded description:*
      §2.3 is now the keystone, and `full` is a **fixed 50/50 logit-level sum**:
      `stream1_weight = 0.5`, a hyperparameter that is never learned
      (`src/models/pytorch_models.py:32,46`;
      `src/analysis/behavior/cue_conflict.py:126-145`), with
      `magno_only`/`parvo_only` the same sum with one term zeroed. Two
      consequences. (a) **Methods §4.3 must state this** — the section is not
      reproducible as written; it never mentions the fusion weight at all.
      (b) The prose "a readout that inherited would sit at 0.878; one reading
      the texture-biased stream would sit near 0.418; it does neither" names
      w=1 and w=0, which this architecture cannot produce — with w fixed at 0.5
      an intermediate value is close to arithmetically forced, and 0.675 is a
      deterministic function of the two streams' logits. **This is the same
      defect class as §2.6.1's ×1.00 identity cells, which the reframe just
      repaired.** The fix is cheap: `w` is a scalar in the forward pass and
      `cue_conflict.py` already reads it, so evaluating cue-conflict shape bias
      at w ∈ {0, 0.1, …, 1.0} on existing checkpoints is CPU-only minutes on
      `--partition=main`. That instantiates the Discussion's own stated "natural
      next test" — *manipulating the weighting should move the rate without
      moving any component* — as a **dose-response curve**, converting §2.3 from
      an arithmetic demonstration into a measurement. A `dkl_bcs_mheavy`
      condition trained at w=0.9 is already registered in
      `src/training/standalone_trainer.py:58`, giving a second arm where the
      weighting differs at *training* time (v2 referee sim, M3).
- [x] **Carry Limitation 9's role-vs-content caveat into the abstract and §1**
      (2026-09-06) — the abstract now closes on it explicitly.
      The model's texture-biased component is chromatic (L−M); the human
      ventral texture finding is not about chromaticity. The concession exists
      only in Limitations while the title and abstract present the resolution
      without it (v2 referee sim, M2).
- [ ] **Length — still the largest open item, and it got worse before it got
      better.** Main text is **~9,930 words** against NHB's ≈4,000–6,000, and
      **13 display items** against ≤8. Two rounds of trimming moved ~800 words
      out (five self-correction narratives → Supplementary Note S1; the
      koniocellular control → S2) and the abstract went 636 → 400 → **229**, but
      the required *additions* outweighed them: the single-stream control, the
      premise's citations and species-scope paragraph, the corrected §4.7, and
      §2.3's sweep. Nothing further can be cut without an editorial pass that
      decides what leaves the main text. The candidates, largest first: §2.2
      (1,556 → the six front-end controls could go to Extended Data, keeping the
      gradient table); §2.5 (1,291 → the trajectory, checkpoint-invariance and
      provenance paragraphs); §2.6.2 (859 → the corruption table could be a
      figure). Converting §2.1, §2.3, §2.6.2 and §2.6.3 from tables to panels
      addresses the word count and the display-item cap together, and is the
      referee simulation's ranked suggestion 7.
- [ ] **Resolve the provenance of §2.2(a)'s raw statistic before submission.**
      A parallel session cannot reproduce the no-diet raw value (−0.033): their
      aggregation, median VSB over NC>0.1 voxels, returns **+0.104** for that
      cell while reproducing the *normalised* +1.534 as +1.53. The two candidate
      readings **disagree in sign**, and PLAN §73 already found "the mislabel,
      the raw/_raw trap" in this exact measure, so this is not a case where
      "probably the wrong mask" is good enough. The interpretive clause that
      depended on the sign has been cut above, so nothing in the manuscript now
      rests on it — but the cell itself still needs certifying.
      *Two concrete leads, from the code.* (i) **The likely definition:** `norm`
      is `vsb = (η²_shape − η²_texture)/(η²_shape + η²_texture)`
      (`voxel_shape_bias.py:69`) and `raw` is almost certainly the
      **un-normalised numerator** η²_shape − η²_texture. The arithmetic fits:
      1.534 × a typical denominator of 0.40 = 0.614 against a printed 0.616. A
      norm-positive/raw-negative sign flip is then possible whenever the magno
      band carries the larger denominator, which per-band normalisation permits.
      Both η² fields are stored per band, so this is directly checkable. (ii)
      **A live mislabel hazard:** in `tmp_analysis/cc_ladder.py` the key literally
      named `"raw"` is **not** the raw stream dissociation — it is
      `step(cc_joint_vsb)`, the hV4−V1 profile step on the *joint* band
      (lines 52–66), while the dissociation is the separate key `"mp"`. Anyone
      reading `"raw"` from that script and pairing it with `"mp"` pairs two
      different quantities. That is exactly the §73 trap and it is still in the
      tree.
- [ ] **Recovery analysis: pilot in, production pending — DO NOT paste pilot
      numbers into the text.** The parallel session's pilot (PLAN §82, commit
      db09b1e; production array 26439650 queued, 8 subjects, `main`, no GPU)
      returns what §2.2 needs: condition A recovers the true ordering exactly
      (ρ = +1.000 against a true +1.000, errors ≤0.025), and condition B — every
      ROI given an identical generative process, differing only in noise ceiling
      — produces **no** gradient (ρ = −0.321, ns, wrong sign). The pipeline does
      not manufacture a gradient from correlated bands plus varying SNR. Place
      this in §2.2 or §4.5 **only on the production numbers** (1000 components,
      200 voxels, 8 subjects vs the pilot's 60/20/1).
      *Three things to carry when it lands.* (i) **A null bias toward magno**
      where the true difference is near zero — direction trusted, magnitude not
      until production. It does not touch the gradient (a large effect in the
      accurate regime, with B ruling out the SNR mechanism directly) but it does
      bear on the VO null, whose sign claim has already been withdrawn above on
      independent grounds. A magno-directed bias makes a null *more*
      conservative, so nothing about the VO null as a null is at risk; what it
      forbids is reading its small positive value as real, which the draft no
      longer does. (ii) A possible **discretisation artefact** — null estimates
      clustering near two values rather than scattering — which if it survives
      at production settings means neighbouring ROIs can differ by ~0.02 for
      purely numerical reasons and needs a sentence. (iii) The **identifiability
      sentence for §4.5**: smallest canonical correlation ≈0 and largest ≈0.81 at
      pilot scale, i.e. unique variance *is* identifiable in these feature
      spaces. Phrasing agreed: identifiable-in-these-spaces is what the spectrum
      supports; recovered-correctly is what the recovery analysis supports — two
      sentences, two claims, different evidence, the first placed before the
      gradient.
- [ ] **Two cortical-instrument analyses claimed by a parallel session**
      (2026-09-06, from the NN referee sim). Do not duplicate; apply their
      output when it lands. (a) **Uₘ/Uₚ recovery analysis** — simulate voxels
      with a known unique-variance split at the measured SNR, ROI size and
      inter-band correlation and show `variance_partition` recovers both the
      split and the gradient rather than manufacturing one from correlated
      bands. `banded_ridge.py:208-223` is pure subtraction
      (`unique_m = r2_joint − r2_parvo`) and its own docstring warns that
      correlated bands make unique components understate a band's contribution,
      which is exactly the regime at issue. This is the instrument all of §2.2
      rests on and it is currently unvalidated. (b) **Permutation null over
      hierarchical-rank labels** across the 20 Kastner ROIs, to sit *beside* the
      family-size disclosure now written into §2.2(b) — **not to replace it**. A
      null tests whether ρ exceeds random labelling; it cannot undo post-hoc
      selection, so the honest report is "survivor of seven, and here is how far
      it exceeds chance", never the null alone.
- [x] **`±` estimator unified 2026-09-06** (cell by cell, not by a blanket
      factor). Three groups existed, not two: two cells of §2.5's "after
      training" column were already ddof=1 (`dkl_bcs_seed0` ±0.059,
      `dkl_bcs_swap_seed0` ±0.085) because they were re-scored at best-saved
      checkpoints through `pr_seeds.py:118` rather than `pr_table.py:41`;
      rescaling them would have printed 0.063 and 0.091. All 17 cells were
      re-derived and each reproduced its printed value before replacement, so
      the cache mapping is confirmed for every one. 15 restated, 2 left. No mean,
      ordering, separation or claim moved. Computed by a parallel session
      (PLAN §81, `results/pr_ddof1/pr_spi_ddof1_2026-09-06.txt`) and applied
      here verbatim. **If any of the four 2-dp cells in §2.6.1 is ever raised to
      3 dp, regenerate from that module rather than hand-editing** — they are
      unchanged only because they round the same, not because their underlying
      value held.
- [x] **§2.6.1's `±` column reprinted at 3 dp** (2026-09-06), all eight cells,
      from the parallel session's module at 6 decimals. This **resolves** the
      knife edge rather than disclosing it: 0.094891 now prints 0.095
      unambiguously. Each of the four newly supplied values reproduces its
      previously committed 2-dp form (0.14/0.13/0.17/0.13), so nothing already
      pasted was wrong — this only adds precision. **Means deliberately left at
      2 dp**: `swap`/Gabor is 3.3665, exactly on the 3-dp boundary, rendering
      3.366 under round-half-even and 3.367 under round-half-up, so raising it
      would encode a rounding convention rather than a measurement (PLAN §81).

- [ ] **Figures.** Only 2 exist (`fig1_convergence`, `fig2_ventral_gradient`)
      and the text has 2 figure callouts across seven result sections. Needed:
      §2.1's two routes, §2.3's **w-sweep curve** (the data are in
      `results/fusion_sweep/`, ready to plot), §2.4's 2×2, §2.6.2's degradation
      curves, §2.6.3's shape-bias-against-κ scatter.
- [x] **Methods expanded** (2026-09-03) — §4.1–4.7, every parameter read from
      the code and run scripts rather than from memory; 3 seeds per condition
      stated, and the subject vs seed inferential axes separated in §4.7.
- [ ] Supplementary: the VSB cortical profile with its negatives intact; the
      capacity ladder (PLAN §42) with its non-monotonicity caveat.
- [x] **Cortical seed replication complete** (2026-09-04) — array 26416428,
      32/32 COMPLETED. §2.2 replicates across three seeds with both statistics
      agreeing (diet seed range 0.032). **The paper now has no n=1 claim in its
      main results** (PLAN §66).
