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

Shape bias — how often a model names an object's shape rather than its texture
when the two disagree — is the most-cited measure of human alignment for vision
models, and a model that matches the human rate is routinely described as
having human-like representations. We show that the rate is a free parameter of
the readout. In a two-stream network, reweighting a single scalar over frozen
components moves shape bias from 0.40 to 0.90 — the range the field's models
occupy — while neither representation changes. Because the head is affine this
is exact interpolation between two measured endpoints rather than an empirical
trend, and a nonlinear head refit on the same frozen features preserves 86% of
the span, agreeing with a refitted linear head to within 1%.

What the metric then fails to track is measurable. Across 189 readouts with the
mixing weight optimised post hoc, trial-by-trial agreement with ten human
observers never reaches 21% of the human–human ceiling, and the weight that
maximises agreement is neither stable across runs nor the one that matches the
human rate. Shape bias likewise comes apart from effective dimensionality and
from out-of-distribution robustness.

The same mechanism dissolves a standing tension. Humans name shape, yet
representations along the ventral stream — the pathway most identified with
object recognition — are well described by texture statistics (Jagadeesh &
Gardner, 2022; Okazawa et al., 2015). In a network receiving approximations of
the magnocellular and parvocellular signals reaching cortex, the streams end
training with opposite shape–texture biases by different routes: the luminance
stream's shape bias is supplied by its input statistics, the chromatic stream's
texture bias is built by learning. Fitted to eight subjects of the Natural
Scenes Dataset, it is the *texture-biased* stream whose unique variance carries
the ventral end of the hierarchy, while behaviour, reading out from both, is
shape-biased. Shape-biased behaviour therefore need not imply shape-biased
representations: a behavioural rate is a property of a readout over components,
not a report of any one of them. The correspondence we draw to cortex is one of
explanatory role, not of content.

---

## 1. Introduction

Humans presented with an image whose shape and texture point to different
categories name the shape. Standard convolutional networks name the texture.
That dissociation, measured on a cue-conflict set that ten human observers also
viewed (Geirhos et al., 2018a), is the most conspicuous behavioural gap between
human and machine object recognition, and closing it has become a standard
target for biologically motivated models (Hermann et al., 2020; Jang & Tong,
2021; Lu et al., 2026).

In the process the measurement has acquired a second life. **Shape bias is
measured behaviourally and interpreted representationally**: a model whose rate
approaches the human one is routinely described as having more human-like
representations, and the rate has become the field's most-cited single number
for human alignment in vision. Whether a behavioural rate licenses a
representational description is exactly what is at stake here, and this paper's
answer is that it does not. The rate is a free parameter of the readout. In a
system holding a shape-biased and a texture-biased component, one scalar over
frozen representations moves it across the whole range the field's models
occupy (§2.3); the readout that best matches human trial-by-trial behaviour is
neither the one that matches the human rate nor stable across runs (§2.6.2);
and the rate travels with neither effective dimensionality nor
out-of-distribution robustness (§2.6).

That conclusion is reached through a standing tension in the literature, which
the same mechanism resolves. A second body of evidence points the other way
from the behavioural gap: along the ventral stream, object representations are
well described by texture statistics. In humans,
category-selective cortex fails to discriminate natural images from
feature-matched scrambles that destroy spatial arrangement while preserving
texture statistics, yet still discriminates between categories (Jagadeesh &
Gardner, 2022). In macaque, V4 responses to natural textures are predicted by
their image statistics (Okazawa et al., 2015), and in IT both shape and texture
are represented, texture the more accurately decodable of the two (Jagadeesh &
Livingstone, 2024); material and surface appearance recruit an extended ventral
network (Komatsu & Goda, 2018).

*We state the scope of this premise precisely, because the paper rests on it.*
The human evidence is one study using one image-synthesis manipulation; the
remainder is macaque, and "texture-like" means that a texture-statistics
description suffices for the measured responses, not that shape is absent. Our
argument needs only that the ventral description is *substantially*
texture-like.

Both findings are robust and appear to be in tension. If behaviour is
shape-biased and the ventral stream texture-biased, then either the two measures
disagree about what shape bias is, or **behaviour is not a readout of one
representation.** This paper takes the second horn and shows it suffices: a
system containing a shape-biased and a texture-biased component, whose behaviour
reads out from both, will look shape-biased behaviourally and texture-biased
wherever the texture-biased component dominates — and in a model of the
subcortical split, the component that dominates the ventral end is the
texture-biased one.

Two facts about the visual system motivate the model we use.

**(i) The primate visual system does not receive one image.** It receives a
luminance-dominated, temporally fast, spatially coarse magnocellular signal and
a chromatically opponent, spatially finer parvocellular one, separated before
cortex. Models of ventral-stream computation almost universally discard this,
taking RGB pixels as input.

**(ii) That input is not stationary over development.** Infant acuity and
contrast sensitivity are poor and improve over the first years; the visual
system learns first from a low-passed world. A *developmental visual diet* (DVD;
Lu et al., 2026) implements this as a transform on the achromatic channel. The
term and the construct are theirs; §4.1 states the parameterisation we use, which
evaluates the age-dependent contrast-sensitivity cut at a fixed 48 months rather
than scheduling it over training.



We build such a system and show it does both at once. Its two streams end
training with opposite shape–texture biases by different routes, in an
architecture whose trunks are bit-identical at initialisation, so every
divergence between them is attributable to input by construction (§2.1). Fitted
to eight NSD subjects, it is the *texture-biased* stream whose unique variance
carries the ventral end of the hierarchy (§2.2), while behaviour, reading out
from both, is shape-biased (§2.3). We then locate what produces the shape-biased
component — an achromatic carrier × developmental transform interaction (§2.4) —
and what preserves it through training (§2.5). Finally, over identical inputs,
shape bias separates from effective dimensionality, from out-of-distribution
robustness and from trial-by-trial agreement with human observers (§2.6):
properties the field treats as a package do not travel together, which is what a
readout account predicts. §2.7 states what the account costs.
---

## 2. Results

### 2.1 The two streams acquire opposite biases, by different routes

*Source: PAPER_PLAN §28.1, §28.5/§35, §12.*

The model splits its input once, before any convolution: stream 1 receives the
luminance channel carrying the developmental transform, stream 2 the
untransformed L-M chromatic channel (Methods §4.1). Read out separately with a
linear probe, the two streams end training on opposite sides of the
shape/texture boundary, and the same holds behaviourally (**Figure 1a**).

**And the decomposition is asymmetric in how it arises** — scoring the same
probe on an untrained network with the same front end separates the two routes
(Table 1; **Extended Data Fig. 1a**).

**Table 1. The two streams end on opposite sides of the boundary, by different
routes.**

| | magno stream (lum + transform) | parvo stream (L−M) | integrated |
|---|---|---|---|
| linear probe, shape−texture bias (0 = sign flip) ᵃ | **+0.819** | **−0.518** | — |
| cue-conflict shape bias (0.5 = parity) ᵃ | **0.878** | **0.418** | **0.675** |
| linear probe, untrained ᵇ | **+0.796** | −0.087 | — |
| linear probe, trained ᵇ | **+0.810** | **−0.482** | — |
| *learning contributes* ᵇ | **+0.014** | **−0.395** | — |

ᵃ Means of 3 training seeds. The probe classifies every image; the cue-conflict
denominators are reported per cell in §2.4. ᵇ Single model (seed 0); the
untrained references carry no training seed, so the route asymmetry is reported
at one seed per cell and should be read that way. It is large relative to the
trained arm's 3-seed range of 0.014 (Extended Data §E1).

The magno stream's shape bias is essentially complete before any weight is
updated -- the low-pass and contrast-sensitivity transform does it -- while the
parvo stream begins barely biased and acquires its texture preference over
training. **The shape-biased component is supplied by the input; the
texture-biased component is built by learning.** We state this plainly rather
than let it be inferred: nothing in this paper claims that learning creates the
shape-biased component, and §2.4's 2x2 is a decomposition of the *input*, not
of what training adds.

**Ordinary training destroys the bias the input supplies; the transform is what
prevents that.** Removing the transform, the untrained magno bias falls from
+0.796 to +0.417 -- and training then drives it to **-0.219**. The unblurred
luminance stream does not merely fail to become shape-biased; it becomes
*texture*-biased. §2.5 finds the same pattern on an independent instrument,
where diet-trained networks end magno-dominant from front ends spanning 3.26x
while every no-diet network collapses to parity.

**Every divergence between the streams is attributable to their inputs, by
construction.** `TwoStreamNetwork` initialises the two trunks bit-identically —
0 of 120 stream-1/stream-2 tensor pairs differ, in every condition — and the
trained `rgb_tsn` control, both streams fed the same image, returns all 120
pairs still bit-identical after 150 epochs and a stream difference of exactly
**+0.0000** on every fold of every seed. This is an algebraic identity and we
report it as one: no stream difference reported anywhere in this paper can be a
lucky draw in one stream's weights, which is why §2.4's front-end 2×2 can carry
causal weight. It is *not* evidence that a two-stream architecture fails to
specialise spontaneously — a hypothesis this architecture cannot test (Extended
Data §E1, Limitations).

### 2.2 The texture-biased stream explains the ventral end of the hierarchy

*Source: PAPER_PLAN §43.1, §17, §48, §29, §66. Tables, controls and provenance:
Extended Data §E2.*

If the two streams carry opposite biases, the paradox turns on which of them
the ventral hierarchy is explained by. We fit voxelwise banded-ridge encoding
models per stream on NSD (8 subjects; Methods §4.5) and use them as measurement
instruments twice: first to confirm that the two streams' cortical mappings are
oppositely organised, then to locate where along the hierarchy each does its
explanatory work (**Figure 2**). **Figure 1b** is the same instrument's verdict
on §2.1's stream contrast, reached with no probe and no readout.

**(a) In-silico cue conflict.** The encoding models are fitted on natural
scenes only; no participant ever viewed a cue-conflict image, so passing the
1,200 cue-conflict images through the *fitted* models yields a genuine
model-based prediction — a per-voxel predicted shape-bias index. The
magno−parvo difference in that index is **+1.534 with the diet and +0.110
without** (n = 8, p = .0078, 8/8), and the raw statistic agrees (**+0.616
against −0.033**), likewise in seeds 1 and 2. **We draw no inference from the
sign of the no-diet values**, which sit near zero with no test addressing which
side of it they fall on; the claim is that the arms separate (Extended Data
§E2). The effect is carried entirely by the magno stream (+0.791 vs **−0.638**
— without the diet the magno stream is *texture*-dominated), the parvo streams
being indistinguishable (−0.743 vs −0.748). This confirms §2.4–2.5's
representational result on independent machinery, through the brain data, with
no probe and no readout.

**(b) The ventral gradient of stream division of labour.** The same fitted
models answer a second, orthogonal question: not *what* each stream's cortical
mapping is sensitive to, but *where along the ventral hierarchy* each does its
explanatory work. Partitioning every voxel's explained variance into the part
only the magno stream accounts for and the part only the parvo stream does
gives a stream-preference profile from V1v to hV4. It **declines steeply with
the diet and much more shallowly without it** — early visual cortex relatively
more magno-explained, hV4 relatively more parvo-explained, the direction a
luminance/low-SF versus chromatic split predicts. On the contrast hV4 −
mean(V1v,V1d) of the per-ROI median (§4.7), the diet arm runs **−0.745 to
−0.834** on the normalised statistic and **−0.145 to −0.166** on the raw one
across three seeds, against **−0.049 to −0.096** and **−0.023 to −0.031**
without the diet. **Every one of those six comparisons reaches the exact
Wilcoxon floor (p = .0078) with complete seed separation** — the two-statistics
rule of §8 satisfied six times over. Monotonicity tracks it: the per-subject
Spearman ρ against hierarchical rank is **−0.964 on both statistics** with the
diet, against **−0.503 (norm) and −0.571 (raw)** without, effect sizes
differing by a factor of six (dz −6.60 vs −1.09). Per-seed table: Extended Data
§E2.2. *An earlier version gave the no-diet figure as a single −0.536, the
average of the two statistics rather than either of them; the diet value is
unaffected because norm and raw coincide there exactly (Supplementary Note
S1.8).*

**The observed gradient is the most extreme labelling of these ROIs that
exists.** The hierarchical-rank vector [1,1,2,2,3,3,4] has ties, so the seven
ROIs admit exactly **630 distinct label assignments** — the permutation null is
enumerable rather than sampled, and we enumerate it. Applying each labelling to
every subject and recomputing the mean ρ, the observed diet-arm value is **the
single most extreme of all 630**, in all three seeds and on both statistics (p
= 1/630 = **.0016**, the exact floor of the test). The no-diet arm does not
reach that floor (p = .019 to .092), and in seed 2 the two statistics disagree
(.037 norm against .092 raw), so §8 refuses a claim there — the same asymmetry
the rest of §2.2 reports. Code: `tmp_analysis/hierarchy_permutation_null.py`;
full table in Extended Data §E2.3.

**The diet makes the gradient large; it does not make it exist.** Correcting
each arm for its own measured instrument bias gives **−0.720** with the diet
and **−0.101** without, each non-zero in every subject at the exact Wilcoxon
floor, so the diet's effect is a factor of **7.1** on a gradient already
present. The comparative claim is the robust one — it holds across every
treatment of the instrument we tried, including the one we reject — and it is
what §2.4–2.5's mechanism needs. Three earlier framings are withdrawn
(Supplementary Note S1.6).

*One subject carries a third of the no-diet effect, and that is a property of
the subject, not of the fit.* We first read subject 01's steep no-diet gradient
as a provenance failure and were wrong: the anomaly is not arm-specific
(displaced from the sibling median by **−0.195 no-diet and −0.189 diet**), so
what remains is an influence result and is reported as one. Excluding it moves
the no-diet arm from −0.079 to −0.054, still non-zero in 7/7 at p = .0156. n =
8 is primary and n = 7 a leave-one-subject-out check, not the removal of a bad
fit. *These values, and the four-property replication below, supersede an
earlier table that cannot be regenerated, its no-diet figures having come from
a subject-01 fit later quarantined and deleted; Extended Data §E2.2 dates the
table, solves for the missing value against its seven siblings, and leaves two
residual gaps disclosed as provenance drift.*

**This gradient was not pre-specified: it is the survivor of a family, and
neither the null above nor the controls below touch that.** **Seven regional
hypotheses were examined on these same fitted encoding models; six failed and
are reported as failures** — a diet-specific hV4/LO step, the pre-declared
dorsal−ventral interaction on the `streams` atlas, hMT+ specificity, the
foveal/peripheral eccentricity contrast, a dorsal/ventral organisation of the
20-ROI profile, and a parvo preference in colour-biased cortex, each with its
numbers and cause of death under *Claims deliberately not made*. A permutation
test asks whether the observed labelling beats an arbitrary one, and the
controls rule out ways the gradient could be an artefact; neither can undo its
having been chosen after six siblings failed. It should be weighed as one
surviving regional result out of seven, with the failures visible, and the
strongest thing we can say for it is not a p-value but that it is the only one
of the seven that survives removal of the very front-end manipulation that
produces the apparent regional structure in the first place.

**The estimator cannot manufacture this gradient; it works against it.** The
mean noise ceiling *declines* from V1v to hV4 (ρ = −0.893), a nuisance falling
along the same axis as the effect, so we simulated it: synthetic voxels from
the fitted feature spaces, measured noise ceilings imposed, true magno−parvo
difference solved to zero everywhere. The resulting floor is **positive and
rising** with hierarchical rank (+0.051 no-diet, +0.038 diet) where the
observed effect is negative and falling — wrong sign twice over — and is
indistinguishable from zero on the raw statistic. Because the floor is a
*signed* bias the right comparison is a paired displacement rather than a
ratio: real − floor gives **−0.871** with the diet and **−0.130** without
(**−0.103** at n = 7), each non-zero in **every subject** at the exact Wilcoxon
floor for its sample size, the diet arm's **6.7× larger** (Extended Data
§E2.1).

*What this leaves.* The subtraction relocates its assumption from the contrast
to the individual ROI — weaker and more plausible, but not removed — and a
residual misfit remains that we do **not** correct for, because the correction
is under-specified: six variants of it swing the no-diet arm between 4/8 and
6/8 and between p = .055 and p = 1.00, while **the diet arm is non-zero in all
eight subjects at p = .0078 in all six**. The no-diet arm therefore rests on
the floor correction alone with real-data misfit unquantified, weaker footing
than the diet arm's; and because the simulation is not magnitude-faithful we
read −0.834 as a sign-and-ordering result rather than a calibrated effect size.
Residual profiles, the six-variant table, a tautological check we withdrew and
an earlier simulation that ran the nuisance backwards: Extended Data §E2.1.

Four properties make this more than a single fitted number, each with its
numbers in Extended Data §E2. **Training decides the direction; the input only
supplies the starting point** — an untrained network on the diet front end
already sits at −0.544 (8/8, p = .0078), the diet takes it to −0.834, and
training without it leaves nothing resolvable above the floor. **It replicates
across codebases** (−0.834 against an independent thesis checkpoint's −0.793, p
= .20). **It survives all six front-end controls on both statistics**, each at
p = .0078 in 8/8 subjects, with the qualification §2.7 places on two of them.
And **it is not a global stream preference**: the two models sit on opposite
sides of the global stream-bias axis (−0.495 against +0.849), and the strong
regional gradient belongs to the model whose global preference points the
*other* way — a model that simply preferred magno everywhere could not produce
this.

**This is the step the paradox needs, and a correlational analysis cannot take
it.** Asked the same question with representational similarity analysis, the
answer is a *global* stream preference rather than a regional split: one stream
correlates better everywhere, ventral and dorsal alike, and the correlation
cannot separate that general factor from regional structure. Partitioning
unique variance removes it.

### 2.3 Behaviour reads out from both streams, and the mixture is shape-biased

*Source: PAPER_PLAN §18.2, §52.3, §20.3, §28.1.*

The two streams are oppositely biased (§2.1) and the texture-biased one carries
the ventral end (§2.2). What remains is what the behavioural readout does with
them.

**Behaviour, against human observers.** The integrated readout's
cue-conflict shape bias is **0.675 ±0.025** with the diet against **0.416
±0.020** without, across three seeds each. The thesis checkpoint reproduces at
0.697 against its reported 0.677, inside our bootstrap CI (§18.2).

§2.6.2 qualifies everything in this section: the agreement below is one of
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

**Integration mixes rather than inherits**: the integrated readout lands
between its two components (0.878 magno, 0.418 parvo), not at the shape-biased
one.

*We are careful about what that single number can show.* `w` is a
hyperparameter fixed at 0.5, not a learned quantity (§4.3), and given two
oppositely biased components an intermediate result is close to arithmetically
forced — so **the intermediate value is not by itself a prediction that could
have failed**, and we do not present it as one.

**The readout weight moves behaviour across the entire range, and we now state
exactly how much of that is arithmetic.** Sweeping `w` moves the readout while
holding `y₁` and `y₂` exactly fixed, and **Figure 3a** shows the result:
cue-conflict shape bias at 21 values of `w`, 3 seeds, both checkpoint policies,
running from **0.399 to 0.904** at the best saved checkpoints.

The classifier is affine (§4.3) and the mixture weights sum to one, so
`clf(w·y₁ + (1−w)·y₂) = w·clf(y₁) + (1−w)·clf(y₂)` holds exactly (maximum
deviation 1.1 × 10⁻⁶ on a logit scale of 0.32). **The sweep is therefore exact
arithmetic on the two endpoint logit vectors and contains no information beyond
`w` = 0 and `w` = 1**, both already reported above; monotonicity is very nearly
analytic rather than a discovered property, and an earlier version of this
section wrongly presented it as a measurement (Supplementary Note S1.7).

**The span is not, however, an artefact of the affine head.** That the sweep is
an identity says the *affine* readout carries no extra information; it does not
say the span would vanish under a readout that is not affine. We tested this
under a rule fixed before the test was run (PLAN §104). Re-fitting a readout on
the frozen features at each of the 21 `w` — a two-layer nonlinear head, and a
refitted **linear** head as a control for the refit itself — reproduces the
sweep. The nonlinear head's median span across seeds is **0.466**, **86%** of
the affine span these same features give, against a registered survival
threshold of 80%. The decisive comparison is the one the linear control
supplies: **nonlinear and refitted-linear spans agree to within 1%** (ratio
1.01), so the span is not a property of the head's functional form at all, but
of having two oppositely biased components and any readout over them.

**Accuracy bounds where this applies, and we report the bound rather than the
headline alone.** Restricted to readouts within 5 points of peak validation
accuracy, the span narrows to **0.283** and shape bias moves from **0.421 to
0.760** across `w` = 0 to 0.75. The extreme rates near `w` = 1 are bought with
accuracy — which is §2.7's cost accounting seen from the readout side rather
than a separate finding. Full sweeps, both heads and both checkpoint policies:
Extended Data §E3.1. What
survives is the statement the argument needs: a behavioural shape-bias rate is a
property of a readout over components, not a report of either component, and a
single scalar carries the system across the range the field's models occupy
while both representations stay untouched. It does **not** establish that any
particular mixture is the one a human uses.

*Validation.* At w = 0.5 the sweep reproduces the published per-seed values
exactly (0.6922 / 0.6861 / 0.6464), the anchor showing that sweep and main
pipeline are the same computation; two caveats on the endpoints and the
decided-image count are in Extended Data §E3. **Nothing here fits `w` to
humans** — no weighting was estimated from human data, and the sweep does not
identify which mixture a human readout uses.

**A parameter-matched control withdraws the causal claim we had attached to the
split.** The objection to a two-component account is that the components may be
decorative -- that one trunk carrying both channels, with the identical diet,
would behave the same. We ran that control twice. Against a half-capacity
single stream (`dkl_bcs_single`, one ResNet-18 trunk taking both DKL channels
stacked, 11.2M parameters) the two-stream model separates completely at both
checkpoint policies, **0.655** against **0.567**. Against the pre-registered
parameter-matched arm (`dkl_bcs_single34`, one ResNet-34 trunk, 21.3M against
22.4M, same diet and schedule) it does not: **0.655** against **0.626**, with
seed ranges overlapping under both policies. Under the rule fixed before either
arm was scored -- complete separation of three seed ranges a side, and raw
ranges rather than a p-value at a 3v3 floor of .10 -- that is no separation,
and **we therefore do not claim that the split causes the shape-bias advantage
over one stream.**

The confound is directly visible rather than merely possible. The two
single-stream arms differ from one another by capacity alone -- same input,
same diet, same schedule, one trunk in both -- and they separate completely at
both policies, 0.567 against 0.626. Capacity moves shape bias by about what the
split had been credited with. We do not read the overlap the other way either:
three seeds a side cannot establish that the split buys nothing, so what we
withdraw is the attribution, not the difference.

**The accuracy half of the trade survives the matched control and grows under
it.** At matched parameters the single stream's advantage widens to **0.623**
against the two-stream **0.556**, with all three arms completely separated, and
late-training stability moves the same way (sd of validation top-1 over the
last 20 epochs: 0.015--0.041 matched, against 0.043--0.073 two-stream). The
split pays for itself in accuracy, and that half is not a capacity artefact.
One complication travels with the matched arm and is disclosed rather than
resolved: its decision count is 536.7/1200 [515, 556] against 502.0 [483, 520]
two-stream, so the rates that match to within 0.003 for the ResNet-18 control
diverge by roughly ten times that for the matched one, and its shape bias is
computed over a partly different, larger subset. Seed-level ranges, the full
table and the two scope limits: Extended Data §E3.

> magno stream alone **0.903**  >  two-stream integrated **0.655**  >
> parameter-matched single stream **0.626**  >  half-capacity single stream
> **0.567**,

which still locates the integration cost precisely -- 0.248 relative to the
best single component -- while the spacing among the lower three rungs is
within what capacity alone produces. The ladder carries a claim about the cost
of integration, not about a benefit of the split. It is also, read the other
way, a third instance of this section's thesis: shape bias moved here under a
change of trunk capacity, a manipulation no more about shape than the readout
weight was.

This is the mechanism we propose for the paradox: **shape-biased behaviour does
not require shape-biased representations**, because a behavioural rate is a
property of a readout over components. It is an existence proof with a
cortically grounded demonstration, not a measurement of human cortex, and §2.6.2
bounds how far the behavioural correspondence extends.


### 2.4–2.5 Where the shape-biased component comes from, and what keeps it

*Source: PAPER_PLAN §25, §13, §18, §21, §45, §49, §50. Full tables, the
three-seed replications and the controls: Extended Data §E4 and §E5.*

§2.1 showed the shape-biased component is supplied by the input rather than by
learning. Two subsidiary questions follow — *what about the input* supplies it,
and what stops ordinary training from destroying it — and we answer both here
because the readout account of §2.3 needs the component to exist, not because
either is the paper's claim.

**It is a conjunction, not the achromatic channel.** Decomposing the front end
into two orthogonal factors — which **carrier** a stream receives (achromatic
luminance vs chromatic L−M) and whether the **developmental transform** is
applied to it — gives four cells, two of which we built (`swap`, the transform
on the chromatic channel; `parvo_highsf`, an untransformed high-acuity
achromatic carrier). Linear-probe stream shape−texture bias, means of 3 seeds,
seed range in brackets:

| | no transform | with transform |
|---|---|---|
| **chromatic carrier (L−M)** | −0.504 [0.035] | −0.145 [0.043] |
| **achromatic carrier (lum)** | **−0.271** [0.023] | **+0.819** [0.014] |

**Both single factors move a stream substantially toward neutral and neither
crosses zero. Only the conjunction crosses, and it goes all the way** — by
**+0.936** with no seed overlap, replicated independently by behavioural
cue-conflict shape bias on the same four cells (0.364 / 0.495 / 0.396 /
**0.878**, separation +0.353). Both instruments are drawn in **Figure 1a**,
where the interaction is the non-parallelism of the two lines. This is an
interaction measured on all four cells rather than inferred from two, and it is
the strongest causal statement in the paper about *what* produces the shape
bias: the developmental low-pass transform applied specifically to an
achromatic channel. Cell-to-stream mapping, the denominator caveat on the
chromatic-transform cell and the excluded `parvo_highsf` behavioural seed:
Extended Data §E4.

*Accuracy control.* Diet-trained models are less accurate (§2.6.1, §2.7), so
the conjunction could in principle be an accuracy effect. We pre-registered the
fatal outcome before testing — *if a no-diet model at the diet's accuracy shows
a shape-biased magno stream, this paper's central claim is wrong* (PLAN
§69.3–§69.4) — retrained `dkl_base` with per-epoch checkpointing and selected
by rule the epoch nearest the diet arm's 0.556 (epoch 10 at **0.548**).
Matching accuracy closes **7.6%** of the probe gap and **7.0%** of the
behavioural gap, two instruments agreeing to within 0.6 percentage points, and
the matched model's probe bias stays *negative* — still on the texture side, in
every no-diet seed. **The conjunction effect is not an accuracy effect.** The
control rests on **one** matched run and should be read as n = 1 (Extended Data
§E4).

**What training keeps is the side of parity, not the level.** The four front
ends hand a network very different stream asymmetries before any learning — the
untrained magno/parvo dimensionality ratio spans 0.59 to 1.91 (**Figure 1c**) —
and the obvious reading, that the diet makes training *keep* the asymmetry its
input supplies, is wrong. `parvo_highsf` separates *keep* from *install*
because its front end is inverted (0.586, below parity in 8/8 subjects); we
pre-registered that a diet-trained network on it would stay inverted, and **it
does not** (0/8 after training, ending at 1.640, above a model handed 1.402).
Nor does the diet drive training to one value: the identical transform moved
onto the chromatic channel (`swap`) lands at **2.327**, differing from both
diet-on-luminance arms at p = .0078, 8/8.

> **The claim.** A developmental visual diet neither preserves the stream >
> asymmetry its front end supplies nor converges on one universal value. What >
> it fixes is **which side of parity training ends on** — every diet-trained >
> model ends magno-dominant (≥1.60) from front ends spanning **3.26×**, every >
> no-diet model collapses to parity (0.947, 0.988) from the most magno-dominant
> > front end in the set, and all **six** diet × no-diet contrasts give p =
> .0078 > in 8/8 subjects. The **level** is set by which channel the diet is
> applied to, > not by the input asymmetry.

We report no level, set point or compression factor: the within-arm seed range
reaches 0.79, larger than the 0.075 gap between the two diet-on-luminance arm
means, so "training converges to X" is unsupported at n = 3. **One disclosure
the claim depends on**: with one cached trajectory per arm, both arms sit on
the same side of parity at epochs 75–100 and the no-diet arm crosses below only
in the final quarter, while its accuracy stays flat at 0.658–0.699 — so the
effect is carried by a late change in the control arm that is not driven by
accuracy, and with one trajectory per arm we do not promote it further
(Extended Data §E5, PLAN §71.9). Two earlier readings of our own data are
retired by this, a "convergence to within 2%" and a channel-specific set point
(PLAN §58.3).

This is what rules out the filtering account. The diet is not supplying a
better input representation — the transform *removes* bandwidth rather than
adding it (Extended Data §E6.1), and the no-diet front end hands over more
asymmetry, not less. It is a constraint on the solution learning converges to.

### 2.6 Properties that do not follow from shape bias

If shape bias is a property of a readout over components rather than of a
representation, then quantities the field treats as travelling with it need not.
Over identical inputs, three do not. Two are shown here; the third — **effective
dimensionality** — comes apart in the direction opposite to intuition, and is
reported in full in Extended Data §E6.1. It tracks the bandwidth of the carrier
a stream is handed, reproduces with no network at all, and reverses when the
carrier moves: `parvo_highsf`'s second stream becomes the *more* dimensional one
(0.56) while remaining texture-biased (probe −0.266), so high effective
dimensionality does not confer shape bias.

#### 2.6.1 Shape bias does not confer out-of-distribution robustness


*Source: PAPER_PLAN §53 (pre-registered), §56. Reported in the main text, not
the supplement, because a reader who accepts §2.4 will otherwise assume the
robustness follows.*

The standard reading of Geirhos et al. is that shape bias buys out-of-
distribution robustness. We pre-registered that prediction and tested it:
corruptions applied to the image *before* the front end, scored in the models'
own 112-way label space, **3 seeds per condition** on all four cells of §2.4's
2×2, at each seed's best saved checkpoint (full table: Extended Data §E6.2).

**The diet-trained model is less robust on four of the five corruptions it was
not trained on** (**Figure 4**), with complete seed separation on all four and the same
ordering in both statistics (§8). On phase-scrambling the two arms are
indistinguishable, so we claim nothing there; the difference is bounded by
equivalence to within **±0.033**, below the smallest OOD effect this paper
claims. That is the weakest of the paper's four equivalence results and is
labelled as such — at 3 seeds per arm it rests on a Welch interval at 2.3
degrees of freedom. The two conditions without a shape bias (`swap`,
`parvo_highsf`) pattern with the no-diet model on 8 of 10 comparisons.

The **only** corruption on which the diet is more robust is low-pass — which we
declared trained-in before running, because the diet *is* a low-pass blur. The
experiment finds robustness exactly where robustness was trained in and nowhere
else, which is the strongest available evidence that the four nulls are real
rather than a broken measurement. Note that the diet is *not* more robust to
contrast reduction even though its front end includes a contrast-sensitivity
cut, so "trained-in" does not automatically confer robustness either.

**The schedule does not rescue it.** Our front end is fixed at a single
developmental age (48 months, §4.1), whereas the developmental proposal it
implements is about the *order* in which acuity and contrast sensitivity are
relaxed. A reader may reasonably object that the nulls above test a static
approximation and not the developmental trajectory. We therefore ran the same battery on an
independently trained arm that does carry the progressive age transition
(`thesis_dvd_trans`, thesis codebase). **It behaves like the static diet, not
like the no-diet control**: its relative-accuracy AUC falls inside the static
diet arm's three-seed range on five of the seven corruptions, is closer to the
diet arm on six of seven, and on the two exceptions sits *further* from the
no-diet control rather than nearer; clean accuracy likewise (0.568 against 0.553
static-diet and 0.695 no-diet), so the schedule recovers neither the robustness
nor the accuracy. This arm has one seed and one saved checkpoint, so the claim
is containment within the diet arm's observed range, not an equivalence test
(per-corruption values: Extended Data §E6.2).

**This differs from the source of the diet, and we state how rather than leave
it to be noticed.** The developmental visual diet implemented here is Lu et
al.'s (2026), whose title reports that adopting it *yields robust, shape-based
AI vision*. We reproduce the shape-based half — their released benchmark
summary gives their `resnet50-DVD-B` a mean cue-conflict shape bias of 0.822,
and our diet-trained magno stream reaches 0.878 — and we do not reproduce the
robust half. Three differences are declared in our Methods rather than
discovered here, and any of them could carry the divergence. *(i)* Our front
end is evaluated at a **fixed** 48 months rather than scheduled across training
(§4.1) — and this is the one we can rule out, because the schedule-trained arm
above behaves like the static diet, not like the no-diet control. *(ii)* Our
diet is a front-end transform on the DKL luminance channel of a **two-stream**
network whose second stream never receives it, not an augmentation applied to a
single trunk's input. *(iii)* We train on MiniEcoset at 112 classes and score
corruptions in that native label space, applied *before* the front end (§4.6),
so each model meets the corruption through its own optics; the families and
severities are our declaration. We cannot determine which difference matters
from the public release, because the benchmark data accompanying it carries
cue-conflict responses for those models and no corruption benchmarks, so their
checkpoints cannot be scored through this battery from released data alone.
What we claim is correspondingly bounded: **in this implementation, at this
fixed age, in this architecture, shape bias and robustness came apart** — not
that a developmental diet cannot deliver robustness in the setting its authors
report.

> **The claim.** Shape bias and OOD robustness are not intrinsically coupled.
> This does not contradict Geirhos et al., who showed that *Stylized-ImageNet
> training* produces models that are both shape-biased and robust. It shows the
> coupling is a property of that route, not of shape bias: a developmental route
> to comparable shape bias *in rate* (probe +0.819, behaviour 0.878 against a
> human 0.958 — though §2.6.2 shows the rate match does not extend to trial-level
> agreement)
> delivers **none** of the robustness. "Increase shape bias to get robustness"
> does not generalise across the manipulations that produce it.

#### 2.6.2 Matching the human shape-bias rate is not matching human decisions

*Source: PAPER_PLAN §57 (pre-registered), §59. In the main text, not the
supplement, because it bounds this paper's own behavioural claim.*

Two decision-makers with the same marginal rate can agree on no individual
image beyond what those rates force. Error consistency -- Cohen's kappa on
trial-by-trial agreement -- subtracts exactly that forced agreement (Geirhos et
al., 2020). Across the 10 human observers who viewed these identical stimuli
the human--human ceiling is **kappa = 0.234** (45 pairs, sd 0.095).

**We pre-registered that the diet-trained model would be more human-consistent
than its control. It is not**, on either readout. The sharpest true statement
is stronger than any single comparison: **our most shape-biased model has the
lowest error consistency of all seven.** The diet-trained magno stream carries
shape bias 0.878 -- 5x VGG-16's 0.172 -- and returns mean kappa **0.0089**,
3.8% of the ceiling, against VGG-16's 0.0170 (7.3%). Moving a model 5x up the
field's most-cited alignment metric moved it *down* on agreement with humans.
Seven-model table, pipeline validation and a self-correction: Extended Data
§E6.3.

**No readout of these components reaches human decision structure.** A
seven-model comparison varies architecture, training data and objective at
once, so it cannot say whether some *readout* of these components would agree
with humans. §2.3's sweep answers that directly, because it moves the rate over
frozen representations: error consistency at all 21 values of `w`, for 3 seeds
x 2 checkpoint policies of the diet arm and 3 seeds of the no-diet arm -- 9
runs, **189 readouts** -- against the same 10 observers (**Figure 5a**). With
`w` chosen post hoc, per seed, to maximise agreement, the best kappa **never
exceeds 20.7% of the ceiling** in any run. The maximising `w` is neither stable
across runs (0.35 to 1.00) nor the one that matches the human rate, where
agreement is sometimes *negative* (-6.1%); and the diet buys nothing even with
the readout optimised (no-diet 17.3-18.0% against diet 7.6-20.7%). Because the
readout weight was given every opportunity to succeed and did not, this is a
**bounded** null rather than an unrejected one: equivalence intervals fixed
before any interval was computed (PLAN §77.5, §78) exclude a kappa change
larger than **0.037, 15.8% of the ceiling**, across the observed 0.706 span of
shape bias. Spanning 5x in shape bias moves trial-level agreement with humans
by at most a sixth of the ceiling. Per-`w` values and the full table: Extended
Data §E6.3; method §4.3 and `tmp_analysis/sweep_error_consistency.py`.

*The null is specific to the decision, not a property of the instrument.* Kappa
scores one binary response per image, so a reader may reasonably ask whether a
flat result reflects the models or the measure. We repeated the comparison with
a better-estimated target -- the same ten observers' per-image profile,
reliable at split-half **r = .658** for the shape rate and **.794** for whether
either cue resolved at all -- and a continuous model statistic, the
shape-minus-texture logit margin, choosing `w` on half the images and scoring
on the held-out half, as a percentage of those ceilings with category effects
removed (**Figure 5b**):

| | predicts *which cue* humans pick | predicts *whether either cue* resolves |
|---|---|---|
| diet arm (3 seeds) | 8.7% [2.0, 15.5] | **24.1%** [22.9, 25.4] |
| no-diet arm (3 seeds) | 8.0% [5.8, 10.8] | 9.2% [5.2, 13.0] |

**The instrument does detect a correspondence when one is there.** The same
machinery, on the same images and the same observers, reaches 24.1% of the
ceiling on image difficulty against 8.7% on the shape--texture decision -- a
2.8-fold gap, 3.6-fold on ranks, replicated under the other checkpoint policy
(22.5% against 11.2%). This is the positive control an equivalence interval
cannot supply on its own. The claim is not that the measure is three times more
sensitive in general, but that on this system and these images it demonstrably
recovers roughly three times as much structure from a neighbouring aspect of
the same human responses.

*What we do not conclude from it* (**Figure 5c**)*.* The diet arm's difficulty
correspondence (24.1%) exceeds the no-diet arm's (9.2%) with no overlap across
seeds, at both checkpoint policies and on both correlation types -- which
invites the reading that the developmental front end makes the model's
difficulty profile more human-like. **We do not make that claim, because our
own control does not support it.** Each arm is evaluated with its own front
end, so the two differ in test-time input as well as in learned weights.
Completing the 2x2 as in §2.4 yields a train/test *matching* pattern rather
than a weights effect: both matched cells are positive (24.1%, 9.2%) and both
crossed cells are not (-0.6%, -3.7%), and with the front end held off the
ordering reverses. This design cannot separate the learned representation from
the input transform, because separating them breaks the model; the 2x2 itself
is clean, so the failure is one of attribution, not of implementation.
Degradation of the crossed cells, the independence of the two streams' image
rankings, and an independent code path reproducing the rate range: Extended
Data §E6.3.

Method: `tmp_analysis/human_perimage.py`, `perimage_logits.py`,
`perimage_correspondence.py`, `perimage_frontend_2x2.py` (§4.3).

### 2.7 What the account costs


*Source: PAPER_PLAN §43.3, §48.2, §20.2, §20.3.*


Stated here rather than in a limitations footnote, because a reviewer finds these
immediately and because they bound the claim.

**Prediction accuracy** (**Figure 1d**)**.** The no-diet model predicts BOLD
**better**: median R²/NC **0.464 without the diet against 0.405 with it**, p=.0078 on both
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
a *readout over components*, they are not in competition at all. This paper
builds a system in which both hold at once: the luminance stream is
shape-biased (+0.819), the chromatic stream texture-biased (−0.518), it is the
texture-biased one whose unique variance carries the ventral end of the
hierarchy, and the readout that sees both is shape-biased at 0.675 — below the
shape-biased stream alone. A system needs only more than one component and a
readout that mixes them.

We are precise about what this licenses. It is an existence proof plus a
cortically grounded demonstration, not a measurement of the human readout: a
single-stream account in which texture-like ventral representations feed
shape-biased behaviour directly is not excluded by these data.

### Shape bias is a readout property, not a representational one

A model matching the human rate is routinely described as having human-like
representations. Our system shows that inference fails twice over. The same
network holds components at +0.819 and −0.518 and behaves at 0.675, so the rate
does not identify a representation; and across seven models spanning a 5× range
of shape bias, trial-by-trial agreement with ten human observers stays flat at
7–18% of the human–human ceiling, our most shape-biased stream lowest of all,
so the rate does not identify a decision process either. We pre-registered the
opposite prediction. The methodological consequence is direct: shape bias
should be reported as the readout statistic it is, alongside error consistency
wherever human trial data exist, and should not on its own license a claim
about a model's representations.

**This dissociation has precedent, and the precedent sharpens the claim rather
than pre-empting it.** Hermann et al. (2020) report that ImageNet-trained
networks decide on texture even where shape is decodable from their hidden
representations; Islam et al. (2021) find shape encoded more widely than the
behavioural rate implies. Both establish that a representation can hold more
shape than its behaviour reveals, and any reading of our title that claims
priority over them is wrong.

What is new is the *form* of the dissociation. In both prior demonstrations a
separately trained decoder recovers shape the classifier does not use, or a
different training run yields a different rate — so the representation is
either interrogated by something other than the network's own decision, or not
held fixed. Here it is held fixed. The classifier is affine and the mixture
weights sum to one, so the sweep is exact interpolation between two endpoints
(§2.3), and the rate moves across most of the range the field's models occupy
while both representations stay bit-identical. That the affine case is an
identity invites the objection that the result is a property of the head; it is
not, because a refitted nonlinear readout on the same frozen features
reproduces 86% of the span and agrees with a refitted linear head to within 1%
(PLAN §104). Neither prior study reports brain data, so §2.2's ventral gradient
has no counterpart in them, and neither asks whether *any* readout of a fixed
pair of representations reaches human decision structure. §2.6.2 bounds that
directly, and the answer is no.

**The sweep does not reach the human value**, and that sharpens the claim
rather than softening it. Its maximum over all 21 weights, 3 seeds and both
checkpoint policies is 0.913 — below the lowest individual observer (0.919), and 0.045
below the human mean. A
single scalar moves the rate across most of the modelled range *without the
family ever reaching a human observer*, so within this family the rate is a
free parameter of the readout, and a rate measured at one operating point
constrains the representation no further than the family's range already does.

The two-condition literature invites the reading that the magnocellular channel
*is* the shape pathway. §2.4–2.5 rules that out: shape bias requires a carrier
× transform conjunction, neither factor crossing the boundary alone, while
effective dimensionality needs only bandwidth and reverses when the carrier
moves. Both properties are relational — facts about what a stream receives, not
about what it is called. That matters beyond our architecture, because
"magno-like" and "parvo-like" are routinely used as if they named intrinsic
computational roles; here the trunks are bit-identical at initialisation, so
the labels can name nothing but the input.

### What a developmental constraint does, and what comes apart

The intuitive account of developmental low-pass input is informational: coarse
input first protects a learner from high-frequency texture, so it learns shape.
Our results do not support it. The transform *removes* bandwidth rather than
adding structure, and the shape bias it produces is already present untrained
(+0.796). What it decides is what learning *does not destroy*: without it,
training drives the luminance stream from +0.417 to −0.219; with it, +0.796 is
still +0.810 after 150 epochs. The same signature appears on a second
instrument — diet-trained networks end magno-dominant from front ends spanning
3.26×, including one beginning parvo-dominant in 8/8 subjects, while every
no-diet network collapses to parity from the most magno-dominant front end in
the set. What replicates is the *side*, not a level: a single condition's
trained ratio ranges over 0.79 across seeds, wider than the gap between
conditions, and two readings we ourselves previously advanced do not survive
that spread.

Shape bias, dimensionality, out-of-distribution robustness and human-likeness
are routinely discussed as a package. Over identical inputs this paper
separates them three times, and a readout account says why they should
separate: none of the three is a property of a readout. Dimensionality comes
apart opposite to intuition, the `parvo_highsf` cell being more dimensional
*and* more texture-biased. Robustness comes apart in a way its own control
sharpens — the diet-trained network is less robust on four of the five
corruption families it was not trained on and more robust on exactly the one
the diet installs — so where Geirhos et al.'s stylisation route yields shape
bias together with robustness, our developmental route yields comparable shape
bias in rate and none of it. The coupling belongs to the route, not to shape
bias. And human agreement comes apart most cleanly of all.

### What generalises, and what does not

The readout account makes a prediction beyond this architecture: in any system
whose behaviour is measured by a cue-conflict rate, that rate should be
recoverable from a weighted combination of component biases, and moving the
weighting should move the rate without moving any component. §2.3 tests it
directly. What remains open is the *learned* version — a weighting fitted to
data rather than swept, and whether any principled objective selects the human
operating point. Nothing here selects it.

What does not generalise is any recommendation to use a developmental diet as a
training method. It makes models less accurate (0.556 against 0.701 at the best
saved checkpoint), less stable late in training, and less robust on four of
five corruptions it was not trained on. Its interest is as a model of
development.

The load-bearing cortical result (§2.2) rests on eight subjects from a single
dataset; the `parvo_highsf` condition varies architecture alongside the front
end; our diet is a fixed 48-month front end rather than a trajectory; human
comparisons are against published observers and of aggregate rate; and the
correspondence between our chromatic stream and texture-selective ventral
cortex is by explanatory role, not by content. See *Limitations* for the full
accounting.

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

**Error consistency** (§2.6.2). Cohen's kappa on the binary shape/texture choice
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

*Is unique variance identifiable in these two bands?* Unique variance is a
subtraction of cross-validated R² (§4.7), so it means something only if each band
carries directions the other cannot predict. Measured on the fitted
1,000-component feature spaces, the canonical-correlation spectrum between the
magno and parvo bands runs from ≈0.000 to 0.925 (diet) and ≈0.000 to 0.954
(no diet), with median 0.292 and 0.306 across the 8 subjects. Each band therefore
retains directions essentially unpredictable from the other — the quantity is
identifiable in these spaces — while the two bands also share a strongly
correlated subspace, which is the regime `banded_ridge.py`'s own docstring warns
makes unique components *understate* a band's contribution.

Identifiable and correctly recovered are two claims with two different sources of
evidence, and only the first is supported by the spectrum. The second is answered
by the recovery simulation reported in §2.2: with each band's real feature
geometry and a known across-ROI split imposed, the estimator returns the true
ordering exactly (Spearman ρ = +1.000 against a true +1.000, in 8/8 subjects on
both arms, per-ROI absolute error ≤ 0.137). Magnitude fidelity is a separate
matter and is treated where it bears on the result (§2.2, the multiplicative term
and its residual misfit); it is why §2.2 is reported as sign, ordering and
monotonicity rather than as calibrated magnitude.

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

**Source directories, and the one exception.** Encoding sources are declared per
table rather than globally, because neither directory is a superset of the other
(§2.2, Extended Data §E2.2). §2.2's seed table comes from
`results/encoding_cc/`; the front-end control table and the **untrained
baseline** come from `results/encoding/`, which is the only directory holding
those models. The untrained baseline is therefore the one value in §2.2 not
drawn from that section's declared source, and we state it rather than repair it,
for two reasons. First, it changes nothing: the two directories differ only by
`run_encoding.py`'s `--cue-conflict` flag, which applies the *already fitted*
estimators to an additional design and adds keys, while all band statistics are
taken from the training split only — so the flag cannot move `r2_joint`,
`r2_magno` or `r2_parvo`. Checked rather than assumed: across the 40 files
present in both directories (5 models × 8 subjects) the maximum absolute
difference in all three quantities is exactly zero, and `verify_new_claims.py`
now locks both that invariant and the untrained anchor itself. Second, refitting
it into `encoding_cc` would replace the row with the longest unbroken provenance
in the analysis — six of its eight subject fits date from 2026-08-27/28 and
predate the quarantine described in §E2.2 — with a new fit, which is a loss of
audit value rather than a gain. Its regenerable source is the
provenance-stamped artefact `results/visualrois/untrained_seed0_2026-09-07.json`,
which records every input file with size and mtime.

**Pre-registration.** The front-end 2×2 (§2.4), the inverted-front-end prediction
(§2.5), the OOD families (§2.6.1) and the error-consistency hypothesis (§2.6.2) were
each written down before the corresponding analysis was run. Two of the four
predictions failed and are reported as failures in the sections above.

**What "pre-registered" means here, exactly.** These predictions were written
into a version-controlled project plan (`docs/PAPER_PLAN.md`) and committed
before the corresponding analysis was run, so the ordering is verifiable from
the git history and the relevant commit hashes are given in Supplementary Note S3.
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
largest, dz = −6.60, returns exactly the same .0078125 as a marginal 8/8).
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

**Sex as a biological variable.** Both human samples are existing published
datasets and we collected no new human data, so participant sex was not under
our control and was not a design factor. No analysis in this paper is
disaggregated by sex, and none could be: the neuroimaging sample is eight
participants and the behavioural reference ten observers, and at those sizes the
paired tests used throughout already sit at their exact resolution floor on the
*full* sample (§4.7), leaving no power for any subgroup contrast. We also do not
restate participant sex, and the reason is specific to each release rather than a
blanket deferral. For the **behavioural** sample it is not available to us at
all: the Geirhos per-observer files carry only trial-level fields (`subj`,
`session`, `trial`, `rt`, `object_response`, `category`, `condition`,
`imagename`) and no demographic column, so no per-observer sex exists in the data
we analysed. For the **neuroimaging** sample the NSD release documents its
participants, but the subject-information file is not part of the subset we
mirrored, so we have not verified any value against the released metadata and do
not reproduce one here; the Reporting Summary points to the NSD release for it
rather than restating it. An earlier version of this paragraph said sex "is
recorded in the source releases for both datasets" and promised a per-subject
table; the first half is not true of the behavioural release and the second could
not have been delivered for it. We therefore make no claim, in either direction,
about sex differences in any quantity reported here.

**Data availability.** The Natural Scenes Dataset is available at
`naturalscenesdataset.org` under its own data-use agreement, which requires
registration; we used the `betas_fithrf_GLMdenoise_RR` (b3) preparation and the
`nsdgeneral` mask. The Geirhos cue-conflict stimuli and human observer responses
are distributed at `github.com/rgeirhos/texture-vs-shape` under CC BY 4.0 (see
*Materials*); we read the observer data from that repository's
`raw-data/style-transfer-512-nomask-experiment/`. The same experiment is
mirrored at `github.com/bethgelab/model-vs-human` under
`raw-data/cue-conflict/`, and the two mirrors were parsed independently by two
scripts that agree to four decimals on the human reference (mean 0.958, sd
0.019, range 0.919–0.987), which is how that number was validated. MiniEcoset is
at `osf.io/msna2` (Thorat & Kietzmann, 2024). Derived results tables and the fitted encoding-model summaries
needed to reproduce every figure will be deposited with a DOI on acceptance.

**Code availability.** Training, front-end, encoding, behavioural and analysis
code will be released with a DOI on acceptance; the pinned environment manifest
(`requirements.txt`) and the reproduction path are already in the repository,
whose `README.md` gives a single command that re-derives reported numbers from
the committed result files without any dataset download or GPU. The
modules that produce the numbers reported here are named by path in §4.1–§4.7,
and none of them imports any of the four laboratory git dependencies
`setup.cfg` also lists — the manifest's header records where each of those is
actually used.

**Materials.** The cue-conflict stimulus set and the ten observers' trial-level
responses are the public Geirhos release (Geirhos et al., 2019, data release),
distributed under the **Creative Commons Attribution 4.0 International (CC BY
4.0)** licence stated in that repository's `DATASET_LICENSE` file, which we
verified against the file itself rather than inferring it from the repository
page. We use the release under that licence, with attribution as given in the
reference list. Our copy is the `raw-data/style-transfer-512-nomask-experiment/`
directory (ten observers plus the three published CNN sessions). **Modifications,
as CC BY 4.0 requires us to indicate:** we restricted the set to the 16
categories and excluded the 80 trials whose shape and texture cue coincide, as
described in §4.3, and we recomputed the observer summary statistics from the
per-trial CSVs rather than using published aggregates; the stimulus images
themselves are unmodified. Note the repository separately states that the
ImageNet-derived *images* underlying the stylisation cannot be redistributed by
its authors, which does not affect the observer response data we use.

**Software and code.** No software was used for data collection: both human
datasets are existing public releases (see *Data availability*). Analysis ran
under Python 3.11.15 with NumPy 2.4.6 (Harris et al., 2020), SciPy 1.17.1
(Virtanen et al., 2020), PyTorch 2.13.0 (CUDA 13.0 build; Paszke et al.,
2019), torchvision 0.28.0+cu130, himalaya 0.4.11 (Dupré la Tour et al., 2022)
for the banded-ridge solver, scikit-learn 1.9.0 (Pedregosa et al., 2011),
Matplotlib 3.11.1 (Hunter,
2007), pandas 3.0.5, nibabel 5.4.2, h5py 3.16.0, Pillow 12.3.0, joblib 1.5.3 and
tqdm 4.70.0, and kornia 0.8.3 for the GPU augmentation set of §4.2. Every
version above was read from the installed environment's
package metadata rather than transcribed from memory. Jobs were scheduled with
Slurm 25.11.8 on CPU nodes for all encoding, statistical and plotting work and
GPU nodes for network training only. Cortical ROIs use the probabilistic atlas of
Wang et al. (2015) as distributed with NSD (`prf-visualrois`, `prf-eccrois` and
`streams`); the beta preparation is NSD's `betas_fithrf_GLMdenoise_RR` (b3). An
environment manifest pinning these versions is in the repository as
`requirements.txt`, generated from that same package metadata.

*On RRIDs.* Nature-portfolio reporting asks for research-resource identifiers
where a resource is registered. This study used no antibodies, cell lines,
organisms or other wet-lab reagents, so those categories do not apply. For the
software above we give name, exact version and, where one exists, the citation
the authors ask for — which is what a reader needs to reconstruct the
environment. We do not list SciCrunch RRIDs for the individual Python packages,
because we have not verified registry entries for each one and an unverified
identifier is worse than none; any that the journal requires will be taken from
the registry at submission rather than asserted here.

**Reporting Summary.** The content for every item of the Nature-portfolio
Reporting Summary is drafted in `docs/REPORTING_SUMMARY.md`, so
that completing the journal's form at submission is transcription rather than
recall and each entry has a source in the repository. The form itself is
supplied by the journal at submission; the content was drafted against the
Behavioural & Social Sciences variant, and the items it covers — statistics,
software and code, data, human participants, and study design — are common to
the Nature-portfolio variants. Machine-learning reporting items (data
provenance and splits, leakage controls, optimisation, model and baselines,
evaluation) are stated in §4.1–§4.7. Two of its items cannot be
completed from the data we hold and are marked as gaps in place rather than
filled: participant age and sex (not present in the behavioural release, and not
in the NSD subset we mirrored), and non-participation during the original
collections.

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

Geirhos, R., Rubisch, P., Michaelis, C., Bethge, M., Wichmann, F. A., & Brendel,
W. (2019). *Data and code for "ImageNet-trained CNNs are biased towards
texture"* [Data set]. GitHub. https://github.com/rgeirhos/texture-vs-shape
(raw human observer data: `raw-data/style-transfer-512-nomask-experiment/`;
licensed CC BY 4.0 per the repository's `DATASET_LICENSE`).

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

Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P.,
Cournapeau, D., et al. (2020). Array programming with NumPy. *Nature*, 585,
357–362. https://doi.org/10.1038/s41586-020-2649-2

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image
recognition. *IEEE Conference on Computer Vision and Pattern Recognition*,
770–778. https://doi.org/10.1109/CVPR.2016.90

Hermann, K., Chen, T., & Kornblith, S. (2020). The origins and prevalence of
texture bias in convolutional neural networks. *Advances in Neural Information
Processing Systems*, 33. https://doi.org/10.48550/arXiv.1911.09071

Hubel, D. H., & Wiesel, T. N. (1968). Receptive fields and functional
architecture of monkey striate cortex. *The Journal of Physiology*, 195(1),
215–243. https://doi.org/10.1113/jphysiol.1968.sp008455

Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. *Computing in
Science & Engineering*, 9(3), 90–95. https://doi.org/10.1109/MCSE.2007.55

Islam, M. A., Kowal, M., Esser, P., Jia, S., Ommer, B., Derpanis, K. G., &
Bruce, N. (2021). Shape or texture: Understanding discriminative features in
CNNs. *International Conference on Learning Representations*.
https://doi.org/10.48550/arXiv.2101.11604

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

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O.,
et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine
Learning Research*, 12, 2825–2830.

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

Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T.,
Cournapeau, D., et al. (2020). SciPy 1.0: Fundamental algorithms for scientific
computing in Python. *Nature Methods*, 17, 261–272.
https://doi.org/10.1038/s41592-019-0686-2

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
   saved checkpoint (§2.6.1); late-training stability (§2.6.1); and robustness on
   four of the five corruptions it was not trained on (§2.6.1). The paper makes no
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
16. **A control removed a claim: the shape-bias advantage over one stream is
   not attributable to the split.** This limitation previously recorded that the
   single-stream control was not parameter-matched and that the confound ran
   *in favour of* the shape-bias result. It fired. The pre-registered `resnet34`
   matched arm was run and does not separate from the two-stream model (0.626
   against 0.655, overlapping ranges under both checkpoint policies), while the
   half-capacity control does (0.567); capacity alone accounts for a difference
   of the same size. The accuracy half moved the other way and strengthened
   (§2.3, Extended Data §E3). This is no longer a missing control but a
   withdrawn attribution.


## Claims deliberately not made

Recorded so the omissions are visible rather than silent.

- **That the two-stream split causes the shape-bias advantage over a single
  stream.** The pre-registered parameter-matched control was run and does not
  separate from the two-stream model; the half-capacity control does, but a
  capacity change alone moves shape bias by about the same amount (§2.3,
  Extended Data §E3). The comparison therefore cannot carry a causal claim
  about the split. We equally do not claim the converse -- that the split buys
  nothing for shape bias -- which three seeds a side cannot establish. The
  accuracy side of the trade is claimed, and is stronger at matched parameters.

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
- **That the diet makes the model's image-level difficulty profile more
  human-like.** The per-image analysis in §2.6.2 shows the diet arm predicting
  which images humans fail to resolve to either cue at 24.1% of the human noise
  ceiling against the no-diet arm's 9.2%, with no overlap across three seeds, at
  both checkpoint policies and on both correlation types. We ran the crossed
  front-end 2×2 before writing that up, and it does not support the reading. The
  pattern is train/test *matching*, not a weights effect — both matched cells are
  positive and both crossed cells are not, and the ordering reverses with the
  front end held off — and the crossed cells are too degraded to arbitrate
  (resolving to neither cue on 70–85% of images against 49–63% matched). The
  design cannot separate the learned representation from the input transform,
  because separating them breaks the model. The comparison is reported in §2.6.2
  with the control that kills it, rather than omitted.
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

An earlier version of §2.2 compared the cortical instrument's seed stability
against §2.5's and called it **28-fold**. That ratio was computed against a §2.5
seed range of 0.904 which was itself a final-checkpoint artefact: on weights
selected by validation accuracy the range falls to 0.120 (§S1.3). The cortical
instrument remains the tighter of the two, but not by the margin previously
claimed, and we no longer attribute §2.5's spread to the participation-ratio
measure — it belonged to the checkpoint (PLAN §71.10).

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
late-training degradation documented in §2.6.1: on weights selected by validation
accuracy the two ranges fall to **0.120 and 0.787**, a 7.6× and 3.0× tightening,
while the two arms whose best checkpoint *is* the final one do not move at all.
The same correction appeared independently on accuracy (§2.6.1, seed range
0.180 → 0.012). The diet arm is therefore **not** markedly less reproducible than
the control on either instrument; it was being scored on degraded weights
(PLAN §71.10).

### S1.4 — checkpoint policy on the OOD table (§2.6.1)

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

### S1.5 — how the models were selected, and the 37% / 16.5% / 20.7% history (§2.6.1)

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


### S1.6 — the three withdrawn framings of the no-diet gradient (§2.2)

§2.2 reports that the diet multiplies an already-present gradient by 7.1, and
states that three earlier framings of that comparison are withdrawn. What each
claimed, and why it failed, is recorded here.

Three earlier framings are withdrawn. The first — "nearly flat without it", with a 79%
reduction — used numbers that do not regenerate (§E2.2), but its *structure* was
right, and the honest correction is 85% rather than a withdrawal: our objection
that it "compared two numbers as though both measured something" was wrong,
because both do. The second, which replaced it, claimed no gradient was
resolvable without the diet. That was an error of ours in the opposite
direction: it compared the magnitude of the no-diet contrast to the magnitude of
the floor, as though the floor were a noise level. The third, ours again, declined to claim the no-diet
gradient in either direction on the strength of a residual correction that we
have since identified as a simulation artefact (below). The claim above is the
second framing's structure with the first framing's direction and numbers that
regenerate. The floor is a **signed
positive bias**, and a positive bias cannot manufacture a negative
observation — it masks one. Treating a signed quantity as an unsigned tolerance
is the same mistake that made subject 01's outlier look arm-specific, and it is
the reason the paired displacement above, not a ratio, is the statistic we
report.


---

### S1.8 — the no-diet monotonicity ρ was an average of two estimators (§2.2)

§2.2 previously gave the no-diet hierarchical-rank correlation as **−0.536**.
That value does not reproduce under either declared statistic: the `norm`
profile gives −0.5025 and the `raw` profile −0.5713 (both from
`results/encoding_cc`, and identical in `results/encoding`). Their mean is
−0.5369, which is the reported figure to within rounding, and no other
construction tried comes within six times as close. It was therefore an average
of the two estimators rather than either of them.

The diet value is unaffected: `norm` and `raw` both give −0.9637 there, because
Spearman is rank-based and the two statistics are monotonically related within a
profile — they can only diverge where the profile's ranks differ between them,
which happens in the shallow no-diet arm and not in the steep diet one. §2.2 now
reports both statistics separately, as §8 requires. This is the same failure
mode §81 records for the `±` column: a number reported without its estimator.

### S1.7 — the readout sweep was presented as a measurement (§2.3)

§2.3 now states that its `w` sweep is exact arithmetic on two endpoint logit
vectors. An earlier version claimed the shape of that dependence was "not
forced" and was therefore a measurement.

*An earlier version of this section claimed that the shape of that dependence
was "not forced" and was therefore a measurement. That was wrong, and the
correction matters enough to make here rather than in a note.* The classifier is
`AdaptiveAvgPool2d → Flatten → Linear` (§4.3) — affine — and the mixture weights
sum to one, so

> `clf(w·y₁ + (1−w)·y₂) = w·clf(y₁) + (1−w)·clf(y₂)`

exactly (verified numerically: maximum deviation 1.1 × 10⁻⁶ on a logit scale of
0.32). The swept logits are therefore the **convex combination of the two
endpoint logit vectors**, and the sweep contains no information beyond `w` = 0
and `w` = 1, both of which this section already reports. Per image each class
logit is linear in `w`, so the decision is the argmax of 16 lines and the
aggregate rate is close to the empirical distribution of per-image switch
points. **Monotonicity is very nearly analytic, not a discovered property**, and
we no longer present it as one.

What survives is the weaker statement the section actually needs, and it is
enough for the argument: a behavioural shape-bias rate is a property of a
readout over components, not a report of either component, and here a single
scalar in the readout carries the system across the entire range the field's
models occupy while both representations are untouched. What that does **not**
establish is that any particular mixture is the one a human uses, or that the
dependence is an empirical law rather than interpolation between two measured
endpoints.


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


## Supplementary Note S3 — the pre-registration commit ledger

§4.7 states that each pre-registered prediction was committed to
`docs/PAPER_PLAN.md` before the corresponding analysis was run, and that the
ordering is inspectable from the git history. This note gives the hashes, so the
claim can be checked rather than taken on trust. All times are the commit's
author time in `+0200`.

| Draft section | Plan section | Declaring commit | Declared | Result available | Gap |
|---|---|---|---|---|---|
| §2.4 front-end 2×2 | §45 (ordering, and §45.5's read-only-after-all-32 rule) | `6bc04b8` | 2026-09-01 13:12:50 | job 26406434 first task ended 2026-09-01 15:03:23 | 1 h 50 m |
| §2.5 inverted front end | §47.6 | `35d7af7` | 2026-09-01 19:01:33 | falsified in `9963cf3`, 2026-09-02 11:24:50 | 16 h 23 m |
| §2.6.1 OOD families | §53 | `a45b8bb` | 2026-09-02 13:56:53 | first OOD result file 2026-09-02 15:46:49 | 1 h 50 m |
| §2.6.2 error consistency | §57 | `b9c5296` | 2026-09-02 22:06:03 | κ sweep 2026-09-07 01:34:13 | 4 d 3 h |

Three things this ledger does **not** establish, and we would rather name them
than let the table imply them.

*Git dates are author-controlled.* A commit's author and committer timestamps
can both be set by the person making the commit. The ordering above is therefore
an internal audit trail we are exposing, not a third-party attestation — exactly
the restricted sense §4.7 defines. A referee who wants the ordering
independently attested should treat it as unattested; nothing here can
retroactively supply what a registry entry would have.

*The plan came under version control after the project started.*
`docs/PAPER_PLAN.md` was first committed on 2026-08-30, and the branch's first
commit is 2026-08-25. Predictions made in the first five days of the project
have no commit that predates their analysis, and none of the four listed above
relies on that window — but the audit trail does not cover it, and we do not
claim pre-specification for anything analysed before 2026-08-30.

*Checking it requires the repository.* The hashes are only useful to a reader
who can resolve them. The plan file and its history are therefore part of what
the *Code availability* statement commits to depositing, not an artefact that
stays private; a DOI deposit that ships the code without the plan would leave
this table uncheckable.

Two of the four predictions failed. §47.6's falsification is in the table above;
both failures are reported as failures in the sections that carry them, which is
the only reason the ledger is worth publishing at all — a pre-registration
ledger that never records a loss is not evidence of anything.
