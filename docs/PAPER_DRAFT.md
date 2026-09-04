# Shape bias requires a conjunction of achromatic carrier and developmental transform

**Status: FIRST DRAFT, 2026-09-02.** Structure and argument are settled; prose is
first-pass. Every number is tagged with its `PAPER_PLAN.md` section so it can be
traced to the run that produced it. Nothing here is written from memory.

**Positioning (decided 2026-09-02).** The paper leads with what survives its
controls. It is not an audit of the BA thesis: where a thesis-era analysis was
superseded, the better version is presented without a post-mortem. Two claims
the thesis's framing invited are simply **not raised**, because we do not make
them — see *Claims deliberately not made* at the end.

---

## Abstract

Primate vision splits its input at the retina into a low-acuity achromatic
magnocellular channel and a chromatically opponent parvocellular one, and the
statistics of that input change over development. Two-stream networks trained on
biologically shaped channels have been reported to develop a shape-biased
stream — the property on which standard CNNs most conspicuously depart from
human observers — but *what* produces it has not been isolated, because no
previous comparison varied the channel and the developmental transform
independently.

We train two-stream networks under a developmental visual diet — a low-pass
transform matched to 48-month acuity — and evaluate them behaviourally, against
published human observers, and against eight subjects of the Natural Scenes
Dataset.

**The shape bias is an interaction, not a channel property.** Crossing which
carrier a stream receives (achromatic luminance vs chromatic L−M) with whether
the transform is applied to it gives four front ends. Neither factor alone moves
a stream across the shape/texture boundary (linear-probe bias −0.266 and −0.145);
only the conjunction crosses, and it crosses decisively (**+0.819**).
Cue-conflict behaviour on the same four cells replicates this independently
(0.364 / 0.495 / 0.396 / **0.878**, mean of 3 seeds; human observers on these
identical stimuli are at 0.958, though that compares a whole observer to our
single stream,
and the whole model sits at 0.692). Shape bias is therefore not a property of
the magnocellular channel, which is the reading the two-condition literature
invites.

**What the diet contributes is a constraint on learning, not better input.** A
front end whose stream asymmetry is *inverted* (0.586, parvo-dominant in 8/8
subjects) trains to **magno-dominant**, overtaking a network handed 1.402. The
diet does not preserve the asymmetry it is handed, nor drive training to any
particular value — across three training seeds the trained level ranges over
2.39 in a single condition. What replicates is **which side of parity** training
ends on: all nine diet seeds end magno-dominant in 8/8 subjects, across a 3.26×
spread in what they were handed, while all three no-diet seeds collapse to
parity from the *most* magno-dominant front end in the set (1.634). Seed-level
separation is complete for every diet arm (p=0.100, the exact 3×3 floor), and a
model trained on a real developmental schedule in an independent codebase lands
on the same side. The divergence is measurable through human visual cortex:
per-stream voxelwise encoding models yield a stream dissociation of **+1.534**
with the diet and **+0.110** without (p=.0078, 8/8), and the diet-trained
network's division of labour tracks the human ventral hierarchy (hV4−V1v −0.862)
where the no-diet network's does not (−0.117).

**Three properties the field treats as coupled to shape bias do not follow from
it.** Effective dimensionality over these same front ends is explained entirely
by carrier bandwidth and is reproduced with no trained network at all. Across
five corruption families the model was not trained on, the diet-trained network
is *less* robust than every control — winning only on the low-pass family the
diet literally installs — so a developmental route to shape bias delivers none
of the robustness a stylisation route delivers, locating that coupling in the
route rather than in shape bias. And against the 10 human observers who viewed
these stimuli, error consistency is flat: across six models spanning a 5× range
of shape bias (0.17–0.89), trial-by-trial agreement with humans stays at 7–18%
of the human–human ceiling, with our shape-biased stream no better than AlexNet.
We pre-registered the opposite prediction. Matching the human shape-bias rate
does not confer human decision structure.

The diet's effect is not a filtering artefact and not a property of any
privileged channel. It is a constraint on what learning converges to — bought at
a substantial cost in accuracy and stability that we report in full.

> TODO: tighten to ~150 words for submission. Current draft is a working abstract.

---

## 1. Introduction

> DRAFT — argument fixed, prose to be tightened.

Three observations motivate this work.

**(i) The primate visual system does not receive one image.** It receives a
luminance-dominated, temporally fast, spatially coarse magnocellular signal and a
chromatically opponent, spatially finer parvocellular one, separated before
cortex. Models of ventral-stream computation almost universally discard this,
taking RGB pixels as input.

**(ii) That input is not stationary over development.** Infant acuity and
contrast sensitivity are poor and improve over the first years; the visual system
learns first from a low-passed world. A *developmental visual diet* (DVD)
implements this as a progressive transform on the achromatic channel.

**(iii) Shape bias is the property at stake.** Standard CNNs classify by texture
where humans classify by shape. A two-stream network trained under a
developmental diet has been reported to develop a shape-biased stream. Two
questions have been left open, and this paper answers both. *What* produces the
bias — the achromatic channel, the developmental transform, or their
conjunction? No previous comparison varied the two independently, so the field's
default reading assigns it to the magnocellular channel. And *how* does the diet
act, given two readings that no previous comparison could separate:

- **Inheritance.** The transform changes the input's statistics; the network
  merely reflects what it is handed.
- **Installation.** The transform changes what learning converges to,
  irrespective of what the input hands over.

These make identical predictions whenever the input asymmetry and the trained
asymmetry point the same way — which was true of every condition previously
tested. Distinguishing them requires a front end whose asymmetry points the
*other* way. We build one and use it as the decisive case.

**Contributions.**

1. A four-cell causal decomposition (achromatic carrier × developmental
   transform) showing that shape bias is an **interaction**, not a
   property of either factor — measured on all four cells, in a linear probe and
   in cue-conflict behaviour, against published human observers (§2.1).
2. The inheritance/installation test, using a front end whose asymmetry is
   inverted: the diet does not preserve what it is handed, and does not drive
   training to a fixed value. It fixes **which side of parity** training ends on
   (§2.2).
3. Confirmation of the stream divergence through human visual cortex, on
   independent machinery (§2.3).
4. **Three dissociations.** Over identical inputs, shape bias separates from
   effective dimensionality — explained by carrier bandwidth and reproducible
   with no network at all (§2.4); from out-of-distribution robustness, where the
   diet-trained network is *less* robust than every control on five corruption
   families it was not trained on (§2.5); and from human decision structure,
   where a 5× range of shape bias across six models leaves trial-by-trial
   agreement with 10 human observers flat at 7–18% of the human–human ceiling
   (§2.6). Properties the field treats as a package do not travel together.
5. An honest accounting of what the diet costs (§2.7).

---

## 2. Results

### 2.1 The shape bias requires a conjunction, not an achromatic channel


*Source: PAPER_PLAN §25, §13, §18, §21.*

The diet-trained network's magno stream is strongly shape-biased and its parvo
stream is not. The natural reading — that shape bias belongs to the achromatic
magnocellular channel — is wrong.

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
three checkpoints vary by **0.904** on the magno/parvo dimensionality ratio of
§2.2. Seed instability is a property of the dimensionality measure, not of these
models, which is why §2.2 claims a side rather than a level while §2.1 can quote
values.

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

### 2.2 The diet fixes which side of parity training ends on


*Source: PAPER_PLAN §45, §49, §50. This is the paper's mechanistic core.*

The four front ends of §2.1 hand a network very different stream asymmetries
before any learning — the untrained magno/parvo dimensionality ratio spans 0.59
to 1.91 across them (quantified in §2.4). What does training do with them?

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
| diet | luminance | `dkl_bcs_seed0` | 1.402 ±0.089 | **1.603** ±0.032 |
| diet | luminance | `dkl_bcs_parvo_highsf_seed0` | **0.586** ±0.013 | **1.640** ±0.030 |
| diet | **red−green** | `dkl_bcs_swap_seed0` | **1.909** ±0.125 | **2.225** ±0.079 |
| no diet | — | `dkl_base_seed0` | 1.634 ±0.118 | 0.947 ±0.035 |
| no diet | — | `dkl_base_seed1` | 1.634 ±0.118 | 0.988 ±0.040 |

**Seed replication (n=8 subjects x 3 seeds x 4 conditions).** Every level above
is one checkpoint. Array 26413641 extracted seeds 1-2 for all four conditions;
the levels turn out not to be reproducible, and the side does:

| arm | front end | seed 0 | seed 1 | seed 2 | seed range |
|---|---|---|---|---|---|
| diet / luminance | 1.402 | 1.603 | 2.507 | 1.823 | 0.904 |
| diet / parvo_highsf | **0.586** | 1.640 | 1.497 | 1.664 | 0.166 |
| diet / red−green | 1.909 | 2.225 | 4.383 | 1.995 | 2.388 |
| no diet | 1.634 | 0.947 | 0.988 | 0.914 | 0.074 |

All nine diet seeds are above parity in **8/8 subjects**. All three no-diet seeds
sit at or below it. Seed-level separation between the diet arms and no-diet is
**complete** for all three arms (p=0.100, the exact two-sided 3x3 floor).

We therefore report no level, no set point, and no compression factor anywhere:
the within-arm seed range reaches 2.39, so any statement of the form "training
converges to X" is unsupported. What replicates is the side, and that is what
this section claims. *(This retires two earlier readings of our own data — a
"convergence to within 2%" between the two diet-on-luminance arms, whose 0.037
gap is ~24x smaller than the seed noise on that same arm, and a channel-specific
set point, which n=3 seeds cannot separate: PLAN §58.3.)*

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
identical transform onto the chromatic channel (`swap`) lands at **2.225**,
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
(§2.4), and the no-diet front end hands over more asymmetry, not less. It is a
constraint on the solution learning converges to.

*The references are exact for the two decisive arms.* `untrained_parvo_highsf`
and `untrained_swap` each carry provenance identical to their trained model on
every recorded key except the checkpoint, so those are matched front-end
pairings rather than approximate ones (§49.2, §50.1).

### 2.3 The divergence is measurable through human visual cortex

**Seed replication (n=8 subjects × 3 seeds).** Because §2.2 showed that
single-checkpoint levels are not always reproducible on this project's
instruments, the cortical contrast was re-run for seeds 1–2 of both arms:

| seed | diet | no diet | difference | p | subjects |
|---|---|---|---|---|---|
| 0 | +1.534 | +0.110 | +1.424 | .0078 | 8/8 |
| 1 | +1.566 | +0.098 | +1.468 | .0078 | 8/8 |
| 2 | +1.558 | +0.083 | +1.474 | .0078 | 8/8 |

The diet arm's seed range here is **0.032**, against **0.904** for the same three
checkpoints on the dimensionality ratio of §2.2 — a 28-fold difference in seed
stability on the same models. The instability documented in §2.2 belongs to the
participation-ratio measure, not to the models or to this paper's instruments
generally: §2.1's probe (0.014) and this dissociation (0.032) are both an order
of magnitude tighter. The raw statistic agrees on every seed, so the
two-statistics rule is satisfied three times over.


*Source: PAPER_PLAN §43.1, §17, §48, §29.*

We fit voxelwise banded-ridge encoding models per stream on NSD (8 subjects, b3
betas, `nsdgeneral`, test set = the shared 515 images), then use them as
measurement instruments in two ways.

**(a) In-silico cue conflict.** Passing the 1,200 cue-conflict images through the
fitted models yields a per-voxel predicted shape-bias index; the magno−parvo
difference is **+1.534 with the diet and +0.110 without** (n=8, both statistics,
p=.0078, 8/8 subjects). It is carried entirely by the magno stream (+0.791 vs
**−0.638** — without the diet the magno stream is *texture*-dominated), with the
parvo streams indistinguishable (−0.743 vs −0.748). This confirms the
representational result of §2.1 on independent machinery, through the brain data,
using no probe and no readout.

**(b) The ventral gradient of stream division of labour.** The stream-bias
gradient from V1v to hV4 is **−0.862 with the diet** and **−0.117 without**, with
the untrained front ends at −0.488 — i.e. training with the diet nearly doubles
the input-level gradient while training without it reduces it by ~76%. The effect
size separates by a factor of ~19 (dz −11.70 vs −0.61) and monotonicity collapses
(ρ −0.964 vs −0.512).

Three points make this more than a single fitted number:

- **It replicates across codebases.** Our standalone-trained model and the
  independent thesis checkpoint give −0.8618 and −0.8593 — two trainers, two
  augmentation pipelines, two seeds, agreeing to 0.0025 (§17).
- **It survives its front-end controls.** Against `swap` (−0.027) and
  `parvo_highsf` (−0.119) as well as no-diet (−0.101), the diet arm separates at
  p=.0078 on all three (§48.1). See §2.7 for what those controls cost.

*The gradient is the weaker of this section's two instruments and is not quoted
with equal confidence.* Across the three seeds it replicates in direction and in
both statistics (normalised: −0.447 / −0.424 / −0.501 for the diet against
−0.202 / −0.166 / −0.160 for no diet, p=.0078 and 0/8 throughout), but the raw
statistic reaches the exact floor only in seed 2 (p = .0234, .0156, .0078). The
stream dissociation above is the claim; the gradient is corroboration.
- **It is not explained by a global stream preference.** The diet-trained and
  no-diet models sit on *opposite* sides of the global stream-bias axis (−0.454
  vs +0.727), yet the strong regional gradient is in the model whose global bias
  points the other way (§17.2).

**(c) Behaviour, against human observers.** The integrated readout's
cue-conflict shape bias is **0.675 ±0.025** with the diet against **0.416
±0.020** without, across three seeds each. The thesis checkpoint reproduces at
0.697 against its reported 0.677, inside our bootstrap CI (§18.2).

§2.6 qualifies everything in this subsection: the agreement below is one of
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

Three instruments — representations, cortex, behaviour — three data types, one
pattern.

### 2.4 Effective dimensionality is set by carrier bandwidth — and dissociates from shape bias


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
`parvo_highsf` leaves magno at **×1.00** exactly and doubles parvo; `noblur`
leaves parvo at **×1.00** exactly and raises magno. The carrier account does not
merely fit the ordering — it predicts which stream moves in each cell, correctly
four times out of four.

**Dimensionality is not a proxy for shape bias.** §2.1's probe factorises as an
interaction; this measure factorises additively, over *identical* inputs. The
decisive case is `parvo_highsf`, where the second stream becomes the **more
dimensional** one (0.56) while remaining **texture-biased** (probe −0.266). High
effective dimensionality does not confer shape bias, and we do not present these
as two views of one construct.

### 2.5 The shape bias does not confer robustness — a second dissociation


*Source: PAPER_PLAN §53 (pre-registered), §56. Reported in the main text, not
the supplement, because a reader who accepts §2.1 will otherwise assume the
robustness follows.*

The standard reading of Geirhos et al. is that shape bias buys out-of-
distribution robustness. We pre-registered that prediction and tested it:
corruptions applied to the image *before* the front end, scored in the models'
own 112-way label space, **3 seeds per condition** on all four cells of §2.1's
2×2.

Relative accuracy (accuracy under corruption ÷ clean accuracy), area under the
degradation curve:

| corruption | diet | no diet | swap | `parvo_highsf` |
|---|---|---|---|---|
| uniform noise | 0.687 | 0.729 | 0.745 | 0.767 |
| high-pass | **0.619** | 0.827 | 0.821 | 0.783 |
| phase-scrambling | 0.351 | 0.373 | 0.368 | 0.382 |
| rotation | 0.552 | 0.615 | 0.602 | 0.611 |
| false colour | 0.842 | 0.938 | 0.969 | 0.874 |
| *low-pass (trained-in)* | *0.441* | *0.356* | *0.349* | *0.386* |

**The diet-trained model is less robust on every corruption it was not trained
on**, in both statistics, with complete seed separation on four of five. The two
conditions without a shape bias (`swap`, `parvo_highsf`) pattern with the
no-diet model throughout, so this is not a quirk of one arm.

The **only** corruption on which the diet is more robust is low-pass — which we
declared trained-in before running, because the diet *is* a low-pass blur. The
experiment finds robustness exactly where robustness was trained in and nowhere
else, which is the strongest available evidence that the other five nulls are
real rather than a broken measurement.

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
accurate. Degradation does not produce the effect reported in §2.1; if anything
it dilutes it (PLAN §67, §68).

> **The claim.** Shape bias and OOD robustness are not intrinsically coupled.
> This does not contradict Geirhos et al., who showed that *Stylized-ImageNet
> training* produces models that are both shape-biased and robust. It shows the
> coupling is a property of that route, not of shape bias: a developmental route
> to comparable shape bias *in rate* (probe +0.819, behaviour 0.878 against a
> human 0.958 — though §2.6 shows the rate match does not extend to trial-level
> agreement)
> delivers **none** of the robustness. "Increase shape bias to get robustness"
> does not generalise across the manipulations that produce it.

This is the paper's second dissociation of properties the field treats as
coupled: §2.4 separates shape bias from representational dimensionality over
identical inputs, and the present section separates it from robustness.


---

### 2.6 Matching the human shape-bias rate is not matching human decisions — a third dissociation

*Source: PAPER_PLAN §57 (pre-registered), §59. Reported in the main text, not
the supplement, because it qualifies this paper's own headline comparison.*

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
title. §2.1's comparison against human observers is a comparison of aggregate
**rates**, and additionally compares a single stream to a whole observer; both
qualifications are now stated wherever that comparison appears.

### 2.7 What the diet costs


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
gradient could appear. So the correct statement of §2.3(b)'s control result is
**not** "the controls abolish the gradient" — it is that only the
diet-on-luminance configuration yields two streams that *both* carry substantial
unique cortical variance **and** a hierarchical gradient between them. The
controls do not relocate the diet within a fixed two-stream decomposition; they
destroy the decomposition. That is narrower than a clean control result and it is
what the data support.

**Integration loses shape bias; it does not create it.** The integrated readout's
bias (0.675) sits below the magno stream's (0.87–0.89). We therefore do not
write that integration *yields* shape-biased behaviour.

**The human comparison is directional, and now quantified.** Human observer data
is in hand (§52), so the gap is measured rather than gestured at: the best stream
sits **0.080 below the human mean** and below the lowest individual human, and
the integrated readout sits 0.283 below. The paper therefore states that the diet
moves the network most of the way from the texture-biased CNN regime toward human
observers and that the no-diet model stays in that regime, **with the numbers
given**. It does not claim a match, does not use "human-like", and does not
report any stream as falling in the human range.

## 3. Discussion

### What a developmental constraint actually does

The intuitive account of developmental low-pass input is informational: a learner
given coarse input first is protected from high-frequency texture, and so learns
shape. Our results do not support that account, and §2.2 is where it breaks. The
diet *removes* bandwidth rather than adding structure, and it *reduces* the
stream asymmetry of the input it is applied to. Yet networks trained under it end
up more asymmetric than what they were handed — including one that begins
parvo-dominant in 8/8 subjects and ends magno-dominant in every seed.

The constraint therefore acts on what learning converges to, not on what the
input supplies. This is a different kind of claim from the usual developmental
argument, and it makes a different prediction: what matters is not how much
information the early input carries, but which solutions it makes reachable.

We are careful about how far this generalises. What replicates across seeds is
the *side* of parity, not a level. Across three seeds a single condition's
trained ratio ranges over 2.39, and two readings we ourselves previously
advanced — a convergence "to within 2%", and a channel-specific set point — do
not survive that spread. A reader should take from §2.2 that the diet decides a
direction, and nothing about a destination.

### Against channel essentialism

The two-condition literature invites the reading that the magnocellular channel
*is* the shape pathway. §2.1 and §2.4 jointly rule that out. Shape bias requires
a carrier × transform conjunction: an achromatic carrier alone does not produce
it, and the transform alone does not produce it, and only their combination
crosses the boundary — by +0.936 with no seed overlap. Effective dimensionality,
meanwhile, needs only bandwidth, and reverses when the carrier moves.

Both properties are relational — facts about what a stream receives, not about
what it is called. This matters beyond our architecture, because "magno-like" and
"parvo-like" are routinely used as if they named intrinsic computational roles.

### Three dissociations, one lesson

Shape bias, representational dimensionality, out-of-distribution robustness and
human-likeness are routinely discussed as a package: a model that has more of one
is assumed to have more of the others. Over identical inputs, this paper
separates them three times.

Dimensionality and shape bias come apart in the direction opposite to intuition:
§2.4's `parvo_highsf` cell is more dimensional and more texture-biased, so richer
is not more shape-like. Robustness comes apart from shape bias in a way that is
sharpened by its own control: our diet-trained network is less robust than every
comparison on all five corruption families it was not trained on, and more robust
on exactly the one family the diet installs. And human agreement comes apart from
shape bias most cleanly of all — across six models spanning a 5× range of shape
bias, trial-by-trial agreement with human observers stays flat at 7–18% of the
human–human ceiling.

The lesson is not that shape bias is uninteresting. It is that **"shape bias" is
not a quantity a model has more or less of.** What produced it determines what
travels with it. Geirhos et al.'s stylisation route yields shape bias together
with robustness; our developmental route yields comparable shape bias in rate and
none of the robustness. Both are real; the coupling belongs to the route.

This has a direct methodological consequence. Shape bias is currently the most
cited human-alignment metric for vision models, and §2.6 shows it does not track
human decision structure at all. A model can be moved to the human *rate* without
moving toward human *behaviour*. Reporting error consistency alongside shape bias
should be standard, and is cheap wherever the human trial data exist.

### What generalises, and what does not

The side-of-parity result predicts that any manipulation changing what learning
converges to — rather than what the input supplies — should show the same
signature: a trained asymmetry that is insensitive to the input's asymmetry but
consistent in direction. Chromatic onset and acuity schedules are the natural
next tests, and a schedule-trained checkpoint from an independent codebase
already lands on the predicted side (§2.2).

What does not generalise is any recommendation to use a developmental diet as a
training method. It makes models less accurate (0.556 vs 0.701 at the best
saved checkpoint),
less stable late in training, and less robust on every corruption it was not
trained on. Nothing here is an argument that this is a
better way to train a network. Its interest is as a model of development.

### Limits

The cortical result (§2.3) is the least-replicated claim in the paper and is
flagged as such. The `parvo_highsf` condition varies architecture alongside the
front end. Our diet is a fixed 48-month front end rather than a trajectory. Human
comparisons are against published observers rather than an experiment of our own,
and are comparisons of aggregate rate. See *Limitations* for the full accounting.

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
transform from the trajectory — and its consequence is examined in §2.2 against
a schedule-trained checkpoint.

Four front-end conditions are used throughout:

| condition | stream 1 | stream 2 |
|---|---|---|
| `dkl_bcs` (diet) | luminance + blur + CS cut | L−M |
| `dkl_base` (no diet) | luminance, untransformed | L−M |
| `dkl_bcs_swap` | luminance, untransformed | L−M **+ blur + CS cut** |
| `dkl_bcs_parvo_highsf` | luminance + blur + CS cut | L−M **+ untransformed luminance** (2 ch) |

`parvo_highsf` gives stream 2 two input channels rather than one; this
architectural difference is measured rather than argued away (§2.1, capacity
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
of §2.1 and §2.2.

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

**Error consistency** (§2.6). Cohen's kappa on the binary shape/texture choice
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
and never pooled (§2.2).

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
in §2.3, computed by pushing the cue-conflict set through each fitted encoding
model.

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

**Pre-registration.** The front-end 2×2 (§2.1), the inverted-front-end prediction
(§2.2), the OOD families (§2.5) and the error-consistency hypothesis (§2.6) were
each written down before the corresponding analysis was run. Two of the four
predictions failed and are reported as failures in the sections above.

---

## Limitations

1. **The diet costs, on every engineering axis we measured.** ~13% of explained
   BOLD variance (§2.7); 20.7% relative classification accuracy at the best
   saved checkpoint (§2.5); late-training stability (§2.5); and robustness on
   every corruption it was not trained on (§2.5). The paper makes no claim that
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
   confound that would fake the result is ruled out (§2.1), but the comparison
   is not architecture-matched. This qualifies §2.2's decisive case and must
   travel with it.
4. **There is no set point, and the trained level is not reproducible across
   seeds.** Earlier drafts of this work reported the two diet-on-luminance arms
   as converging to within 2% (a gap of 0.037), and then that the level was
   channel-specific (red−green at 2.225 against luminance at 1.603). The seed
   replication retires both: the within-arm seed range on the luminance arm is
   **0.904**, ~24× that gap, and the red−green and luminance ranges overlap
   (1.995–4.383 vs 1.603–2.507), so n=3 seeds cannot establish
   channel-specificity either. The paper therefore reports **no level, no set
   point and no compression factor anywhere** — only the side of parity, which
   replicates 12/12 seeds with complete separation (§2.2, PLAN §58.3).
5. **The independent thesis checkpoint lands at 1.158** — below every one of our
   diet seeds (1.497–4.383) but on the same side of parity, in 8/8 subjects. It
   is a different codebase and a real age schedule rather than our fixed age 48,
   feature-extracted through a schedule that is not its own, so it is the least
   comparable arm. It is reported as a convergent check on the *side* claim, and
   as the paper's only evidence that a genuine developmental trajectory behaves
   like our static front end (§2.2, PLAN §58.5).
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

---

## Claims deliberately not made

Recorded so the omissions are visible rather than silent.

- **Where in visual cortex the diet's shape-bias effect lands.** We tested
  whether the per-voxel predicted shape-bias profile carries a diet-specific
  hV4/LO step. Under the full control set it does not (PLAN §43.2, §44.1). We
  therefore make no claim about cortical localisation of the shape bias, and the
  analysis is reported in supplementary material rather than the main text. Note
  this is a *different* measure from §2.3(b)'s stream-bias gradient, which is
  diet-specific and survives its controls.
- **That integration produces human-like shape bias** (§2.7).
- **That the diet improves neural predictivity** — it does not (§2.7).

---

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
      null (§2.6) is stated in the same paragraph.
- [x] **Figure 1 built** — `results/figures/fig1_convergence.{png,pdf}`, caption
      in `fig1_convergence_caption.md`, built by `tmp_analysis/fig1{,_data}.py`
      (PLAN §51). Panel (c) is the set point as a front-end→trained slopegraph,
      superseding §46.3's retention spec per §49.
- [x] **Ventral-gradient figure renumbered** (2026-09-03) —
      `fig1_diet_gradient.{png,pdf}` → `fig2_ventral_gradient.{png,pdf}`,
      caption → `fig2_ventral_gradient_caption.md`. It is §2.3(b)'s evidence;
      Figure 1 is the four-panel convergence figure (PLAN §51.4).
- [x] **Venue decided: NHB** (2026-09-02). NeurIPS/NMI dropped — there is no ML
      payoff to report, only costs on accuracy, stability and robustness
      (PLAN §56.6). The human-facing claims are what carry the paper.
- [x] **Methods expanded** (2026-09-03) — §4.1–4.7, every parameter read from
      the code and run scripts rather than from memory; 3 seeds per condition
      stated, and the subject vs seed inferential axes separated in §4.7.
- [ ] Supplementary: the VSB cortical profile with its negatives intact; the
      capacity ladder (PLAN §42) with its non-monotonicity caveat.
- [x] **Cortical seed replication complete** (2026-09-04) — array 26416428,
      32/32 COMPLETED. §2.3 replicates across three seeds with both statistics
      agreeing (diet seed range 0.032). **The paper now has no n=1 claim in its
      main results** (PLAN §66).
