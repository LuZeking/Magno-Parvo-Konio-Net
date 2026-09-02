# A developmental visual diet installs functional specialisation in two-stream networks

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
statistics of that input change over development. Two-stream artificial networks
trained on biologically shaped input channels have been reported to develop
specialised streams, but whether specialisation is *inherited* from the input or
*produced* by learning has not been separable, because in every previous
comparison the input already carried the asymmetry being measured.

We resolve this using two-stream networks trained under a developmental visual
diet — a progressive low-pass transform applied to the achromatic channel — and
evaluated against eight subjects of the Natural Scenes Dataset.

First, the resulting shape-biased stream is not a property of the achromatic
channel at all. It requires the **conjunction** of an achromatic carrier and the
developmental transform: neither factor alone moves a stream across the
shape/texture boundary (probe bias −0.266 and −0.155 for the single factors,
**+0.819** for the conjunction).

Second, we build a front end whose stream asymmetry is *inverted* and train a
network on it. The diet does not preserve whatever asymmetry the input supplies
— it **installs one**. Two networks sharing a training recipe and seed, handed
front ends 2.39× apart in effective-dimensionality ratio (1.402 and 0.586, the
latter inverted in 8/8 subjects), converge after training on 1.603 and 1.640: a
front-end gap of 0.816 becomes 0.037, a 22-fold compression, and the inverted
case flips in every subject. Networks trained without the diet converge instead
on parity (0.947, 0.988) from the *most* magno-dominant front end in the set
(1.634).

Third, the divergence is measurable through human visual cortex. Passing
cue-conflict images through per-stream voxelwise encoding models yields a stream
dissociation of **+1.534** with the diet and **+0.110** without (p=.0078, 8/8
subjects), and the diet-trained network's
stream-level division of labour tracks the human ventral hierarchy (hV4−V1v
−0.862) where the no-diet network's does not (−0.117).

The diet's effect is therefore not a filtering artefact and not a property of any
privileged channel. It is a constraint on what learning converges to.

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
developmental diet has been reported to develop a shape-biased stream — but the
mechanism has been ambiguous between two readings that no previous comparison
could separate:

- **Inheritance.** The transform changes the input's statistics; the network
  merely reflects what it is handed.
- **Installation.** The transform changes what learning converges to,
  irrespective of what the input hands over.

These make identical predictions whenever the input asymmetry and the trained
asymmetry point the same way — which was true of every condition previously
tested. Distinguishing them requires a front end whose asymmetry points the
*other* way. We build one and use it as the decisive case.

**Contributions.**

1. A four-cell causal decomposition of the input factors (achromatic carrier ×
   developmental transform) showing the shape bias is an **interaction**, not a
   property of either factor (§2.1).
2. Evidence that effective dimensionality across those same four front ends is
   explained entirely by carrier bandwidth, is reproducible with **no network at
   all**, and **dissociates** from shape bias (§2.2).
3. The inheritance/installation test, using an inverted front end: the diet
   **installs** a representational set point (§2.3).
4. Confirmation of the stream divergence through human cortex, on independent
   machinery (§2.4).
5. An honest accounting of what the diet costs (§2.5).

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

Linear probe, stream shape−texture bias (zero is the sign flip; n=2 seeds
agreeing to 0.017):

| | no transform | with transform |
|---|---|---|
| **chromatic carrier (L−M)** | −0.504 | −0.155 |
| **achromatic carrier (lum)** | **−0.266** | **+0.819** |

Behavioural cue-conflict shape bias on the same four cells replicates it
independently (0.364 / 0.484 / 0.396 / **0.878**).

**Both single factors move a stream substantially toward neutral and neither
crosses zero. Only the conjunction crosses, and it goes all the way.** This is an
interaction, measured on all four cells rather than inferred from two, and it is
the strongest causal statement in the paper about *what* produces the shape bias:
the developmental low-pass transform applied specifically to an achromatic
channel.

*Capacity control.* `parvo_highsf` gives its second stream an extra input
channel. The confound that would matter is if the extra channel *weakened* that
stream and faked the null. It does not: 16-way accuracy rises to 0.486 from
~0.379 in the standard diet models, a ~28% relative gain (§25.2). The extra
channel helped, and the stream still did not become shape-biased.

### 2.2 Effective dimensionality is set by carrier bandwidth — and dissociates from shape bias

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

### 2.3 The diet installs a representational set point — it does not preserve the input's

*Source: PAPER_PLAN §45, §49. This is the paper's mechanistic core.*

§2.2 establishes that different front ends hand a network very different stream
asymmetries before any learning. What does training do with them?

The first-order answer, from the two conditions available previously, was that
the diet makes training *keep* the asymmetry the input supplies while no-diet
training destroys it. That description is arithmetically correct and cannot be
right, because in both of those conditions the input asymmetry pointed the same
way — so *keep* and *install* predicted the same outcome.

`parvo_highsf` separates them. Its front end is inverted (0.586, below 1.0 in 8/8
subjects). We pre-registered the prediction that a diet-trained network on that
front end would stay inverted. **It does not.**

| arm | model | front end | after training |
|---|---|---|---|
| diet | `dkl_bcs_seed0` | 1.402 ±0.09 | **1.603** ±0.032 |
| diet | `dkl_bcs_parvo_highsf_seed0` | **0.586** ±0.013 | **1.640** ±0.030 |
| no diet | `dkl_base_seed0` | 1.634 ±0.118 | 0.947 ±0.035 |
| no diet | `dkl_base_seed1` | 1.634 ±0.118 | 0.988 ±0.040 |

The two diet models share a training recipe and a seed and are handed front ends
**2.39× apart** (p=.0078). After training they differ by 0.037 — a **22×
compression** of the input gap. The inverted case flips completely: 8/8 subjects
below 1.0 before training, **0/8 after**.

Meanwhile the no-diet models, handed the *most* magno-dominant front end in the
set (1.634), converge on parity — 0.947 and 0.988, i.e. two streams of equal
effective dimensionality. Every diet × no-diet contrast on the trained ratio is
p=.0078, 8/8 subjects.

> **The claim.** A developmental visual diet does not preserve the stream
> asymmetry its front end supplies. It installs one. Diet-trained networks
> converge on a magno-dominant geometry from front ends that are magno-dominant
> *and* from front ends that are parvo-dominant; no-diet networks converge on
> parity from a front end that is strongly magno-dominant. **What training ends
> with is set by the diet, not by the input.**

This is what rules out the filtering account. The diet is not supplying a better
input representation — §2.2 shows it *removes* bandwidth, and the no-diet front
end hands over more asymmetry, not less. It is a constraint on the solution
learning converges to.

*The reference here is exact.* `untrained_parvo_highsf` carries provenance
identical to the trained model on every recorded key except the checkpoint, so
this is a matched front-end pairing rather than an approximate one (§49.2).

### 2.4 The divergence is measurable through human visual cortex

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
  p=.0078 on all three (§48.1). See §2.5 for what those controls cost.
- **It is not explained by a global stream preference.** The diet-trained and
  no-diet models sit on *opposite* sides of the global stream-bias axis (−0.454
  vs +0.727), yet the strong regional gradient is in the model whose global bias
  points the other way (§17.2).

**(c) Behaviour.** The integrated readout's cue-conflict shape bias is
**0.675 ±0.025** with the diet against **0.416 ±0.020** without, across three
seeds each. The thesis checkpoint reproduces at 0.697 against its reported 0.677,
inside our bootstrap CI (§18.2).

Three instruments — representations, cortex, behaviour — three data types, one
pattern.

### 2.5 What the diet costs

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
gradient could appear. So the correct statement of §2.4(b)'s control result is
**not** "the controls abolish the gradient" — it is that only the
diet-on-luminance configuration yields two streams that *both* carry substantial
unique cortical variance **and** a hierarchical gradient between them. The
controls do not relocate the diet within a fixed two-stream decomposition; they
destroy the decomposition. That is narrower than a clean control result and it is
what the data support.

**Integration loses shape bias; it does not create it.** The integrated readout's
bias (0.675) sits below the magno stream's (0.87–0.89). We therefore do not
write that integration *yields* shape-biased behaviour.

**No human comparison is made.** Human observer data for these stimuli is not in
our possession, so the paper makes a *directional* statement — the diet moves the
network toward the human end of the shape/texture axis and the no-diet model sits
in the texture-biased CNN regime — and **no matching statement**. No human number
appears anywhere in this manuscript.

---

## 3. Discussion

> DRAFT — points fixed, prose to be written.

- **Reframing what a developmental constraint does.** The intuitive account of
  developmental low-pass input is informational: coarse input first means the
  learner is protected from high-frequency texture. §2.3 is not that. The diet
  *removes* bandwidth and *reduces* the input's stream asymmetry, and the network
  ends up more asymmetric than its input — converging to the same place from an
  inverted start. The mechanism is about the attractor, not the diet.
- **Against channel essentialism.** §2.1 and §2.2 jointly rule out "magno is the
  shape pathway". Shape bias needs a carrier × transform conjunction; effective
  dimensionality needs only bandwidth and reverses when the carrier moves. Both
  properties are relational, not intrinsic to a labelled channel.
- **Two properties that look alike and are not.** Dimensionality and shape bias
  are frequently discussed as though richer representations were more
  shape-like. §2.2's `parvo_highsf` cell is a direct counterexample: the more
  dimensional stream is the texture-biased one.
- **What generalises.** The set-point result predicts that any manipulation
  changing *what learning converges to*, rather than what the input supplies,
  should show the same compression signature. That is a testable prediction for
  other developmental manipulations (chromatic onset, acuity schedules).
- **Limits.** See §2.5 and *Limitations*.

---

## 4. Methods

> OUTLINE — to be expanded from `PAPER_PLAN.md` §1 (A1) and the run logs.

- **Architecture.** Two-stream ResNet-18 backbones, DKL front end
  (RGB→XYZ(D65)→LMS(HPE)→DKL). Magno = luminance + blur σ=4 + 48-month CSF cut;
  parvo = L−M untransformed. Note and handle the legacy asymmetric-mixing forward
  (`src/models/pytorch_models.py`) for thesis checkpoints (PLAN §0).
- **Front-end conditions.** `blur_cs` (diet), `noblur` (no diet), `swap`
  (transform on the chromatic channel), `parvo_highsf` (stream 2 receives lum+rg).
- **Training.** MiniEcoset 256px, 112 classes. Seeds as reported per condition.
- **Behaviour.** Geirhos cue-conflict, 1,200 images, 16-way. Bootstrap CIs.
- **Probe.** Linear shape−texture probe on stream features.
- **Participation ratio.** (Σλ)²/Σλ² on the feature covariance, computed
  Gram-side, 1,000 NSD images per subject.
- **Encoding.** NSD b3 betas, PSC + per-session voxel z-scoring, `nsdgeneral`
  mask; split by 73k image id, test = shared-515; banded ridge (himalaya) with
  GroupKFold by image id; noise ceiling from NSD `ncsnr`; stream-wise variance
  partitioning.
- **Statistics.** Paired exact Wilcoxon within subject, n=8 (exact floor
  p=.0078). **Two-statistics rule:** every claim is made only where the
  normalised and raw statistics agree; disagreements are reported as unresolved
  (PLAN §8).

---

## Limitations

1. **The diet costs ~13% of explained BOLD variance** (§2.5). Stated in the main
   text, not buried.
2. **Two front-end controls are degenerate on the cortical instrument** (§2.5),
   which narrows the control claim.
3. **`parvo_highsf` varies architecture alongside the front end** — its second
   stream has two input channels rather than one, so input and first-layer
   fan-in change together and no existing model separates them. The capacity
   confound that would fake the result is ruled out (§2.1), but the comparison
   is not architecture-matched. This qualifies §2.3's decisive case and must
   travel with it.
4. **The set point is per-recipe, not universal.** The independent thesis
   checkpoint converges to 1.158 rather than ~1.62 on the same front end; it is a
   different codebase and diet schedule and was feature-extracted through a
   schedule that is not its own. The tight-convergence claim is made for the two
   matched models; the diet-vs-no-diet claim holds across all three (§49.4).
5. **The two diet arms are not statistically identical** (1.603 vs 1.640,
   p=.0391). The claim is "converge to within 2%", never "identical".
6. **n=8 is subjects-as-image-samples for the PR analyses** — participation ratio
   is computed on each subject's cached image set, so the paired test speaks to
   stability across image samples, not to population inference. The encoding and
   behavioural results are not affected.
7. **No human observer data**, hence no human-likeness claim (§2.5).
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
  this is a *different* measure from §2.4(b)'s stream-bias gradient, which is
  diet-specific and survives its controls.
- **That integration produces human-like shape bias** (§2.5).
- **That the diet improves neural predictivity** — it does not (§2.5).

---

## Open items before submission

- [ ] Obtain Geirhos human observer data, or keep every human reference
      directional. **Blocks any use of the word "human-like".**
- [ ] Figure 1: four-panel convergence figure (PLAN §46.3) — panels (a) input
      2×2, (b) in-silico cue conflict, (c) the set point (§2.3's table as paired
      per-subject lines, front-end reference row shown), (d) the variance cost.
      Panel (d) is not optional.
- [ ] Decide venue; PLAN targets NMI / NeurIPS tier.
- [ ] Expand Methods from PLAN §1 and the run logs; add per-condition seed counts.
- [ ] Supplementary: the VSB cortical profile with its negatives intact; the
      capacity ladder (PLAN §42) with its non-monotonicity caveat.
