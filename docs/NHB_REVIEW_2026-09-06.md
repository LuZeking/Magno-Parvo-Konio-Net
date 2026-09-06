# NHB Referee Simulation — `docs/PAPER_DRAFT.md`

Run 2026-09-06 against the draft at commit `107ba04`. Three-persona simulation per
`~/.claude/skills/nhb-reviewer`. **The numeric rubric is internal to the simulation and is
not an NHB form.**

## NHB Review Summary

- **Title:** Shape bias requires a conjunction of achromatic carrier and developmental transform
- **Article type:** Article
- **Triage:** **discuss** — §2.6 is a genuine meta-science contribution, but the paper's
  stated conclusion is about what makes a CNN shape-biased, and every human-facing claim is
  framed as a disclaimer rather than a finding.
- **Breadth test:** cognitive neuroscience (visual) + computational vision modelling —
  **effectively a single constituency.** Developmental psychology is reachable but not
  claimed, and the front end is static rather than a trajectory.
- **Predicted Decision:** **Reject, suggest transfer** (Nature Communications; eLife;
  Communications Psychology)
- **Recommendation (avg of 3):** 2.7 / 5 · **Weighted simulation score: 5.9 / 10**
- **Significance gate (NHB-caliber?):** borderline — R1 borderline, R2 no, R3 borderline
- **Confidence:** 3.7 / 5
- **Ethics Flag:** **yes** — no ethics approval, consent, secondary-use, demographic, or
  constraints-on-generality statement for either human dataset (NSD n=8; Geirhos observers n=10)
- **Preregistered:** claimed in §4.7, **not verifiable** — no registry, date, or link
- **Estimated revision time:** 3–6 months · **New data required:** no

## Per-axis Scores (1–4) — simulation rubric, not an official NHB form

| Axis | R1 | R2 | R3 | Avg |
|---|---|---|---|---|
| Advance | 3 | 2 | 3 | 2.7 |
| Soundness | 3 | 3 | 3 | 3.0 |
| Evidence | 3 | 3 | 2 | 2.7 |
| Relevance | 2 | 1 | 2 | 1.7 |
| Statistics | 3 | 3 | 2 | 2.7 |
| OpenScience | 2 | 2 | 2 | 2.0 |
| Ethics | 2 | 2 | 2 | 2.0 |
| Clarity | 2 | 1 | 2 | 1.7 |
| Recommendation (/5) | 3 | 2 | 3 | 2.7 |
| Confidence (/5) | 4 | 3 | 4 | 3.7 |

Hard overrides checked: the undefined-error-bar item and the EthicsFlag each **cap** the
decision at Major revision; neither is binding because the weighted score already sits below
that band. No override raises the outcome.

## Consensus Strengths

1. **The 2×2 is a real causal manipulation, not an observational comparison** (§2.1). Two
   of the four front ends were built specifically to break the confound, and the
   conjunction cell separates from all three others by +0.936 with no seed overlap.
2. **The inheritance-vs-installation test is well designed** (§2.2). Building a front end
   whose asymmetry points the wrong way (0.586, below parity in 8/8 subjects) and watching
   training overturn it (1.640) is the right experiment for the question, and it is the
   kind of design most modelling papers do not attempt.
3. **Confirmatory/exploratory separation is exemplary** (§4.7, §2.5, §2.6). Four
   preregistrations named per section, **two of four predictions failed and are reported as
   failures**, and *Claims deliberately not made* records six retired analyses including
   two the authors had previously promoted. This is rarer than it should be.
4. **The two-statistics rule is a genuine robustness discipline** — claims made only where
   raw and noise-ceiling-normalised statistics agree, with disagreements (phase-scrambling,
   hMT+, pooled VO unmatched) left unresolved rather than resolved favourably.
5. **Nesting is handled correctly** (§4.7). Subjects and seeds are never pooled, and a
   seed-level claim requires the within-seed subject test to hold in all three seeds
   independently. Limitation 6 correctly flags that the participation-ratio n=8 is
   subjects-as-image-samples, not population inference.
6. **Out-of-sample discipline in the encoding models** (§4.5): PCA fit on training images
   only, splits by 73k image id, shared-515 held out, never used for selection.

## Consensus Weaknesses

1. **No References section — zero citations.** The manuscript names Geirhos, Sincich, and
   Kastner2015 in running text with no reference list. A referee cannot assess novelty,
   priority, or attribution, which is R1's core function. This alone is a return-to-authors
   condition at any Nature-portfolio journal.
2. **`±` is never defined anywhere in the manuscript.** It carries the §2.2 headline table
   — the section the paper calls "the paper's mechanistic core" — plus §2.4's table and the
   human benchmark (0.958 ±0.019). Statistical-rigor item 16; the checklist lists undefined
   headline error bars as fatal.
3. **The preregistration is unverifiable.** §4.7 states the predictions "were each written
   down before the corresponding analysis was run." No registry, no timestamp, no link.
   The word "pre-registered" appears eleven times and is the paper's principal credibility
   asset; none of it can be checked by a referee.
4. **~40 subject-level significance claims (35 × "8/8", 5 × "0/8"), ~44 reported p-values,
   and 2 reported effect sizes.** No family is defined and no FWER or FDR control is stated
   anywhere. The disclosed dead claims (hMT+, eccentricity, dorsal/ventral, VO) confirm a
   screening design was run. The two-statistics rule is a robustness check, not multiplicity
   control.
5. **All three headline dissociations are nulls asserted without equivalence tests or Bayes
   factors.** §2.6 ("flat at 7–18% of ceiling"), §2.5 phase-scrambling (p = 0.70), and the
   VO failure (p = .46) are the paper's selling points, and item 20 of the checklist
   explicitly forbids stating a null as "no effect" without an equivalence bound.
6. **The behaviour/cognition link required by NHB scope is absent as a positive claim.**
   The conclusion is about a CNN. Human data serve as a benchmark (§2.1, §2.3c), an
   instrument (§2.3), and the target of a null (§2.6). §2.6 removed "human-like" from the
   title; Limitation 7 states the human comparison is "directional only."
7. **No ethics, consent, secondary-use, demographic, or constraints-on-generality statement**
   for NSD (8 subjects) or the Geirhos observers (10). The Geirhos licence is unverified by
   the draft's own Open Items.
8. **No data availability, code availability, materials, or Reporting Summary statements.**
9. **Format overruns.** Main text 7,956 words against NHB's ≈4,000–6,000; abstract 636 words;
   10 tables + 2 figures = 12 display items against a ≤8 cap.
10. **The developmental framing outruns the manipulation.** The title says "developmental
    transform" and the Discussion is headed "What a developmental constraint actually does,"
    but the front end is static at a fixed 48 months (§4.1). The one trajectory-trained
    checkpoint is a different codebase and the authors call it "the least comparable arm."
11. **§2.2's mechanistic core rests on one trajectory per arm.** The two arms are on the
    *same* side of parity at epochs 75–100; the dissociation is carried by a late change in
    the control arm. Disclosed, and the disclosure is to the authors' credit, but the
    installation claim is n=1 on the only measurement that could establish it.
12. **`parvo_highsf` is not architecture-matched** and it is the decisive case for §2.2
    (Limitation 3).
13. **Post-hoc exclusion** of `parvo_highsf` seed 0 from the behavioural row (§2.1),
    disclosed but not pre-specified.
14. **Anatomical labels do rhetorical work the model does not support.** "Magno" and "parvo"
    here are luminance and L−M inputs to two identical ResNet-18s; temporal dynamics,
    contrast gain, receptive-field size, and cortical targets are not modelled. §3's
    "Against channel essentialism" argues this point while the labels persist throughout.

## Author Questions

1. Where is the preregistration deposited, and what is the timestamp relative to first data
   access for each of the four predictions in §4.7? If the records are lab-internal, can
   they be deposited with hashes and access dates so a referee can verify ordering?
2. What does `±` denote in the §2.2 and §2.4 tables and in the 0.958 ±0.019 human value —
   SD across subjects, SEM, or a bootstrap interval? The §2.2 table is the paper's
   headline and the quantity is not defined in text, legend, or Methods.
3. §2.6's conclusion is that shape bias does not track human decision structure. What is the
   equivalence bound, or Bayes factor, that licenses accepting that null over six models —
   and what κ difference would the design have detected at n = 10 observers?
4. How many statistical tests does the manuscript report in total, what family do they
   belong to, and what multiplicity control applies? The retired hMT+, eccentricity,
   dorsal/ventral, and VO analyses indicate a screen over ROIs.
5. §2.2's two arms are on the same side of parity through epoch 100 and separate only in the
   final quarter. With one trajectory per arm, what distinguishes "the diet stabilises which
   side a network sits on" from "the control arm drifts late for reasons unrelated to the
   diet"? Would the cached checkpoints for seeds 1–2 settle it without new training?

## Statistical Rigor Audit

| Item | Verdict | Note |
|---|---|---|
| Sample size justified | **Fail** | No power analysis, no smallest effect of interest. n=3 seeds has an exact two-sided floor of p=0.100, so no seed-level test can reach conventional alpha by construction; §4.7 states this but does not justify the choice. |
| n and unit of analysis defined; nesting handled | **Pass** | §4.7 separates subjects from seeds, refuses to pool, and requires within-seed replication in all three. Limitation 6 correctly bounds the PR n=8. Best-in-class. |
| Test choice, assumptions, one/two-sided | **Pass** | Exact Wilcoxon signed-rank paired within subject; exact Mann–Whitney for 3v3; two-sided stated. Appropriate non-parametric choices at n=8. |
| Multiple comparisons declared and corrected | **Fail** | ~44 p-values and ~40 subject-count claims; no family defined, no FWER/FDR anywhere. Screening confirmed by the four disclosed dead claims. |
| Effect sizes + CIs; error bars defined | **Fail** | 2 effect sizes (dz) against ~40 claims. Almost no CIs outside the cue-conflict bootstrap. `±` undefined throughout, including on the §2.2 core table. |
| Exact p / test statistics / df / software | **Partial** | Exact p reported (.0078, .0391, .1094, .46) — good. Test statistics and df never reported; no software or package versions. |
| Robustness: sensitivity, multiverse, out-of-sample | **Pass** | Two-statistics rule and the best-saved-vs-final checkpoint policy form a genuine two-axis sensitivity analysis, reported with disagreements intact. Out-of-sample handling in §4.5 is clean. |
| Causal language matched to design | **Partial** | Warranted for the model (front ends were manipulated, seeds assigned). Not warranted for the developmental reading, which the static 48-month front end cannot support. |
| Measurement validity / reliability | **Pass** | Noise ceiling from `ncsnr`; leave-one-subject-out ROI-profile ρ = +0.794 (.589–.896); seed ranges reported per instrument. |
| Null results reported with equivalence/BF | **Fail** | All three dissociations are nulls; none has an equivalence test or Bayes factor. |

## Open Science Audit

- **Preregistration:** claimed for four analyses (§4.7); **no registry, date, or link**;
  ordering unverifiable. Deviations: §2.5's H2 refusal and the checkpoint-policy change are
  disclosed in text — good practice, but against an unverifiable plan.
- **Data:** no statement. NSD and the Geirhos release are public but neither is cited, versioned,
  or given an access statement. Geirhos licence unverified by the authors' own Open Items.
- **Code:** no statement. Methods name source files (`src/training/standalone_trainer.py` etc.)
  but there is no repository, DOI, or environment manifest.
- **Materials:** MiniEcoset and the cue-conflict set named, not given access details.
- **Reporting Summary (BSS):** not provided.
- **Replication:** internal (3 seeds per condition) **Pass**; cross-codebase convergent check
  (−0.8618 vs −0.8593) **Pass**; no independent human sample.

## Human Participants and Human Groups Audit

- **Ethics approval / consent / compensation:** **absent for both datasets.** Secondary use of
  existing human data still requires a statement naming the original approvals.
- **Sample demographics:** none. NSD subjects' age and sex are not reported; the 10 Geirhos
  observers are described only by count.
- **Constraints-on-generality:** **absent.** The manuscript says "human observers" and "human
  visual cortex" from n=10 and n=8 single-lab samples with no scope statement.
- **Sex/gender and race/ethnicity:** not analysed; **n/a**, but the absence should be stated.
- **Risk of harm or stigmatisation:** **none** — no group-level claims.
- **Vulnerable populations / sensitive data:** n/a. Neuroimaging incidental-findings protocol
  is NSD's, and should be cited as such.

Everything in this section is fixable in roughly one paragraph of Methods plus a statements
block. The flag reflects absent text, not misconduct.

## Specific Improvement Suggestions (ranked)

1. **Add the References section.** ~50–70 checked citations. Nothing else can be assessed
   until this exists. (≈1 week; no new data)
2. **Define `±` and add CIs + effect sizes to every headline.** dz and a bootstrap or
   analytic CI on the §2.1, §2.2, §2.3 headlines. (≈2 days; no new data)
3. **Deposit the preregistrations** with timestamps, or restate them as "analysis plans
   fixed in advance and recorded in the project log at <commit hashes>" — the git history
   can substantiate ordering, which is stronger than the current bare assertion.
   (≈1 day; no new data)
4. **Add equivalence tests (TOST) or Bayes factors to the three dissociations.** This is the
   single highest-value statistical addition: it converts §2.4/§2.5/§2.6 from "we did not
   detect a difference" into "we bounded the difference below X." (≈2–3 days; no new data)
5. **Add a multiplicity statement** defining the confirmatory family and applying FDR within
   it, with the pre-declared contrasts identified as such. (≈1 day; no new data)
6. **Add the ethics/statements block:** ethics approval (NSD's and Geirhos's, cited),
   consent, data availability, code availability, materials, Reporting Summary,
   constraints-on-generality, author contributions, competing interests. (≈1 day)
7. **Cut main text to ≈5,000 words.** The six self-correction passages in Results
   (the 0.904→0.120 checkpoint artefact, 28-fold→3.7-fold, 37%→20.7%) are project history,
   not results; moving them wholesale to Supplementary recovers ~1,500–2,000 words without
   removing a single real limitation. (≈3 days)
8. **Reduce display items from 12 to ≤8**; build figures for §2.1's 2×2, §2.5's degradation
   curves, and §2.6's error-consistency scatter, converting tables to panels. (≈1 week)
9. **Add §72's single-stream control.** "Why two streams at all?" is the first question a
   referee asks of a two-stream paper and the draft does not answer it; the result exists
   (split buys 0.088 shape bias, costs 0.043 accuracy, half the parameters, complete
   separation under both checkpoint policies). (≈1 day; no new data)
10. **Reframe for scope, or transfer.** See the editor section.

---

## Comments to Authors

### Summary of key messages

This manuscript asks what produces the shape bias in two-stream networks trained on
biologically shaped channels, and answers it with a 2×2 crossing of carrier against
developmental transform. The central result is that neither factor alone moves a stream
across the shape/texture boundary and their conjunction moves it decisively; a second
result uses a purpose-built front end with an inverted asymmetry to show that training does
not preserve what the input supplies but does fix which side of parity it ends on; a third
reproduces the stream divergence through voxelwise encoding models on eight NSD subjects;
and three further analyses dissociate shape bias from effective dimensionality, from
out-of-distribution robustness, and from trial-level agreement with human observers.

The experimental design is the manuscript's strongest feature. Building the two missing
cells of the 2×2, and in particular building a front end whose asymmetry points the wrong
way, converts a comparison the prior literature could only make observationally into a
causal one. The confirmatory/exploratory discipline is likewise better than is typical: four
preregistered predictions, two of which failed and are reported as failures, and a
*Claims deliberately not made* section that retires six analyses including two the authors
had previously advanced. The manuscript's principal shortcomings are that the statistical
reporting does not match the quality of the design, that the artefacts needed to verify the
preregistration and reproduce the analyses are absent, and that the conclusion as stated is
about a convolutional network rather than about human behaviour or cognition.

### Major comments

**1. The manuscript contains no reference list.** Prior work is named in running text
(Geirhos; Sincich; Kastner) without citations, and there is no References section. I cannot
evaluate novelty, priority, or attribution, and I cannot confirm that the two-condition
literature the paper positions against says what the paper says it says. This must be
supplied before substantive review of the claims is possible.

**2. `±` is never defined.** It appears in the §2.2 table you describe as "the paper's
mechanistic core" (1.402 ±0.089 → 1.703 ±0.059), in §2.4's table, and on the human benchmark
(0.958 ±0.019). Neither the text, the tables, nor Methods §4.4/§4.7 states whether this is a
standard deviation across the eight subjects, a standard error, or a bootstrap interval. The
inference differs materially between them, and §2.2's argument turns on comparing these
intervals against a 3.26× spread in front ends.

**3. Multiplicity is not addressed.** By my count the manuscript reports approximately
forty-four p-values and forty subject-count claims of the form "8/8" or "0/8". No family of
tests is defined and no family-wise or false-discovery control is applied. The four analyses
retired in *Claims deliberately not made* — hMT+, the eccentricity headline, the
dorsal/ventral interaction, and the VO prediction — establish that a screen across regions of
interest was run. Your two-statistics rule is a valuable robustness constraint but it is not
a multiplicity correction: two statistics computed from the same fits are not independent
tests. Please define the confirmatory family, identify which contrasts were declared in
advance, and apply FDR control within it.

**4. The three dissociations are nulls stated without an equivalence bound.** §2.4, §2.5, and
§2.6 are the manuscript's most-promoted contributions and each rests on failing to reject.
§2.6 concludes that shape bias "does not track human decision structure" from six models
whose κ values span 7–18% of the human–human ceiling; §2.5 reports phase-scrambling as
"indistinguishable (p = 0.70)"; the VO analysis reports p = .46. None carries a TOST
interval or Bayes factor. Please state, for each, what difference the design could have
detected and what interval the data exclude. This is the change that would most strengthen
the manuscript, and it requires no new data.

**5. The preregistration cannot be verified.** Methods §4.7 states the four predictions "were
each written down before the corresponding analysis was run," and the term "pre-registered"
appears eleven times, including on the §65.4 gate and the §69.3 fatal-outcome declaration.
No registry, date, or link is given. Because the ordering of prediction and analysis is
what makes these statements load-bearing, please deposit the records with timestamps, or
substantiate ordering by reference to dated commits.

**6. The conclusion is about a network, not about human behaviour or cognition.** The title
and the Discussion's organising claim concern what makes a two-stream CNN shape-biased. The
human data enter as a benchmark for aggregate rates (§2.1, §2.3c), as a measurement
instrument (§2.3), and as the target of a null (§2.6). §2.6 removed "human-like" from the
title and Limitation 7 states the human comparison is "directional only." I note this not as
a criticism of the science but because it determines which readership the manuscript serves.
If a claim about human vision or human development is intended, it needs to be made and
supported; if not, the framing should say so plainly.

**7. The developmental framing outruns the manipulation.** The title names a "developmental
transform" and the Discussion is headed "What a developmental constraint actually does," but
Methods §4.1 fixes the front end at 48 months and applies no schedule. The single
trajectory-trained comparison (`thesis_dvd_trans`) is from a different codebase, feature-
extracted through a schedule that is not its own, and is described by you as "the least
comparable arm." Please either soften the developmental language to match a static acuity
manipulation, or add a schedule-trained arm in this codebase.

**8. §2.2's central dissociation rests on a single trajectory per arm, and the two arms do
not differ for most of training.** You report that the no-diet arm sits above parity through
epoch 100 (1.16, 1.24, 1.51, 1.51) and crosses only in the final quarter, and you state
plainly that "at epochs 75–100 the two arms are on the same side of parity and there is no
dissociation." Reporting this is to your credit. But the installation claim — the
manuscript's mechanistic core — is what the trajectory would establish, and it is n=1 there.
If cached checkpoints exist for seeds 1 and 2, extending this to three trajectories per arm
would move the claim from disclosed-weak to supported.

**9. `parvo_highsf` is not architecture-matched, and it is the decisive case.** Its second
stream takes two input channels rather than one, so input and first-layer fan-in change
together (Limitation 3). Your capacity control (§2.1) rules out the confound in the
direction that would fake the null, and that is the right control, but it does not make the
comparison matched. Please state this qualification at the point where §2.2's decisive claim
is made, not only in the Limitations.

**10. Human-participant reporting is absent.** There is no ethics-approval statement, no
consent or secondary-use statement, no participant demographics, and no
constraints-on-generality statement, for either the eight NSD subjects or the ten cue-conflict
observers. Please add these, citing the original approvals for both datasets, together with
data availability, code availability, materials, and a completed Reporting Summary
(Behavioural & Social Sciences). Your own Open Items note that the Geirhos licence is
unverified; that must be settled before submission.

**11. Length and display items exceed the format.** Main text is approximately 7,956 words
and the manuscript carries ten tables and two figures. Much of the excess is the
self-correction record in Results — the checkpoint-artefact history, the 28-fold to 3.7-fold
revision, the 37% to 20.7% accuracy revision. That record is valuable and should be kept, but
it is project history rather than result, and Supplementary is its natural home.

**12. Why two streams?** The manuscript establishes what the two streams must receive but
never tests whether the split is doing work. A single trunk carrying both DKL channels under
the identical diet is the obvious control and its absence is conspicuous in a paper whose
subject is a two-stream architecture.

### Minor comments

m1. §2.1 excludes `parvo_highsf` seed 0 from the behavioural row after observing its decision
count; please state whether this rule was fixed in advance.
m2. Test statistics and degrees of freedom are never reported; software and package versions
are absent.
m3. Effect sizes appear twice (dz −11.70 / −1.06 in §2.3; dz values in the konio paragraph)
against roughly forty claims.
m4. "Magno" and "parvo" denote luminance and L−M inputs to identical ResNet-18 backbones.
Since §3 argues against channel essentialism, consider naming the streams by what they
receive at first use.
m5. The abstract is approximately 636 words and opens with the paradigm rather than the
question.
m6. Acronyms undefined on first use for a non-specialist: DKL, DVD, NSD, VSB, NC, PR, CS cut,
MiniEcoset, `nsdgeneral`, Kastner2015.
m7. §2.3's konio paragraph is dense enough to warrant its own supplementary panel, as its
own source section anticipated.
m8. The status banner, the positioning note, and the section-by-section `PAPER_PLAN`
provenance tags should be removed from the submitted version.

### Notes by criterion

**Validity.** The experimental logic is sound and the manipulations do what they claim. The
flaws I identify are in reporting and verification rather than in design, with two
exceptions: the architecture confound in the decisive `parvo_highsf` case, and the
single-trajectory basis of the installation claim. Both are disclosed.

**Originality and significance.** Cannot be fully assessed without a reference list. Within
the manuscript, the 2×2 decomposition and the inverted-front-end test are genuine
methodological advances over a two-condition literature. The error-consistency result is the
finding most likely to change practice.

**Data and methodology.** Methods are unusually complete on parameters and unusually
incomplete on availability. §4.5's cross-validation and out-of-sample handling are clean.

**Statistics and uncertainties.** The weakest area relative to the design. See major comments
2–4.

**Conclusions.** Generally well matched to the evidence, and in several places the
manuscript is more conservative than its data would license — §2.2's refusal to report a
level, §2.5's refusal to claim phase-scrambling, the withdrawal of the hMT+ and eccentricity
results. The overreach is in the developmental framing rather than in the statistics.

**References.** Absent.

**Clarity and context.** Limitations are conveyed exceptionally well. Accessibility to an
outside-field reader is poor.

---

## Confidential Comments to the Editor

**Triage and breadth.** I would discuss rather than send out. The work is careful and in
several respects exemplary — the disclosed failed predictions and the retired claims are
better practice than most of what I review — but the breadth test does not pass. The
constituencies are visual cognitive neuroscience and computational vision modelling, which
are one readership. Developmental psychology is reachable if the authors made the
developmental claim, but the front end is static and they say so.

**Significance.** Against our heuristics I count one clear hit: §2.6 would change how
vision-model papers report human alignment, and error consistency is cheap wherever human
trial data exist. The remaining heuristics are not met — the finding does not resolve a
debate outside the subfield, is not the first adequately powered test of a long-debated
question in behavioural science, and has no policy or clinical reach. One hit is
`borderline`.

**Recommendation and transfer.** Reject with transfer. The natural home is Nature
Communications, which takes model-plus-brain-plus-behaviour work of this shape without
requiring the behavioural conclusion we do. eLife would suit it particularly well: the
reviewed-preprint model is a good match for a manuscript that carries several explicitly
disclosed n=1 arms and a public correction record, both of which are strengths there and
liabilities here. Communications Psychology is a third option if the authors want to keep
the human-alignment framing central.

**What would change my view.** If the authors reframed around the metric critique — leading
with the demonstration that shape bias, the field's most-cited human-alignment measure, is
uninformative about human decision structure, and using the mechanistic 2×2 as the
explanation for *why* rate and structure come apart — the paper becomes a meta-science
contribution about how behavioural scientists evaluate models of human vision, and the
breadth test becomes arguable. That reframe requires no new experiments. It does require the
equivalence testing in major comment 4, because a meta-science claim resting on an
uncharacterised null will not survive review.

**Revision time.** 3–6 months, no new participants. The reference list and the statistics
block are the long poles.

**Additional analyses, cost/benefit.** TOST or Bayes factors on the three dissociations —
days of work, decisive for the central claims. Trajectories for seeds 1–2 in §2.2 — free if
checkpoints are cached, and it repairs the manuscript's weakest structural claim. The
single-stream control already exists in the authors' records and should be included. A
parameter-matched single-stream arm would cost ~3 seeds × 26 h and is not required for the
claims as stated.

**Expertise and conflicts.** I am confident on encoding models, the shape-bias literature,
and statistical reporting; less so on the developmental-acuity literature, where I have
relied on the manuscript's own characterisation. No conflicts.

---

*This review was generated by an LLM-assisted reviewer simulation for the authors' own
self-improvement; the numeric rubric is internal to the simulation and is not an NHB form.*
