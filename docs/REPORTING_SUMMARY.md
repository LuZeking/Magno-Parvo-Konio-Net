# Nature Reporting Summary — content draft

**Purpose.** The Nature-portfolio Reporting Summary is a fillable PDF supplied by
the journal at submission. This file holds the *content* for each of its items so
that filling the form is transcription rather than recall, and so that every
entry has a source in the repository. **Venue changed to NMI on 2026-09-09**
(draft log; previously NHB, decided 2026-09-02 and re-confirmed 2026-09-06).

The content below was drafted against the **Behavioural & Social Sciences**
variant and transfers item for item: statistics, software and code, data, human
participants and study design are common to the Nature-portfolio variants, and
the form the journal supplies at submission governs which labels apply. Nothing
below was written to a venue-specific prompt. What NMI adds is
machine-learning reporting — data provenance, splits and leakage controls,
optimisation, model and baselines, evaluation — which is stated in
`PAPER_DRAFT.md` §4.1–§4.7 and audited item by item against DOME in
`docs/NMI_REVIEW_2026-09-09.md`.

**Status.** Every factual entry below is either read from the code and results
files or cited to the section of `PAPER_DRAFT.md` that states it. Two items
cannot be completed from what we hold and say so explicitly rather than being
filled with a plausible value — see *Population characteristics* and
*Non-participation*. Those are the only gaps.

---

## 1. Statistics

> *For all statistical analyses, confirm that the following items are present in
> the figure legend, table legend, main text, or Methods section.*

**The exact sample size (n) for each experimental group/condition, given as a
discrete number and unit of measurement.**
Three inferential axes, never pooled (§4.7):
- **n = 8** human participants (unit: NSD participants, subj01–subj08) for every
  cortical encoding contrast.
- **n = 3** networks per condition (unit: independently seeded training runs,
  seeds 0/1/2) for every behavioural, probe and robustness contrast; twelve
  checkpoints in total across four conditions (§4.2).
- **n = 10** human observers (unit: observers, 1,280 trials each) for the
  behavioural reference; **45** observer pairs for the human–human error-
  consistency ceiling (§4.3).
- Stimuli: **1,280** cue-conflict images, **1,200** after excluding those whose
  shape and texture categories coincide (§4.3).
Two arms are single-run and are labelled `n = 1` wherever they appear: the
`swap` and `parvo_highsf` cortical fits, and §2.4's accuracy-matched control.

**A statement on whether measurements were taken from distinct samples or
whether the same sample was measured repeatedly.**
Repeatedly, on both human axes. The same eight NSD participants are measured
under every model condition, so all cortical contrasts are within-participant
and paired. The same ten observers viewed the whole cue-conflict set, giving a
complete 10 × 1,280 design. Network seeds are distinct samples (independent
initialisations), and seed-level comparisons are therefore between-groups.

**The statistical test(s) used AND whether they are one- or two-sided.**
All tests two-sided; no one-sided test is reported anywhere.
- Within-participant: **paired exact Wilcoxon signed-rank**, n = 8.
- Between-seed: **exact Mann–Whitney U**, 3 v 3.
- Sign test, n = 3, where only direction is available.
- **TOST** equivalence tests where a null is bounded (Schuirmann 1987; Lakens
  2017), reported with their interval.
- **Spearman ρ** for monotonicity against hierarchical rank.
- **Welch t** for the phase-scrambling comparison, whose degrees of freedom
  (2.3) are stated in the text because they are the point.

**A description of all covariates tested.** None. No model in this paper
includes a covariate; every reported contrast is a paired or unpaired difference
of the quantity named in its section. Eccentricity enters as a *stratifying
variable* in §2.2's pRF analyses (`prf-eccrois`), not as a regression covariate,
and the one eccentricity-residualised test that was considered is reported as
**not computable** on this atlas because `prf-eccrois` labels 0% of VO2 and
9–23% of VO1 voxels (§4.5).

**A description of any assumptions or corrections, such as tests of normality
and adjustment for multiple comparisons.**
- Normality is **not assumed**: every primary test is exact and
  distribution-free. This is why exact tests were chosen at these sample sizes
  rather than *t* tests.
- **No multiple-comparison correction is claimed anywhere**, and §4.7 states why
  in full: the paper reports 28 p-values, and the exact two-sided Wilcoxon floor
  at n = 8 is p = 2/2⁸ = .0078125, so Holm–Bonferroni over any family larger
  than six tests cannot yield a single rejection at α = .05. Over the 18-test
  pre-specified confirmatory family both Holm and Benjamini–Hochberg return
  .141. This is a resolution limit of an exact test at n = 8, not a
  choice-of-procedure problem. Claims rest instead on complete separation
  (8/8 or 0/8), agreement of the raw and normalised statistics, replication
  across three seeds, and effect size.
- **Two-statistics rule** (§4.7): every contrast is computed both as
  `median(Uₘ−Uₚ)` and as `median(Uₘ−Uₚ)/median(Uₘ+Uₚ)`, and a claim is made only
  where the two agree. Disagreements are reported as unresolved.
- Equivalence bounds were **not** chosen after seeing the intervals; where no
  bound was pre-specified the null is reported as unbounded and is asserted in
  neither direction.

**A full description of the statistical parameters including central tendency
(e.g. means) or other basic estimates (e.g. regression coefficient) AND
variation (e.g. standard deviation) or associated estimates of uncertainty.**
`±` denotes a **standard deviation** throughout, at ddof = 1, over the unit named
in each case: across three training seeds for behavioural and probe quantities,
across ten observers for the human figure (0.958 ± 0.019), across eight
participants for participation-ratio and stream-preference tables. Ranges are
written as explicit intervals, never as `±`. Cue-conflict values additionally
carry a **bootstrap 95% CI over images**, labelled as a CI wherever it appears.
Voxel-level aggregation is median-of-ratios; across-participant aggregation is
the mean of per-participant medians. §4.7 records that an earlier version of the
manuscript mixed ddof = 0 and ddof = 1 across seventeen cells and that all were
re-derived.

**For null hypothesis testing, the test statistic (e.g. F, t, r) with
confidence intervals, effect sizes, degrees of freedom and P value noted.**
Exact test statistics (W, U), exact p-values and separation counts are given
with every contrast. Effect sizes are **dz** for paired contrasts; §2.2's
seed-0 normalised cells are dz = −6.60 (diet) against −1.09 (no diet), and −6.60
is the largest in the table — both values are locked by
`tmp_analysis/verify_new_claims.py::check_effect_size`. Confidence intervals for
paired medians are **Hodges–Lehmann** over the 36 Walsh averages at n = 8; the
attained coverage at α = .10 is **92.19%**, not 90%, and is reported as attained
rather than nominal because at n = 8 the exact interval cannot hit 90%.

**For Bayesian analysis, information on the choice of priors and Markov chain
Monte Carlo settings.** Not applicable; no Bayesian analysis is reported.

**For hierarchical and complex designs, identification of the appropriate level
for tests and full reporting of outcomes.**
The design is hierarchical on two axes that are deliberately never crossed in a
single test (§4.7): voxels within participants, and participants within model
conditions; separately, seeds within conditions. Voxels are aggregated to a
per-participant median before any test, so no test treats voxels as independent
units. A seed-level claim additionally requires the within-seed participant test
to hold in all three seeds independently. Seeds and participants are never
pooled into one n.

**Estimates of effect sizes (e.g. Cohen's d, Pearson's r), indicating how they
were calculated.** dz = mean(d)/sd(d, ddof = 1) over the paired per-participant
differences; Spearman ρ for rank monotonicity; Cohen's κ for trial-level
agreement, computed as (c_obs − c_exp)/(1 − c_exp) with
c_exp = p_m·p_h + (1−p_m)(1−p_h) and reported additionally as a percentage of
the human–human ceiling (§4.3).

---

## 2. Software and code

**Data collection.** No software was used for data collection. Both human
datasets are existing public releases; this study collected no new human data.

**Data analysis.** Python 3.11.15 with NumPy 2.4.6, SciPy 1.17.1, PyTorch
2.13.0+cu130, torchvision 0.28.0+cu130, himalaya 0.4.11 (banded-ridge solver),
scikit-learn 1.9.0, Matplotlib 3.11.1, pandas 3.0.5, nibabel 5.4.2, h5py 3.16.0,
Pillow 12.3.0, joblib 1.5.3, tqdm 4.70.0. Jobs scheduled with Slurm 25.11.8 —
CPU nodes for all encoding, statistical and plotting work, GPU nodes for network
training only. Every version was read from the installed environment's package
metadata and independently confirmed by importing each module; the two agree on
all fourteen. statsmodels and scikit-image are **not** installed and are behind
no number in this paper. Analysis modules are named by path in §4.1–§4.7. Code
will be released with a DOI and an environment manifest on acceptance.

*RRIDs.* No antibodies, cell lines, organisms or other registered wet-lab
resources were used, so those categories do not apply. Software is identified by
name, exact version and citation. SciCrunch RRIDs are not asserted for the
individual Python packages because we have not verified registry entries for
each, and an unverified identifier is worse than none; any the journal requires
will be taken from the registry at submission.

---

## 3. Data

**Data availability statement.** As in the manuscript's *Data availability*.
- **NSD** — `naturalscenesdataset.org`, under its own data-use agreement
  requiring registration. We used the `betas_fithrf_GLMdenoise_RR` (b3)
  preparation and the `nsdgeneral` mask, with ROIs from the Wang et al. (2015)
  probabilistic atlas as distributed with NSD.
- **Geirhos cue-conflict stimuli and observer responses** —
  `github.com/rgeirhos/texture-vs-shape`, directory
  `raw-data/style-transfer-512-nomask-experiment/`, under **CC BY 4.0** per that
  repository's `DATASET_LICENSE` file (verified against the file, 2026-09-08).
  Mirrored at `github.com/bethgelab/model-vs-human`
  (`raw-data/cue-conflict/`); the two mirrors were parsed by two independently
  written scripts that agree to four decimals on the human reference
  (mean 0.958, sd 0.019, range 0.919–0.987), which is how that number was
  validated.
- **MiniEcoset** — `osf.io/msna2` (Thorat & Kietzmann, 2024).
- Derived results tables and fitted encoding-model summaries sufficient to
  reproduce every figure will be deposited with a DOI on acceptance.

**Restrictions.** NSD requires registration and a data-use agreement; we cannot
redistribute it. The Geirhos repository separately notes that the
ImageNet-derived *images* underlying the stylisation cannot be redistributed by
its authors — a restriction on those source images, not on the observer response
data analysed here.

---

## 4. Human research participants

**Population characteristics.**
- *Neuroimaging:* eight adults with normal or corrected-to-normal vision (NSD;
  Allen et al., 2022), United States.
- *Behavioural:* ten adult observers with normal or corrected-to-normal vision,
  tested under controlled laboratory viewing in a German university laboratory
  (Geirhos et al., 2019 data release).
- **GAP — cannot be completed from what we hold.** Neither age nor sex is
  reproduced here. The Geirhos per-observer files carry only trial-level fields
  (`subj`, `session`, `trial`, `rt`, `object_response`, `category`, `condition`,
  `imagename`) and no demographic column, so no per-observer sex or age exists in
  the data we analysed. For NSD the release documents its participants, but the
  subject-information file is not part of the subset mirrored here, so no value
  has been verified against the released metadata. **Action before submission:**
  either fetch NSD's subject-information file and transcribe the eight rows with
  the file named as the source, or state in this item that demographics are
  deferred to the source publications. Do not fill this item from the source
  papers' prose without recording which document the numbers came from.

**Recruitment.** Not applicable to this study — we recruited no participants.
Recruitment for both samples is described in the source publications (Allen et
al., 2022; Geirhos et al., 2019/2021). Both are small, Western,
convenience-recruited adult samples; the manuscript's *Sample and constraints on
generality* states that claims about "human observers" should be read as claims
about these samples, and that nothing here establishes cross-cultural,
developmental or clinical generality. In particular, the developmental
motivation of the model front end is a motivation for the model and not a claim
about any developmental population we measured.

**Ethics oversight.** NSD was collected under approval of the University of
Minnesota institutional review board with written informed consent from all
eight participants; the behavioural data were collected under the approval
described in Geirhos et al. Our secondary analysis of two de-identified,
publicly released datasets required no additional approval under our
institution's policy. No participant-identifying information was accessed at any
point.

---

## 5. Field-specific reporting

**Selected:** *Behavioural & social sciences.*

Rationale, for the editor: the paper's claims concern human visual behaviour and
its cortical basis, tested against human behavioural and neuroimaging data. Note
for internal use — the manuscript's surviving positive claims are cortical
(§2.2, §2.4) while the human-behavioural comparisons are nulls, which is the
open venue question recorded in the draft; if the paper moves to a
neuroscience-facing journal the **Life sciences** form applies instead and the
study-design items below map onto it with no change of content.

**Study type:** Quantitative, secondary analysis of existing human data combined
with computational modelling. Not experimental with respect to human
participants: no human was assigned to a condition by us.

---

## 6. Behavioural & social sciences study design

**Study description.** Secondary quantitative analysis of two existing public
human datasets — trial-level cue-conflict behaviour from ten observers and 7T
fMRI from eight participants — used to test predictions of a two-stream
convolutional network trained under a developmental visual diet. The human data
are fixed; the manipulation is over model conditions (front end, training seed,
readout weight, checkpoint policy).

**Research sample.** See *Population characteristics*. Both samples are existing
public releases chosen because each is the standard instrument for its question:
the Geirhos set is the field's reference measurement for shape bias with human
observers on the identical stimuli, and NSD is the largest available
single-participant-depth 7T dataset of natural-scene viewing.

**Sampling strategy.** No sampling was performed by us; both datasets were used
in their entirety at the level of participants and observers. All eight NSD
participants and all ten observers are included, and no participant, observer or
session was dropped. No sample-size calculation was performed, and none would be
meaningful: both n are fixed properties of the releases. What that leaves is a
**resolution limit rather than a power estimate**, and it is stated as such —
the exact two-sided Wilcoxon floor at n = 8 is p = .0078125 (§4.7), so the
design's discriminating evidence is complete separation and effect size rather
than the p-value.

**Data collection.** No data were collected for this study. Model training and
all analyses were run on a Slurm cluster; the human data were downloaded from
the public repositories named in *Data*. The researcher was not blind to
condition, because there is no human-facing condition to be blind to: all
contrasts are between model conditions, and every analysis is a deterministic
script over stored arrays with seeds fixed and recorded.

**Timing.** Not applicable — no data collection window. Analyses reported here
were run between 2026-08-27 and 2026-09-08; job identifiers and dates are
recorded per result in the manuscript's open-items log, and per-figure build
scripts and source directories are named in each figure caption.

**Data exclusions.** Three exclusions, all pre-specified or reported with the
count:
1. **Stimuli** — the 80 cue-conflict images whose shape and texture categories
   coincide are excluded (1,280 → 1,200), and the set is restricted to the 16
   categories present in MiniEcoset. Both restrictions are stated in §4.3.
2. **Trials decided as neither shape nor texture** are excluded from the
   denominator; because that count varies about four-fold across cells, it is
   reported per cell rather than folded into a rate.
3. **Voxels** below noise ceiling 0.1 are excluded, and any ROI cell with fewer
   than 20 surviving voxels is reported as *missing* rather than estimated
   (§4.5).
No participant, observer, seed or session was excluded. One quarantined
encoding file is documented in Extended Data §E2.2 with its cause established;
the corresponding leave-one-out influence check is reported (participant 01
carries about 31% of the no-diet contrast) alongside the n = 8 primary analysis,
so the sensitivity is disclosed rather than resolved by dropping the case.

**Non-participation.** **GAP — not available to us.** No participant dropped out
of *this* study, since we collected no data. Drop-out and exclusion during the
original collections is a property of the source releases and is not reported at
a level we can restate. Action before submission: state exactly that, rather
than entering "none", which would misdescribe the original collections.

**Randomization.** No allocation of participants to groups occurred: every
participant and observer contributes to every condition, so the design is
fully within-subject on the human axes and there is nothing to randomize. On the
model axis, allocation is by training seed (0, 1, 2), which fixes
initialisation and data order and is recorded per run. Stimulus order within the
original behavioural experiment was controlled by its authors and is described
in the source publication. Cross-validation folds for the encoding models are
`GroupKFold` by 73k image id with a fixed seed, so image identity never spans
train and test, and the PCA is fit on training images only and never refit on
test or on the cue-conflict set.

---

## 7. Reporting for specific materials, systems and methods

| Item | Involved? | Note |
|---|---|---|
| Antibodies | No | |
| Eukaryotic cell lines | No | |
| Palaeontology and archaeology | No | |
| Animals and other organisms | No | No animal was used; all cited primate anatomy and physiology is literature. |
| Human research participants | **Yes** | See §4 above. Secondary analysis of two public releases; no new collection. |
| Clinical data | No | Not a clinical study; no trial registration applies. |
| Dual use research of concern | No | |
| Plants | No | |

**Magnetic resonance imaging.** The fMRI is not ours, and the Reporting Summary's
MRI subsection should be answered by reference to the NSD release rather than
restated: acquisition, preprocessing and the `betas_fithrf_GLMdenoise_RR`
single-trial GLM are Allen et al. (2022), which this paper cites for the full
protocol. The only two acquisition facts asserted anywhere in our materials are
the ones we can source: **7T**, from the cited dataset paper's own title, and the
**1.8 mm** functional resolution, from the `func1pt8mm` paths our code reads
(`src/analysis/encoding/nsd_data.py`). Sequence details are deliberately not
restated — a first draft of this file said "gradient-echo EPI", which is not
verified anywhere in this repository, and it was removed rather than kept as
plausible. What is ours, and is described in §4.5,
is everything downstream of the released betas: conversion to percent signal
change, per-session per-voxel z-scoring, the `nsdgeneral` mask, the 73k-image-id
split with the shared-515 images held out as test, PCA to 1,000 components per
stream fit on training images only, two-band banded ridge with per-voxel alphas
and per-voxel band weights, 3-fold `GroupKFold`, the NSD `ncsnr` noise ceiling
with its 0.1 floor and 20-voxel minimum, and the variance partition. No
whole-brain statistical map is reported and no cluster-correction question
arises: all inference is over pre-declared ROIs of the Wang et al. (2015) atlas,
with the ROI family size disclosed where a set was selected post hoc (§2.2b).

---

## Provenance of this file

Written 2026-09-08. Every numeric entry is either quoted from
`docs/PAPER_DRAFT.md` §4.1–§4.7 or measured from the repository; the software
versions come from the installed environment's package metadata cross-checked
against live imports; the CC BY 4.0 licence was verified against the
repository's `DATASET_LICENSE` file. The two gaps are marked **GAP** in place
rather than filled, and each names the action that would close it.
