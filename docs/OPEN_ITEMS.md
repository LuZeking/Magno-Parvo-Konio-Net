# Open items before submission

Moved out of `PAPER_DRAFT.md` on 2026-09-10 (revision plan T2.5). This is the
project's working checklist, including unstarted analyses; it is useful
internally and must not reach an editor as part of the manuscript. Nothing is
changed by the move — the content below is verbatim.

---
- [x] **subj01 provenance RESOLVED 2026-09-07 — it is the repair, and the
      outlier is a subject property.** Settled from the surviving job log rather
      than a re-fit: `slurm/logs/enccc_26399760_16.out` line 1 records
      `frontend='--no-lum-blur --no-lum-cs'` for task 16 / `dkl_base_seed0` /
      subject 1 — the flags job 26386734_0 omitted — and the run recomputed
      features rather than trusting an unstamped cache. mtime
      (2026-09-01 05:27:47, against siblings on 08-29/30) and size (2.85 MB
      against 1.4–2.0 MB) mark it as the only cc-round file in
      `results/encoding/`, consistent with the `cp -p` the README instructed,
      and it is byte-identical to the `encoding_cc` copy. **Two errors of mine
      here.** I treated the absent provenance stamp as evidence, but no file the
      writer produces carries one — the uncontested subj02 from the same round
      has the same 28 keys and no stamp either, so the README promised a stamp
      the code never implemented (worth fixing, but it cannot discriminate).
      And I inferred a front-end fault from the anomaly being arm-specific,
      which it is not: subject 01 is displaced from the sibling median by
      **−0.195 (no diet) and −0.189 (diet)**, the same to within 0.006, and the
      ratio looked arm-specific only because the baselines differ 15-fold.
      Displacement across all six fits is −0.091 to −0.205 while the ratio spans
      1.15× to 5.73×. **A ratio compared against baselines of different scale —
      the fourth time that exact error has bitten in this thread, and this time
      it was mine.** Now reported as a leave-one-out influence check (subj01
      carries ~31% of the no-diet contrast), n = 8 primary.
- [x] **`visualrois.py` argument + provenance stamp — ALREADY DONE 2026-09-07;
      this item was stale.** The script takes `--model`, `--src`, `--out-dir`
      and `--no-write`, and every run writes a stamped artefact under
      `results/visualrois/` carrying the git commit and dirty flag, the exact
      input files with byte size and mtime, the full convention (nc floor,
      min-voxel count, denominator mask, median-of-ratios, V1 reference,
      test), the per-ROI means and the per-subject contrast values. Three
      artefacts are committed — `untrained_seed0`, `dkl_bcs_seed0_trained` and
      `dkl_base_seed0_trained`, all dated 2026-09-07 — so each published row
      has a regenerable source. `verify_new_claims.py` still reimplements the
      convention independently rather than importing the script, so the values
      stay checked from two directions.
      **One correction went in with the close.** The docstring claimed the
      default row has "continuous provenance back to 2026-08-28". That holds
      for **six** of eight subject fits: `subj04` and `subj08` were written
      2026-09-04 by job 26423414 tasks 3 and 7. Same script and same argument
      as the original round (`encode_subjects.sh untrained_seed0` — no
      checkpoint, no front-end flags, so untrained weights on the diet front
      end), and the logs record that both **recomputed features rather than
      trusting an unstamped cache**, the same discipline as the subj01 repair.
      Configuration matches; the dates do not, and "all eight" overstated it.
      The docstring now says so.
- [x] **The untrained control's source directory — RESOLVED 2026-09-08 by
      stating the exception, not by refitting.** §2.2 declares
      `results/encoding_cc/` authoritative while `untrained_seed0` exists only
      in `results/encoding/`. The item offered two routes; the evidence chose
      the second, and against the recommendation this session started with.
      **Why not refit.** (a) It would change nothing. The directories differ
      only by `run_encoding.py`'s `--cue-conflict`, which applies the *already
      fitted* estimators to an extra design (`banded_ridge.fit_banded`
      docstring) while `prepare_band` takes "all statistics from the training
      split only" — so the flag cannot move `r2_joint`, `r2_magno` or
      `r2_parvo`. Checked, not assumed: over the **40 files present in both
      directories** (5 models x 8 subjects) the maximum absolute difference in
      all three is **exactly 0.000e+00**. (b) It would destroy audit value: the
      untrained row has the longest unbroken provenance in the analysis, six of
      eight fits predating the quarantine, and refitting replaces that with a
      new fit. (c) It is over the memory cap. The `encoding_cc` fits measured
      **77-128 GB** MaxRSS (job 26416428) and this family's script requests
      160G — above the 64G ceiling, so a refit would need `--voxel-chunk`
      footprint reduction or g017 before it could even be submitted. Spending
      that to reproduce numbers already known to be identical fails the standing
      rule against compute that cannot change a conclusion.
      **What was done instead.** §4.7 now states the exception explicitly with
      the mechanism and the check; §2.2 gives the control on **both** statistics
      (−0.544 normalised and −0.120 raw, 8/8 and p = .0078 on each, so the claim
      it supports clears the two-statistics rule — previously only the
      normalised figure appeared); and `verify_new_claims.py` gained
      `check_untrained_anchor()`, which locks the anchor on both statistics and
      **also locks the cross-directory identity invariant**, so the §E2.2
      assertion can no longer go quietly false if one directory is ever refit
      without the other.
- [x] **§2.2's table restated from `results/encoding_cc/` 2026-09-07.** The
      previously reported values (normalised −0.862 / −0.802 / −0.781; raw
      −0.176 / −0.151 / −0.160; dz −11.70; untrained −0.471; cross-codebase
      p = .84) **do not regenerate**. Roughly 200 configurations were searched
      across noise-ceiling threshold, both masks, min-voxel count,
      median-of-ratios vs ratio-of-medians, negative-value handling, four
      contrast definitions, four V1 references, two across-subject
      aggregations, and both output directories (Extended Data §E2.2); the
      closest joint match is still 0.014–0.021 off and dz tops out at −8.74.
      Two useful results came out of it: `encoding_cc` is the only directory
      holding all three seeds for both arms and is now the declared source, and
      the hV4 − V1 contrast reading is confirmed because nothing else reaches
      the right order of magnitude. `verify_new_claims.py` now locks all twelve
      values, the p-values, the separation counts and the source directory.
      Sign, ordering, monotonicity (ρ −0.964 vs −0.536) and significance are
      unchanged; the bias-corrected diet displacement is −0.871.
      **CAUSE ESTABLISHED 2026-09-07 — the search could not have succeeded.**
      The no-diet figures were computed with the defective subj01 file still in
      the set, before the 2026-09-01 quarantine, and transcribed on 2026-09-06
      (−0.862 enters `docs/` in c0002a2, 14:24, with every input frozen since
      08-28/29). Solving for the subj01 value each published mean requires gives
      8.0× the sibling median and 53% of the effect on the normalised statistic,
      6.3× and 46% on the raw — against a quarantine README, written earlier and
      independently, that calls the file a "six-fold outlier" contributing "57%"
      and "the difference between significant and not on the raw statistic".
      Three independent quantities agree. The data no longer exists, so no
      configuration could have reproduced those two numbers. The diet column
      (which would need subj01 at −1.225, outside all 48 observed cells) and the
      untrained control remain disclosed as drift with no established cause.
- [x] **Estimator-mismatch re-run CLOSED 2026-09-07 by bounding it instead.**
      The floor is a ratio-of-medians while §2.2's effect is a median-of-ratios.
      Rather than re-run condition B per-voxel (~8 × 1.5 h), both estimators were
      computed on the six real fits: the gap is 0.4%–11.6% of each effect and
      scales with magnitude (r = +0.754). Extrapolated, the floor is
      0.0437–0.0585 even on the pessimistic constant-offset reading, giving diet
      14.2–19.1× and no-diet 1.3–1.8× — and §2.2 needs only diet ≫ 1 and no-diet
      ≈ 1, which holds across both entire ranges. The re-run would tighten ±0.007
      to ±0.002 and change nothing, so under the standing rule it is not
      justified. The mismatch stays disclosed in §E2.1 **with its bound**, which
      is a stronger disclosure than a promised fix. Contingent on the diet-arm
      floor being the same order as +0.051 — **confirmed 2026-09-07**: the
      diet-arm floor is +0.038, within the same order and not distinguishable
      (Mann–Whitney p = .33, 8v8), so the bound stands. Also pinned:
      the ×1.26 / ×0.71 magnitude-fidelity figures are ratios of across-subject
      means; an OLS slope over the 56 cells gives 1.34 for the ratio index.
- [x] **Artefact floor for §2.2 — DONE (no-diet arm) 2026-09-07** (job
      26440398, array 0-15%6, `main`, no GPU;
      `tmp_analysis/vp_recovery_realnc.py`, Extended Data §E2.1). With the
      *measured* declining noise-ceiling profile imposed and the true contrast
      solved to exactly zero, the floor in §2.2's own hV4 − V1 contrast is
      **+0.051 normalised (8/8, p = .0078) and +0.0009 raw (4/8, p = .84)**,
      with the simulated profile *rising* (ρ = +0.68) where the observed one
      falls. Opposite sign on both statistics and on the gradient, and 6% of
      the effect's magnitude: the estimator opposes §2.2's result rather than
      manufacturing it. **Two corrections went in with it.** The floor was
      first computed as a median over the ROI profile (+0.126 / +0.025) — a
      2.5× overstatement and, more importantly, *not §2.2's statistic*; a
      parallel session's units challenge is what surfaced it, and
      `verify_new_claims.py` now pins all four of §2.2's published numbers to
      the hV4 − V1 reading, which is the only reading they reproduce under.
      And the claim that −0.862 is "a floor rather than a ceiling" is
      **withdrawn**: condition A shows the two statistics biased in opposite
      directions (ratio ×1.26, difference ×0.71), so §2.2 now reads as
      sign-and-ordering, not calibrated magnitude. No number from the earlier
      ascending-ramp pilot is quoted anywhere — it does not reproduce at 1,000
      components, so the registered +0.05 magnitude prediction it was scaled
      from should not be scored against it either. (In §2.2's actual statistic
      that prediction lands at +0.051, which is right; it was judged failed
      only against our wrong statistic.)
- [x] **Diet-arm half of the artefact floor — DONE 2026-09-07** (job 26442320,
      array 0-7%6, `main`, no GPU, 8/8 COMPLETED at 1:36–1:46 per task). The
      floor on diet-arm feature geometry is **+0.038 normalised** (8/8,
      p = .0078, range +0.014 to +0.065) and **−0.0021 raw** (2/8, p = .15,
      indistinguishable from zero), with the simulated profile rising at
      ρ = +0.80. Against the no-diet geometry's +0.051 that is a 27% difference
      in point estimate and **not a detectable one** (Mann–Whitney p = .33,
      8 versus 8), so the floor is a property of the instrument rather than of
      the model being fitted, and the single-geometry caveat is lifted. Matched
      to each arm's own geometry the bias-corrected displacements are
      **−0.871** (diet) and **−0.130** (no diet), each non-zero in every
      subject; the ratio framing this replaced is retired (see the
      displacement item below). Recovery
      fidelity replicates the opposite-direction bias on the second geometry
      too (ratio ×1.15, difference ×0.69), so the withdrawal of "−0.862 is a
      floor" now rests on both arms. The earlier filename collision that lost
      this arm is recorded above.
- [x] **Progressive-transition OOD arm — DONE 2026-09-07** (job 26440500,
      single job, N=1; `results/ood_bestep/thesis_dvd_trans.json`, §2.6.1 and
      Extended Data §E6.2). This answers the strongest objection to §2.6.1 —
      that our front end is static at 48 months while the developmental
      proposal is about the training *order*. The arm that carries the actual
      age transition lands **inside the static diet arm's three-seed range on
      five of seven corruptions**, closer to the diet arm than to the no-diet
      arm on six of seven, and *below* the diet arm on the two exceptions.
      Clean accuracy 0.568 against 0.553 (diet) and 0.695 (no diet): the
      schedule recovers neither robustness nor accuracy. One seed and one saved
      checkpoint (epoch 147), so the claim is containment in the observed seed
      range, not equivalence.
- [x] **Geirhos human observer data obtained** (2026-09-02) —
      `/scratch/byrong/VST_data/geirhos_human_data/cue_conflict`, computed by
      `tmp_analysis/geirhos_humans.py`, recorded in PLAN §52. Human mean 0.958;
      our CNN recomputation reproduces the published values. Every human
      reference stays directional; "human-like" stays out.
- [x] **Geirhos data release cited and licence CONFIRMED 2026-09-08 — CC BY
      4.0.** The terms were checked at the source rather than assumed: the
      repository carries a `DATASET_LICENSE` file separate from any code
      licence, and that file is the Creative Commons Attribution 4.0
      International text. This matches what PLAN §52.4 had guessed, but §52.4
      had flagged it as "confirm the licence file before submission" and the
      guess is now replaced by the file. The manuscript states the licence in
      *Materials*, names our exact source directory
      (`raw-data/style-transfer-512-nomask-experiment/`), and — because CC BY
      4.0 requires modifications to be indicated — lists ours: the 16-category
      restriction, the exclusion of the 80 same-cue trials, and recomputation of
      the observer summaries from per-trial CSVs, with the stimulus images
      unmodified. A **data-release reference entry** is now in the reference
      list alongside the four Geirhos paper entries, which is what §52.4
      required and what the draft previously lacked.
      One thing to keep straight: the repository README says the ImageNet-derived
      *images* behind the stylisation cannot be redistributed by its authors.
      That restriction is about those source images, not about the observer
      response data this paper uses, and the two should not be conflated in any
      later summary of the licence position.
- [x] **Software, versions and the RRID question — DONE 2026-09-08.** A
      *Software and code* statement now sits in the availability section with
      every version **read from the installed environment's package metadata**
      (`*.dist-info`) and independently cross-checked by importing each module
      and printing `__version__`; the two agree on all fourteen. Python 3.11.15,
      NumPy 2.4.6, SciPy 1.17.1, PyTorch 2.13.0+cu130, torchvision 0.28.0+cu130,
      himalaya 0.4.11, scikit-learn 1.9.0, Matplotlib 3.11.1, pandas 3.0.5,
      nibabel 5.4.2, h5py 3.16.0, Pillow 12.3.0, joblib 1.5.3, tqdm 4.70.0;
      Slurm 25.11.8. The cross-check also established that **statsmodels and
      scikit-image are not installed**, so neither can be behind any number here
      — worth knowing, since both are easy to assume for statistical work.
      Four software citations were missing and are now in the reference list
      (NumPy: Harris et al. 2020; SciPy: Virtanen et al. 2020; scikit-learn:
      Pedregosa et al. 2011; Matplotlib: Hunter 2007). PyTorch (Paszke) and
      himalaya (Dupré la Tour) were already cited.
      **RRIDs are deliberately not asserted.** No antibodies, cell lines or
      organisms are involved, so those categories do not apply, and we have not
      verified SciCrunch registry entries for the individual Python packages. An
      unverified identifier is worse than none, so the statement says so plainly
      and defers any journal-required RRIDs to submission. This is a disclosure,
      not a gap to be closed by guessing.
      *Also fixed while in the reference list:* it was not alphabetical. My four
      insertions initially landed in one block before `Geirhos (2018a)`, and a
      pre-existing error had `Islam et al. (2021)` filed after both Jagadeesh
      entries. A throwaway order check over all 59 entries found both; the list
      now passes it.
- [x] **Reporting Summary drafted 2026-09-08** —
      `docs/REPORTING_SUMMARY.md`, all seven sections of the Behavioural &
      Social Sciences form, every entry sourced to §4.1–§4.7 or measured from the
      repository. It is the *content*, not the PDF; the journal supplies the
      fillable form at submission. **The 2026-09-09 move to NMI does not
      invalidate it**: the sections it covers are common to the
      Nature-portfolio variants and none was written to a venue-specific
      prompt, so it transfers item for item. What NMI adds on top is
      machine-learning reporting, which §4.1–§4.7 already state and which the
      NMI simulation audits against DOME
      (`docs/NMI_REVIEW_2026-09-09.md`) — no D or E failures, four Partials
      across O and M.
      **Two items are marked GAP rather than filled**, which is the point of
      doing this early. (i) *Population characteristics* — participant age and
      sex. The Geirhos per-observer files carry only trial-level fields and no
      demographic column, so per-observer sex does not exist in the data we
      analysed; NSD documents its participants but the subject-information file
      is not in the subset mirrored here. (ii) *Non-participation* — drop-out
      during the original collections is a property of the source releases;
      entering "none" would misdescribe them, so the form will say exactly that.
      Each gap names the action that would close it.
      **This corrected an over-promise in the draft.** *Sex as a biological
      variable* previously said sex "is recorded in the source releases for both
      datasets" and that per-subject values "will be tabulated in the Reporting
      Summary before submission". The first half is not true of the behavioural
      release and the second could not have been delivered for it. That
      paragraph now states the per-release position and why it differs.
      One unverified detail was also caught and removed from the summary itself:
      a first draft wrote NSD's sequence as "gradient-echo EPI", which nothing in
      this repository sources. What is sourced is 7T (the dataset paper's title)
      and the 1.8 mm functional resolution (the `func1pt8mm` paths the code
      reads); sequence details are deferred to Allen et al. (2022).
- [x] **Human benchmark is in the abstract** (2026-09-02), but as a *rate*
      comparison carrying its own qualification — the whole-model figure (0.692)
      is given alongside the stream figure (0.878), and the error-consistency
      null (§2.6.2) is stated in the same paragraph.
- [x] **Figure 1 built** — `results/figures/fig1_convergence.{png,pdf}`, caption
      in `fig1_convergence_caption.md`, built by `tmp_analysis/fig1{,_data}.py`
      (PLAN §51). Panel (c) is the set point as a front-end→trained slopegraph,
      superseding §46.3's retention spec per §49.
- [x] **Ventral-gradient figure renumbered** (2026-09-03) —
      `fig1_diet_gradient.{png,pdf}` → `fig2_ventral_gradient.{png,pdf}`,
      caption → `fig2_ventral_gradient_caption.md`. It is §2.2(b)'s evidence;
      Figure 1 is the four-panel convergence figure (PLAN §51.4).
- [x] **Venue: NMI** (changed 2026-09-09, at the author's direction). The
      transferable result for a machine-intelligence readership is the
      measurement one — a shape-bias rate is a property of the readout, does
      not identify a representation (§2.3), and does not track human decision
      structure (§2.6.2: never above 20.7% of the human–human κ ceiling across
      189 readouts with `w` optimised post hoc per seed). NMI's scope covers
      evaluation work of broad implication, and the paper's primary
      interlocutor (Lu et al., 2026) published there. A referee simulation at
      commit `911ba8a` (`docs/NMI_REVIEW_2026-09-09.md`) returns **Major
      revision, 6.85/10**, above NHB v2's 6.4 and Nat Neurosci's 6.09 — but
      **not a clean comparison**: NMI was scored against a materially later
      draft, so part of that margin is draft improvement rather than venue fit,
      and a like-for-like statement needs NHB and NN re-run at `911ba8a`. All
      three personas send it out or discuss and none votes the significance
      gate down. Six major comments, three of them cheap: a nonlinear-readout
      sweep over cached features, a pinned environment manifest, and the
      permutation null over ROI rank labels.
      **The earlier reasoning for dropping NMI was too narrow, and is kept
      below rather than deleted.** It read "no ML payoff" as requiring a
      performance win; the payoff here is a negative result about a metric the
      vision-model community uses constantly, which is inside NMI's remit. What
      that reasoning got right survives as the simulation's major comment 6 —
      the Discussion still disclaims any training recommendation and calls the
      work's interest developmental, so the measurement result has to be
      re-led, not assumed to carry.
- [x] **Superseded venue record — NHB** (decided 2026-09-02, re-confirmed
      2026-09-06 after the reframe; superseded 2026-09-09).
      NeurIPS/NMI dropped at the time — there was judged to be no ML payoff to
      report, only
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
      clear a pre-fixed bound; the achieved bounds are now quoted in §2.6.1,
      §2.6.2 and the VO entry, with the two caveats §78.3 requires. See `docs/NHB_REVIEW_2026-09-06.md` §"Specific
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
      defect class as Extended Data §E6.1's ×1.00 identity cells, which the reframe just
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
- [~] **Length — mechanical pass executed 2026-09-10; the substance calls
      remain open.** Main text is now **8,748 body words** with **7 main-text
      tables**, down from 11,186 words / 9 tables at commit `637dd45` — a
      **2,438-word (22%) cut** made entirely as moves and de-duplications, with
      no claim, caveat, failed control or disclosure dropped. Verified: every
      numeric token that left the main text still resolves in Extended Data
      (three apparent losses checked individually and confirmed present at
      higher precision or as an enumeration compressed to its range), all
      Extended Data and Supplementary cross-references resolve, and
      `verify_new_claims.py` returns FAILURES: none.

      What moved: §E2.1.1 (the floor subtraction's assumption, the
      residual-misfit analysis, the six-variant sensitivity table), §E6.2.1 (the
      progressive-transition arm per corruption), §E6.3.1 (the per-image target's
      construction, its reliability controls and the two-code-path check),
      §S1.6 (the three withdrawn framings of the no-diet
      gradient) and §S1.7 (the readout sweep presented as a measurement). What
      was removed as duplication: §2.2's supersession paragraph, its subject-01
      provenance archaeology and its artefact-floor table, all three already
      carried at greater length in §E2.1/§E2.2, and its six-failure enumeration,
      already set out under *Claims deliberately not made*. Also repaired: a
      botched extraction that had left §S1.1 opening on an orphan word and
      ending mid-sentence, and a duplicated protocol sentence in §2.6.1.

      **Still ~3,700 words above NMI's ≈3,000–5,000, and closing that gap is a
      substance call, not an editorial one.** What is left in Results is not
      accumulated prose but the results themselves: §2.2 1,563, Discussion
      1,399, §2.6.2 1,143, §2.3 779, §2.6.1 745, §2.5 630, §2.4 587, §1 651,
      §2.1 455, §2.7 318. Reaching 5,000 means demoting whole results — the
      candidates are §2.7's cost accounting (which reviewers reward, and which I
      would argue against cutting), §2.5's trajectory result, and §2.6.2's
      per-image correspondence, whose headline would stay while the analysis
      moved wholesale to §E6.3. Those are the author's calls and are left open.
            The NMI simulation raised the length as major comment 5
      (`docs/NMI_REVIEW_2026-09-09.md`). **Re-run 2026-09-10 after the cut**
      (`docs/NMI_REVIEW_2026-09-10.md`): 7.15/10, up from 6.85, still Major
      revision. Only two axes moved — Reproducibility 2→3 and Clarity 2→3 —
      which are exactly the two the intervening work targeted. The re-run also
      corrects an earlier projection of ours: packaging plus the length cut plus
      a re-lead was estimated to reach ~7.53 and cross into Minor revision, but
      measured, that whole editorial bundle tops out at **7.43**. The only
      remaining item that crosses the band by itself is the **nonlinear readout
      sweep over the cached features** (~+0.5, roughly a day, no meaningful
      compute). The 2026-09-07 pass is kept below for what it moved and why.
      *Superseded measurement (2026-09-07):* main text was **~6,830
      words** against NHB's ≈4,000–6,000 and NN's ≈3,000–5,000, down from
      **10,002** — a 32% cut — with **8 display items** (5 tables + 3 figures)
      against a cap of 8, down from 15. The cut was executed as a *move*, not a
      delete: every qualification, control table, seed-coverage note and caveat
      sits in `docs/PAPER_EXTENDED_DATA.md` (§E1–§E6.3), verbatim, cited from
      the point in Results where it applies. Nothing was dropped for being
      inconvenient; the family-size disclosure (§2.2) and §2.7's cost
      accounting stayed in the main text by choice.

      **§2.6.1 (effective dimensionality) was cut to Extended Data on
      2026-09-07** on the user's instruction — the one editorial call this pass
      deliberately left open. It is a claim change as well as a move, and it was
      handled as one: the result still counts as one of the paper's three
      dissociations, §2.6 now states it in a single paragraph carrying the
      decisive cell (`parvo_highsf` more dimensional at 0.56 while texture-biased
      at probe −0.266) and cites **Extended Data §E6.1**, where the full result,
      its table and its identity checks now live. The abstract, Introduction,
      §2.6.2 and the Discussion still say *three*, and all four now resolve to a
      section the reader can reach. The two surviving subsections were renumbered
      §2.6.2→§2.6.1 and §2.6.3→§2.6.2 so §2.6 has no gap; 23 cross-references in
      the draft and 7 in Extended Data were updated with them.

      **The §2.6.1 cut netted ~110 words, not the 203 estimated**, because the
      claim had to stay stated somewhere: the abstract, Introduction, §2.6.2 and
      the Discussion all assert *three* dissociations, so §2.6 retains a 79-word
      paragraph carrying the decisive cell and pointing to Extended Data §E6.1.
      The structural gain — one fewer subsection, §2.6 down from three parts to
      two — is the larger effect.

      **What is left is not fat.** Reaching 6,000 would now mean demoting §2.7
      (316 w, the cost accounting — reviewers reward it and I would argue against
      cutting it) or §2.5's trajectory result. Both are substance calls, left
      open.

- [x] **§2.2(a)'s raw statistic certified 2026-09-06 — no error.** A parallel
      session could not reproduce the no-diet −0.033 and raised a possible
      **sign** error; on investigation the draft number was right and their
      comparison statistic was wrong. `norm` is
      `vsb = (η²_shape − η²_texture)/(η²_shape + η²_texture)`
      (`voxel_shape_bias.py:69`); **`raw` is the un-normalised numerator**
      η²_shape − η²_texture. Re-derived directly from the stored per-band η²
      fields, n=8, NC>0.1: diet **+0.6062 / +0.6475 / +0.6180** against the
      draft's 0.616 / 0.658 / 0.632, and no-diet **−0.0283 / −0.0478 / −0.0505**
      against −0.033 / −0.039 / −0.053 — all three no-diet seeds negative,
      matching the draft's sign, with the ~0.01 residual explained by their not
      applying the <20-voxel ROI-cell drop. The sign readings were cut anyway,
      on the independent ground that a near-zero sign the separation does not
      need is weak whether or not the number is right (PLAN §83.5, §83.6).
- [x] **The §73 `"raw"` trap removed from the tree 2026-09-06.** The project
      carried three quantities across two uses of the word: §2.2(a)'s
      un-normalised dissociation numerator (+0.616); `cc_ladder.py`'s key named
      `"raw"`, which was the unresidualised hV4−V1 step on the **joint** band —
      differing on two axes at once, so a mix-up yields a categorically
      different number rather than a plausibly wrong one; and that script's
      `"mp"`, the normalised dissociation (+1.535). `cc_ladder.py`'s key is
      renamed **`step_unresid`** with a comment recording why. Verified no
      consumer outside that file; its printed label ("hV4-V1 uncorrected") was
      already correct, so only the key was misleading.
- [x] **Recovery analysis CLOSED 2026-09-08 — production landed, and both
      conditions were already in the text.** The item asked for the pilot to be
      replaced by production numbers before anything was written. Production is
      in (`results/vp_recovery_realnc/`, 16 files = 2 arms x 8 subjects, from the
      jobs recorded in the two artefact-floor items above), and on re-reading,
      §2.2 already carries **both** conditions at production scale: condition B
      is the artefact floor (+0.051 / +0.038 normalised, the paired-displacement
      table), and condition A is the multiplicative term *a* = 1.28 / 1.21 with
      its residual-misfit analysis and the six-variant table. Nothing needed
      pasting; the item outlived its own deliverable. Corrected production
      condition A: Spearman rho = +1.000 against a true +1.000 in **8/8 subjects
      on both arms**, per-ROI |est - true| median 0.070-0.074, max 0.137.
      **A trap to leave marked.** `results/vp_recovery/` (array 26439650, 8/8
      COMPLETED) is the *superseded* output: `vp_recovery.py:264-276` records
      that its `roi_nc` is a hardcoded ramp which **ascends** with hierarchical
      rank where the measured profile descends, and is ~2.5x too wide, so its
      condition B rho (median **-0.804**, 8/8 negative) is a normalisation
      artefact whose sign is set by the ramp direction. The file says in terms:
      "Do not read condition B's sign out of THIS file." Read from
      `vp_recovery_realnc/` only. Taken at face value that superseded rho would
      have put a manufactured gradient of §2.2's *own sign* into the text — the
      same class of error as the hardcoded `roi_nc` ramp that inverted the
      artefact's sign in the first place, and the reason both the script warning
      and this note exist.
      *The three carry-items resolve as follows.* (i) The **null bias toward
      magno** is unchanged and already handled: it makes a null more
      conservative, and the VO sign claim it bore on was withdrawn on
      independent grounds. (ii) The **discretisation artefact does not survive**
      at production settings and needs no sentence: condition B's per-ROI
      estimates take 54/56 (diet) and 52/56 (no diet) distinct values at 4 d.p.
      with no gap above 0.01 in the sorted values — they scatter, they do not
      cluster on two levels. What *does* repeat is the Spearman rho itself
      (+0.071 three times), and that is only the discreteness of rho on n = 7
      ROIs, where it is quantised in steps of 1/56 = 0.0179 — a property of the
      statistic, not of the estimator. (iii) The **identifiability sentence is
      now in §4.5**, on production numbers and in the two-claims form agreed:
      the canonical-correlation spectrum between bands runs ~0.000 to 0.925
      (diet) and ~0.000 to 0.954 (no diet), median 0.292 / 0.306, so unique
      variance is *identifiable in these spaces*; that it is *recovered
      correctly* is condition A's claim and is cited as such. Pilot scale had
      reported the largest as ~0.81; production puts it at 0.925 / 0.954, so the
      pilot figure is superseded and is not quoted.
      *Also worth recording:* condition B's residual rho is a **null that is not
      tightly bounded**. Diet median -0.214 (Wilcoxon p = .1094, 5/8 negative),
      no diet median +0.054 (p = 1.000, 3/8) — the two arms disagree in sign and
      neither reaches the n = 8 threshold of .0781, so no ordering artefact is
      detected, but at n = 8 a modest one cannot be excluded. That is stated as
      a non-detection, not as evidence of absence (§103). The argument §2.2
      actually rests on is the *magnitude and sign* of the floor, which is
      bounded: condition B's estimates are positive in every cell (min +0.019
      diet, +0.006 no diet) and span only ~0.04 across the seven ROIs, against an
      observed displacement of -0.871 — wrong sign, and more than 20x smaller.
- [ ] **Cortical-instrument analyses from the NN referee sim — (a) DONE
      2026-09-08, (b) still open.** Claimed by a parallel session 2026-09-06;
      do not duplicate. **(a) is complete**: the Uₘ/Uₚ recovery analysis landed
      as conditions A and B of `vp_recovery_realnc.py` and is written into §2.2
      (floor and displacement tables, the multiplicative term) and §4.5
      (identifiability) — see the closed recovery item above, including the
      warning not to read the superseded `results/vp_recovery/`. **(b) is not
      done.** Original text of both parts retained below.
      (a) **Uₘ/Uₚ recovery analysis** — simulate voxels
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
      here verbatim. **If any of the four 2-dp cells in Extended Data §E6.1 is ever raised to
      3 dp, regenerate from that module rather than hand-editing** — they are
      unchanged only because they round the same, not because their underlying
      value held.
- [x] **Extended Data §E6.1's `±` column reprinted at 3 dp** (2026-09-06), all eight cells,
      from the parallel session's module at 6 decimals. This **resolves** the
      knife edge rather than disclosing it: 0.094891 now prints 0.095
      unambiguously. Each of the four newly supplied values reproduces its
      previously committed 2-dp form (0.14/0.13/0.17/0.13), so nothing already
      pasted was wrong — this only adds precision. **Means deliberately left at
      2 dp**: `swap`/Gabor is 3.3665, exactly on the 3-dp boundary, rendering
      3.366 under round-half-even and 3.367 under round-half-up, so raising it
      would encode a rounding convention rather than a measurement (PLAN §81).

- [x] **Figure 3 built 2026-09-06** — `results/figures/fig3_readout_sweep.{png,pdf}`,
      caption in `fig3_readout_sweep_caption.md`, built by
      `tmp_analysis/fig3_readout_sweep.py`. Panel (a) is the 21-point `w` sweep,
      3 seeds × 2 checkpoint policies, with the `w`=0.5 anchor marked; panel (b)
      puts the field on the same axis so the reader sees one network span the
      range the literature covers with one network per point. Every value is
      read from `results/fusion_sweep/` and the committed cue-conflict JSONs —
      only the three published Geirhos CNN values are hardcoded, and they are
      labelled as published. Two bugs were caught and fixed before it was
      trusted: an unfiltered glob that averaged the no-diet arm across *every*
      condition in `results/cue_conflict` (0.511 instead of 0.416), and a span
      annotation that drew the across-seed envelope (0.54) where the text
      reports the mean-curve span (0.505).
- [x] **Figures complete for the main text** (2026-09-07). Five main figures
      plus one Extended Data figure, each with a caption file in
      `results/figures/`:
      **Fig 1** `fig1_convergence` (§2.1/§2.4/§2.5/§2.7),
      **Fig 2** `fig2_ventral_gradient` (§2.2),
      **Fig 3** `fig3_readout_sweep` (§2.3),
      **Fig 4** `fig4_ood_degradation` (§2.6.1, NEW),
      **Fig 5** `fig5_human_correspondence` (§2.6.2, NEW),
      **Extended Data Fig 1** `figE1_routes_and_accuracy` (§2.1 routes and
      §2.4's accuracy control, NEW). Built by
      `tmp_analysis/fig{4,5}_*.py` and `figE1_*.py` via `slurm/build_figures.sh`
      (array 26444920); every value is read from committed results files.
      **Two entries on the superseded list above were stale:** §2.3's w-sweep
      curve is Figure 3a, built 2026-09-06, and §2.4's 2×2 is Figure 1a, built
      2026-09-02 — neither was missing. The real gap was the callouts: the draft
      had **one** (Figure 3a) across seven result sections, so Figures 1 and 2
      existed uncited. Eleven callouts are now placed, covering every panel of
      every figure.
      *Two choices worth recording.* (i) §2.1's routes and §2.4's accuracy
      control went to Extended Data rather than the main text because each rests
      on one model per cell — the untrained references carry no training seed and
      the matched arm is a single retrained run — so the figure's placement
      matches the evidence's strength. (ii) An earlier draft of Extended Data
      Fig 1 put the probe and cue-conflict instruments on twin y-axes; they have
      different zero points (0 = sign flip, 0.5 = parity), which placed each
      cue-conflict point at a probe value it does not have. Split into two
      single-axis panels.
- [x] **Methods expanded** (2026-09-03) — §4.1–4.7, every parameter read from
      the code and run scripts rather than from memory; 3 seeds per condition
      stated, and the subject vs seed inferential axes separated in §4.7.
- [ ] Supplementary: the VSB cortical profile with its negatives intact; the
      capacity ladder (PLAN §42) with its non-monotonicity caveat.
- [x] **Cortical seed replication complete** (2026-09-04) — array 26416428,
      32/32 COMPLETED. §2.2 replicates across three seeds with both statistics
      agreeing (diet seed range 0.032). This retired the n=1 status of §2.2's
      *main* contrast. It did **not** make the paper free of single-seed
      results, and an earlier version of this line said so incorrectly: the
      `swap` and `parvo_highsf` cortical fits are seed 0 only, and §2.4's
      accuracy-matched control is one training run. Both are labelled as n=1
      where they appear (§2.2, §2.4, Extended Data §E2, §E4) (PLAN §66).


---

## Draft status and positioning (moved out of PAPER_DRAFT.md, 2026-09-10)

Removed from the manuscript head because it is internal metadata of the same
not-for-submission class as the checklist below it — the referee's manuscript-
hygiene comment applies to it just as much. Kept verbatim.

**Status: SECOND DRAFT, 2026-09-06.** Reframed from the first draft
(2026-09-02), which led with the carrier x transform conjunction. Structure and
argument are settled; prose is second-pass. Every number is tagged with its
`PAPER_PLAN.md` section so it can be traced to the run that produced it.
Nothing here is written from memory.

**Positioning (revised 2026-09-06; superseded in part by T3.1, 2026-09-10).**
The paper led with the question the originating thesis posed and could not
answer: why human behaviour is shape-biased when ventral-stream representations
are texture-like. T3.1 has since re-led the abstract and Introduction with the
measurement claim, and the paradox is now the third paragraph rather than the
first. The mechanistic results that previously led (the front-end 2x2, the
side-of-parity result) are retained as the answer to *how the shape-biased
component arises and survives training*, not as the paper's claim, and are now
merged into a single §2.4–2.5. Claims the earlier framing invited and this one
does not make are listed under *Claims deliberately not made*.
