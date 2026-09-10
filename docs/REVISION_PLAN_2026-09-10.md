# Revision plan — pushing the paper past Major revision

Written 2026-09-10 at commit `8dfb4bf`. Current state: **7.15/10, Major
revision** (`docs/NMI_REVIEW_2026-09-10.md`), 8,753 body words, 12 display
items.

## The organising principle

**Close referee comments, do not chase rubric points.** The weighted score is a
proxy; `Recommendation` alone carries 37% of it, and a referee moves from Major
to Minor revision when nothing is left blocking — not when seven axes each gain
a fraction. Every item below is therefore owned by a specific major comment from
the 2026-09-10 simulation, with one exception that is flagged as such.

The arithmetic that shapes the plan: **the entire editorial bundle — re-lead,
DOI, referee access, further length reduction — tops out at 7.43 and cannot
cross the 7.5 band.** Only the nonlinear readout sweep crosses it alone
(≈ +0.5). So the plan has one critical-path item and a lot of cheap work that
must not be mistaken for progress on it.

## Two corrections to the review's own cost estimates

Both found while sizing this plan, and both make the plan more honest rather
than more comfortable.

1. **The features are not cached.** The review said the nonlinear sweep is
   "cheap because the per-stream features are already stored". They are not.
   `tmp_analysis/perimage_logits.py` computes `F1`/`F2` internally but saves only
   `L1`/`L2` — the *logits* (line 111). The affine identity is exactly why
   logits sufficed: `clf(w·y₁+(1−w)·y₂) = w·clf(y₁)+(1−w)·clf(y₂)`. A nonlinear
   readout cannot be computed from logits, by construction. The sweep therefore
   needs a feature-extraction pass first. Checkpoints are available at
   `/scratch/byrong/VST_data/mpk_runs/dkl_bcs_seed{0,1,2}`.
2. **The permutation null is cheaper than billed.** It needs no new fits: the 80
   files in `results/encoding_cc/` already carry `roi_prf-visualrois` and
   `noise_ceiling` per subject. It is a CPU analysis over existing artefacts —
   hours, not a day.

---

# Tier 1 — the critical path (the only band-crossing item)

## T1. Nonlinear readout sweep — closes MC2, and decides the title

**The question.** §2.3's span is an identity of the affine head. Does a
*nonlinear* readout over the same two frozen streams still move the shape-bias
rate across the range, or is the free-parameter result an artefact of
affineness? This decides whether the title is about readouts or about affine
readouts.

**Design (recommended: A).**

| | design | what it tests | cost |
|---|---|---|---|
| **A** | Re-fit a small MLP head (512→h→112) on `w·y₁+(1−w)·y₂` at each of 21 `w`, per seed | Does the span survive a nonlinear *and fitted* readout over frozen features? | extraction + 63 small fits |
| B | Train one MLP on `[y₁,y₂]`, sweep a gate at test time | Weaker — head not matched to each `w` | half of A |
| C | Frozen classifier, nonlinear fusion `clf(g(·))` | Ad hoc; likely just degrades | trivial |

Take **A**. It answers the referee directly and is the version whose negative
result would still be publishable. Keep **B** as the fallback if A's fits prove
unstable.

**Steps.**
1. Extend `perimage_logits.py` (or fork it) to also save `F1`/`F2`. Extract
   penultimate features for MiniEcoset train/val **and** the 1,200 cue-conflict
   images, for 3 seeds × 2 checkpoint policies. One GPU array job, `--array=0-5`.
   Estimated footprint ≈ 3–4 GB on `/scratch`; well within budget.
2. Fit the MLP head at each `w` on the cached features. 21 × 3 = 63 fits on
   512-d inputs — CPU-feasible, one array job on `main`.
3. Score cue-conflict shape bias at each `w`; report **accuracy alongside**, so
   a span that only appears where the head is degraded is visible as such.

**Pre-registered decision rule — write this into PLAN before running anything.**
This project's own standard (§77.5) forbids looking first and deciding after.

- If the nonlinear span covers **≥ 80%** of the affine span (0.373–0.913) with
  clean seed behaviour → the claim generalises beyond affine readouts. §2.3 and
  the title stand as written, strengthened; say so explicitly.
- If it covers **< 50%**, or the span only appears where head accuracy collapses
  → the title takes a qualifier ("in affine readouts of these components") and
  §2.3 states the boundary. **This is an acceptable outcome and must be reported
  either way.**
- Between 50% and 80% → report the measured span and claim the weaker,
  quantified version. No re-running to taste.

**Risk.** A genuine chance this narrows the claim. That is the point of running
it, and the paper's credibility rests on reporting it whichever way it lands —
as it already does for two failed pre-registrations and six failed regional
hypotheses.

**Cost.** ~1 day of work, ~2 days wall-clock with queueing. GPU only for step 1.

---

# Tier 2 — cheap, near-certain, no experiment risk

Run these in parallel with T1; none blocks any other.

## T2.1 Permutation null over hierarchical-rank labels — closes MC3
Permute ROI hierarchical-rank labels across the 20 Kastner ROIs, recompute the
per-subject Spearman ρ, and report where the observed −0.964 falls in the null.
Pure CPU on `results/encoding_cc/`. Report **beside** the family-size
disclosure, never instead of it — a null cannot undo post-hoc selection, and
§2.2 already says so. *(hours, `main`)*

## T2.2 Abstract carries §2.3's concession — closes MC1
The abstract presents the 0.40→0.90 sweep as an empirical finding; §2.3
establishes it is exact interpolation containing no information beyond `w` = 0
and `w` = 1. One clause fixes the gap. This is the cheapest closure of a major
comment in the whole plan. *(1 hour)*

## T2.3 resnet34 capacity-matched arm — closes MC5
Already training (job 26444916, epoch ~113–121/150, ~5 h out). Judge under
§2.3's own checkpoint policy: **two-stream 0.655 at the best saved checkpoint**,
not the final-checkpoint 0.675. If the matched single-stream model reaches 0.655
or above, §2.3 can no longer attribute the 0.088 gap to the architectural split
and must be restated. Current val top-1 is 0.587–0.611, tracking below.
*(free — already running)*

## T2.4 Repository: referee access, DOI, CITATION.cff — closes MC4
Three parts, of which only the first needs a decision from the author:
- **Referee access.** NMI policy is availability *during* review. Decide whether
  to share under the journal's confidential-link mechanism. *(author decision)*
- `CITATION.cff` and a Zenodo DOI. *(hours)*
- Either pin the three git URLs in `setup.cfg`, or state in it what
  `requirements.txt`'s header already establishes — that no module named in
  §4.1–§4.7 imports any of them. The second is cheaper and more truthful.
  *(minutes)*

## T2.5 Strip *Open items before submission*
The authors' working checklist, including unstarted analyses, is still in the
draft. It must not reach an editor. Keep it in the repository, out of the
manuscript. *(minutes — listed because it is exactly the kind of thing that
survives to submission)*

---

# Tier 3 — positioning (changes no result)

## T3.1 Re-lead with the measurement claim — closes MC7
Two results transfer directly to an ML readership: that the field's most-cited
human-alignment metric is a free parameter of the readout, and that across 189
readouts with `w` optimised post hoc, agreement never reaches 21% of the
human–human ceiling. §2.6.1's contradiction of the source paper's
robust-vision claim is a third. All three currently sit deep in the paper while
the abstract opens on ventral texture statistics. Rewriting the abstract and
Introduction to lead with the measurement claim changes no claim; it changes
which claim a reader meets first. *(half a day)*

**Note on ordering:** do T3.1 *after* T1 lands. If the nonlinear sweep narrows
the claim, the re-lead has to lead with the qualified version, and writing it
twice is waste.

## T3.2 Length to ≈6,000 words and ≤ 8 display items — closes MC6
What remains is results, not prose, so this now means demoting a result.
Cleanest candidate: §2.6.2's per-image correspondence analysis — headline
(24.1% vs 8.7%) stays in Results, mechanics move to §E6.3. Second candidate:
§2.5's trajectory result. §2.7's cost accounting should stay; reviewers reward
it. *(1 day)*

---

# Tier 4 — the gap no referee has found yet

## T4.1 Separate the front end's two components (blur vs contrast-sensitivity)
**Not requested by any referee.** The self-contained reason is enough on its
own: §4.1's transform bundles a Gaussian blur and a scale-free
contrast-sensitivity cut, the paper attributes effects to "the transform"
throughout, and it never separates the two. §2.4's 2×2 crosses *carrier* against
*transform*; it does not decompose the transform. A referee who notices will
ask which component does the work.

*A caution on the external motivation.* A component ablation of the diet also
exists on the source paper's side, and an earlier session's read of their
released performance CSVs suggested contrast sensitivity alone carries most of
the shape-bias effect and blur much less. **Those figures are not recorded in
this repository and must not be cited from memory.** Before they appear in the
paper they need one of two things: sourcing to a table the source paper itself
publishes (citable, no licence question), or re-derivation from their released
data (which is re-analysis, and needs the licence question of T4.2 answered
first). Run T4.1 on the internal motivation; add the external comparison only
once it has a source.

Two new training conditions (blur-only, CS-only) × 3 seeds = 6 GPU runs, then
the §2.4 probe and cue-conflict scoring. Slots into the existing 2×2 as a 2×4.
*(~2 days GPU; decide after T1 lands, not before)*

## T4.2 Developmental age-band comparison — **parked, and should stay parked**
The 4–6-year-old human shape bias (0.8755) sits 0.0025 from this model's magno
stream. Blocked on two things, both of which must be resolved *before* anyone
looks at the data again: the mirrored release carries **no LICENCE file**, and
the comparison was found post hoc, so it must be pre-registered first. The
project's own §77.5 forbids target-picking. Do not open this to make a number.

---

# Sequencing

| when | item | resource |
|---|---|---|
| now | T2.3 finishes on its own | GPU (running) |
| now | T2.2, T2.5, T2.4 (DOI, CITATION, setup.cfg) | none |
| day 1 | T2.1 permutation null | CPU array, `main` |
| day 1 | T1 step 1 — feature extraction | GPU array, `--array=0-5` |
| day 2 | T1 steps 2–3 — MLP fits and scoring | CPU array, `main` |
| day 3 | **decision point:** read T1 against its pre-registered rule | — |
| day 3–4 | T3.1 re-lead (in the form T1's outcome dictates) | none |
| day 4 | T3.2 length | none |
| later | T4.1, only if T1 leaves capacity | GPU |

All Slurm work fits the standing rules: one array job per multi-task step, CPU
work on `--partition=main` with no GRES, an explicit `--qos=`, and nothing above
the 64G ceiling. The encoding stage is untouched by this plan, so its ~160G
historical footprint does not arise.

# What I would not do

- **Do not chase the rubric.** T2.2 (one clause in the abstract) closes a major
  comment and scores nothing; T3.1 scores +0.06 and closes another. Both are
  worth doing; neither is worth doing *for the number*.
- **Do not run T4.1 before T1.** If T1 qualifies the title, the front-end
  decomposition is a different paper's problem.
- **Do not touch T4.2** until the licence question is answered and the
  prediction is pre-registered.
- **Do not add seeds or re-run anything that already reports a range.** The
  standing research-output rule: ship the story, note the imperfection.
