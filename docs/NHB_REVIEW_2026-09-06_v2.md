# NHB Referee Simulation v2 — `docs/PAPER_DRAFT.md` (post-reframe)

Run 2026-09-06 against the draft at commit `d38dbdb`, i.e. **after** the reframe
(`c0002a2`) and the §2.6.3 corrections. Three-persona simulation per
`~/.claude/skills/nhb-reviewer`. **The numeric rubric is internal to the simulation and is
not an NHB form.**

The first pass (`docs/NHB_REVIEW_2026-09-06.md`, commit `107ba04`) returned
**Reject-with-transfer, 5.9/10**. This run re-scores the same axes so the delta is
attributable to the draft rather than to reviewer drift.

## NHB Review Summary

- **Title:** Shape-biased behaviour does not require shape-biased representations
- **Article type:** Article
- **Triage:** **send out** — the conclusion is now an inferential claim about how
  behavioural measures license representational descriptions, which is a claim about
  human behavioural science rather than about a CNN. That clears the scope gate the
  first pass failed. It goes out with a conditional: the premise it rests on is
  currently uncited.
- **Breadth test:** vision science / cognitive neuroscience **+** the wider community
  that infers representational properties from behavioural rates (psychology, cognitive
  science, model–human alignment). **Two constituencies, thinly served for the second.**
- **Predicted Decision:** **Major revision** (up from Reject-with-transfer)
- **Recommendation (avg of 3):** 2.7 / 5 · **Weighted simulation score: 6.4 / 10**
  (prior draft recomputed on identical arithmetic: **6.0**; the earlier report printed
  5.9, a rounding difference. The prior draft sat on the band boundary, not deep in the
  reject band, so the move is **+0.4** and it crosses one line.)
- **Significance gate (NHB-caliber?):** borderline — R1 borderline, R2 borderline,
  R3 borderline (**up from R2 "no"**)
- **Confidence:** 4.0 / 5
- **Ethics Flag:** **yes**, unchanged — still no ethics approval, consent, secondary-use,
  demographic or constraints-on-generality statement for either human dataset
  (NSD n=8; Geirhos observers n=10). This is a paperwork gap, not a conduct concern:
  both datasets have published approvals the authors can cite.
- **Preregistered:** internally pre-specified in a version-controlled plan; **not a
  registry**. Must not be called "pre-registered" unqualified.
- **Estimated revision time:** 2–3 months · **New data required: no** (new *compute*: a
  few CPU-hours, see Suggestion 4)

## Per-axis Scores (1–4) — simulation rubric, not an official NHB form

| Axis | R1 | R2 | R3 | Avg | Prior | Δ |
|---|---|---|---|---|---|---|
| Advance | 3 | 3 | 3 | 3.0 | 2.7 | +0.3 |
| Soundness | 3 | 3 | 3 | 3.0 | 3.0 | — |
| Evidence | 3 | 3 | 3 | 3.0 | 2.7 | +0.3 |
| Relevance | 3 | 2 | 2 | 2.3 | 1.7 | **+0.6** |
| Statistics | 3 | 3 | 3 | 3.0 | 2.7 | +0.3 |
| OpenScience | 2 | 2 | 2 | 2.0 | 2.0 | — |
| Ethics | 2 | 2 | 2 | 2.0 | 2.0 | — |
| Clarity | 2 | 2 | 2 | 2.0 | 1.7 | +0.3 |
| Recommendation (/5) | 3 | 2 | 3 | 2.7 | 2.7 | — |
| Confidence (/5) | 4 | 3 | 5 | 4.0 | 3.7 | +0.3 |

**Where the +0.4 came from.** Decomposed on the same weights: the **reframe** contributes
**+0.24** (Advance, Relevance, Clarity) and the **equivalence tests** — added by a
parallel session, not part of the reframe — contribute **+0.18** (Evidence, Statistics,
Confidence). The reframe is the larger contributor and, more to the point, it is the one
that moves **Relevance**, the axis that was driving the transfer recommendation.

Hard overrides checked:
- Triage reject by ≥2 → **no** (all three send out or discuss).
- SignificanceGate "no" by ≥2 → **no** (all three borderline; R2 moved off "no").
- **Fatal statistical item → YES.** `±` remains undefined manuscript-wide while carrying
  the §2.5 headline table and §2.3's `0.675 ±0.025`. **Caps at Major revision.**
- **EthicsFlag → YES.** Caps at Major revision; editorial-consultation note.

Both caps are now **binding**: the score alone would sit at Major revision, and the
overrides prevent anything better. In the prior pass neither cap bound, because the score
was already below the band. **The paper is now held at Major revision by two items that
need no new data and no new compute.**

## Consensus Strengths

Intersection across all three personas.

1. **The reframe is the right move and the argument is now genuinely four-step.** §2.1
   (opposite biases, different routes) → §2.2 (the texture-biased stream carries the
   ventral end) → §2.3 (behaviour mixes both) is a clean chain, and each step is
   measured rather than asserted. The paradox is a real tension and the resolution is
   the interesting kind: nothing has to be wrong for both facts to hold.
2. **§2.1's two-route asymmetry is the strongest new material.** +0.796 untrained →
   +0.810 trained on the luminance stream against −0.087 → −0.482 on the chromatic one
   separates "supplied by the input" from "built by learning" on one instrument. That
   the shape-biased component is *not* something learning creates is a genuinely
   counterintuitive result, and reporting it plainly — rather than letting a reader
   infer that training produces the shape bias — is the single biggest honesty gain
   over the prior draft.
3. **The construction guarantee is correctly stated as an identity.** 0/120 tensor pairs
   differing at initialisation, `rgb_tsn` returning exactly +0.0000, and the explicit
   refusal to read that as "two streams do not spontaneously specialise" (with
   Limitation 12 naming the experiment that *would* test it). This is precisely the
   discipline the prior pass found missing in §2.6.1.
4. **§2.6.1's ×1.00 overclaim is fixed.** The two identity cells are now labelled as
   identity checks rather than counted among "four out of four" predictions. Two real
   predictions, both correct, stated as two.
5. **All four nulls are now bounded, with the right caveats travelling with them.** κ
   within ±0.013 (5.4% of ceiling); a κ change >0.037 excluded across a 0.706 shape-bias
   span; phase-scrambling labelled the weakest at Welch df 2.3; VO's equivalence flagged
   as resting on the point estimate rather than on sensitivity (detectable 0.0220 >
   declared ±0.0195). Checklist item 20 satisfied, and the caveats are the sort most
   papers omit.
6. **Confirmatory/exploratory separation remains exemplary.** Two of four pre-specified
   predictions failed and are reported as failures; *Claims deliberately not made* records
   nine retired analyses including three the authors had previously promoted (hMT+,
   eccentricity, dorsal/ventral). Carried over from the prior pass and still rare.
7. **The two-statistics rule and the nesting discipline** (subjects and seeds never
   pooled; a seed-level claim requires the subject test to hold in all three seeds
   independently; Limitation 6 flagging the PR n=8 as subjects-as-image-samples).

## Consensus Weaknesses

1. **[Title / Abstract / §1 / §3] The paradox's second horn is entirely uncited.** The
   claim that human ventral-stream representations are texture-like appears eight times
   — "have been described as", "well described as", "better described as" — with **no
   author, no year, no study, anywhere in the manuscript**. This is no longer one gap
   among many in a reference-free draft: it is the load-bearing premise of the title,
   the abstract, the introduction and the discussion. If that literature does not say
   what the paper needs it to say, the paradox does not exist and the paper has no
   motivation. **This is the single most urgent item in the manuscript.**
2. **[§1, §2.2, Limitation 9] Even granting the literature, the model realises
   "texture-biased" as *chromatic* (L−M), and the human finding is not about
   chromaticity.** Limitation 9 concedes the correspondence is "by role, not by content"
   — but the title, abstract and introduction carry no such qualification, and the
   paradox argument is weaker than they imply: the paper dissolves *a* paradox in *its
   own* terms and argues by analogy of role to the human one. The existence-proof logic
   survives; the framing does not currently carry its own caveat.
3. **[§2.3, §4.3] The paper's new keystone is not reproducible as written, and its
   framing overreaches.** Methods §4.3 says streams are read out "individually" and
   "jointly" and stops there. It does not say that `full` is a **fixed 50/50 logit-level
   sum** (`stream1_weight = 0.5`, a hyperparameter, never learned — confirmed in
   `src/models/pytorch_models.py:32-46` and `src/analysis/behavior/cue_conflict.py:126-145`),
   nor that `magno_only`/`parvo_only` are that same sum with one term zeroed. Given
   w = 0.5 hard-coded, the observed 0.675 is a *deterministic function* of the two
   streams' logits: there is no residual against which "mixes rather than inherits" could
   have failed. The prose — *"A readout that simply inherited would sit at 0.878; one
   reading the texture-biased stream would sit near 0.418. It does neither."* — names
   two alternatives that correspond to w = 1 and w = 0, neither of which this
   architecture can produce. **This is the same defect class §2.6.1 was just corrected
   for: an arithmetic near-necessity presented as a prediction that could have failed.**
   See Suggestion 4 — the fix is cheap and turns this into the paper's best result.
4. **[Whole manuscript] Still no References section — zero citations.** Geirhos is named
   eleven times, Sincich and Kastner once each, with no reference list. A referee cannot
   assess novelty, priority or attribution. Return-to-authors condition at any
   Nature-portfolio journal.
5. **[§2.3, §2.5, §4.7] `±` is never defined.** It carries §2.5's headline table
   (1.402 ±0.089) and §2.3's 0.675 ±0.025. §2.5's entire argument is that within-arm seed
   ranges exceed between-arm gaps, which cannot be evaluated without knowing whether ±
   is SD, SEM or a CI, and over what unit. Statistical-rigor item 16; listed as fatal.
6. **[§4.7] "Pre-registered" is an internal lab document, not a registry.** The
   predictions were written into `PAPER_PLAN.md` before the analyses ran, and git
   timestamps make that checkable — which is better than most claims of this kind and
   worth saying. But it is not preregistration in NHB's sense. The word appears
   throughout and is the paper's principal credibility asset. Re-word to "internally
   pre-specified, timestamped in version control" and give the commit hashes.
7. **[§4.7] 36 reported p-values, ~40 subject-level 8/8/0-of-8 claims, no multiplicity
   control.** No family is defined; no FWER or FDR is stated anywhere. The disclosed dead
   claims (hMT+, eccentricity, dorsal/ventral, VO) confirm a screening design was run.
   The two-statistics rule is a robustness check, not multiplicity control.
8. **[Format] Overruns, and worse than the prior draft.** Main text **9,555 words**
   against NHB's ≈4,000–6,000 (up from 7,956 — the reframe added framing before anything
   was cut). Abstract ~330–400 against ~150. **11 tables + 2 figures = 13 display items**
   against a ≤8 cap (up from 12).
9. **[Presentation] The Results are eleven dense numeric tables and essentially no
   figures.** Only two figures exist and the text contains two figure callouts across
   seven result sections. §2.1's two-route asymmetry, §2.3's mixture, §2.6.2's degradation
   curves and §2.6.3's shape-bias-vs-κ scatter are all *inherently visual* results
   delivered as tables. For an adjacent-field reader this is the difference between
   following the argument and not.
10. **[No section] The single-stream control is still absent, and the reframe makes it
    first-order.** The new claim is that behaviour reads out from *both* streams. The
    first question any referee asks is whether one stream on the same front end would
    behave the same. PLAN §72 reportedly has the answer (split buys 0.088 shape bias,
    costs 0.043 accuracy, at half the parameters, complete separation under both
    checkpoint policies) and it is not in the draft.
11. **[Ethics / Open science] No ethics, consent, secondary-use, demographic or
    constraints-on-generality statement** for NSD (n=8) or the Geirhos observers (n=10);
    **no data, code, materials or Reporting Summary statements**; the Geirhos licence is
    unverified by the draft's own Open Items.
12. **[§2.4, §2.2] Several load-bearing cells remain n=1.** The `swap` and `parvo_highsf`
    cortical fits are seed 0 only; the accuracy-matched control is one training run; the
    untrained references carry no seed; the konio control is untrained and single-seed.
    Each is labelled — the disclosure is genuinely good — but the count of n=1 arms is
    high for a paper whose Open Items claim "no n=1 claim in its main results".
13. **[§3] The developmental framing still outruns the manipulation.** The front end is
    static at a fixed 48 months (§4.1); the Discussion is headed "What a developmental
    constraint actually does"; the one trajectory-trained checkpoint is a different
    codebase and the authors call it the least comparable arm. Carried over unchanged.
14. **[Throughout] Anatomical labels do rhetorical work the model does not support.**
    "Magno" and "parvo" are luminance and L−M inputs to two identical ResNet-18s; temporal
    dynamics, contrast gain, receptive-field size and cortical targets are not modelled.
    §3's "Against channel essentialism" argues this while the labels persist. Carried over.

## Author Questions

1. **Which studies establish that human ventral-stream representations are texture-like,
   and do they support the claim at the generality the abstract states it?** Name them,
   with region, species and stimulus class. If the finding is specific to particular
   stimulus regimes or to macaque IT rather than human ventral cortex, the paradox needs
   restating at that narrower scope — which it can survive, but not silently.
2. **What is the fusion weight, was it learned, and what happens when you sweep it?**
   State w in Methods. Then report shape bias as a function of w on an existing
   checkpoint. If the curve runs smoothly from ≈0.418 to ≈0.878 with no component
   changing, that is the readout account's central prediction measured rather than
   asserted, and §2.3 becomes the paper's strongest section instead of its weakest joint.
3. **Does a single-stream network on the same diet front end behave shape-biased?**
   PLAN §72 appears to have this. Without it, "why two streams at all?" is unanswered.
4. **What does `±` denote, over what unit, and how many tests are in the family?**
5. **Under what approval and consent were the two human datasets used, and what are the
   demographics and the constraints on generality?**

## Statistical Rigor Audit

| Item | Verdict | Note |
|---|---|---|
| Sample size justified (power / precision) | **Fail** | No a priori justification for 3 seeds or 8 subjects. Exact-test floors are stated honestly (p=.0078 at n=8; 0.100 at 3v3) but a floor is not a power analysis; §2.6.3's VO entry now correctly reports the 80%-power detectable effect. |
| n and unit of analysis defined; nesting handled | **Pass** | §4.7 is exemplary: subjects and seeds never pooled; seed-level claims require all three within-seed subject tests. Limitation 6 correctly flags PR n=8 as image-samples. |
| Test choice, assumptions, one/two-sided | **Pass** | Paired exact Wilcoxon and exact Mann–Whitney are right for these n; two-sided throughout. |
| Multiple comparisons declared and corrected | **Fail** | 36 p-values, ~40 subject-level claims, no family defined, no FWER/FDR anywhere. |
| Effect sizes + CIs; error bars defined | **Fail (fatal)** | `±` undefined manuscript-wide on headline tables. Some dz and CIs reported (dz −11.70; TOST 90% CIs); most claims carry none. |
| Exact p / test statistics / df / software | **Partial** | Exact p given; df given for the Welch equivalence; software versions absent. |
| Robustness: sensitivity, multiverse, out-of-sample | **Pass** | Checkpoint-policy sensitivity run explicitly and both policies reported; two-statistics rule; encoding PCA fit on train only, shared-515 held out and never used for selection. |
| Causal language matched to design | **Pass** | Careful throughout; the existence-proof scope is stated repeatedly and the single-stream alternative is explicitly not excluded. |
| Measurement validity / reliability reported | **Partial** | The κ pipeline is validated against published values before use (good). Probe/PR reliability not reported. |
| Null results with equivalence bounds | **Pass** (new) | Four TOST results with bounds fixed in advance, each carrying its own caveat. This was the prior pass's weakness #5 and it is closed. |

## Open Science Audit

- **Preregistration:** internally pre-specified in `PAPER_PLAN.md` before each analysis,
  verifiable by git timestamp; **no public registry, no DOI**. Primary hypotheses,
  outcomes and analysis plan do match the paper, and two of four predictions failed and
  are reported as failures. **Deviations disclosed: yes** — unusually thoroughly.
  *Verdict: Partial, and must be re-worded.*
- **Data:** **Absent.** No statement. NSD is restricted-access (data-use agreement);
  Geirhos observer CSVs were fetched without LICENSE or README and the licence is
  unverified by the authors' own Open Items.
- **Code:** **Absent as a statement.** The analysis code exists and is specific
  (Methods names seven modules by path), which is a good base for a real statement.
- **Materials:** **Partial.** Cue-conflict stimuli are Geirhos's public release; licence
  unconfirmed.
- **Reporting Summary (BSS):** not provided.
- **Replication:** internal — 3 seeds, 8 subjects, both statistics, plus a cross-codebase
  check (−0.8618 vs −0.8593 from an independent thesis checkpoint, p=.84). No independent
  replication. Stated honestly.

## Human Participants and Human Groups Audit

- **Ethics approval named; informed consent; compensation:** **No** for both datasets →
  `EthicsFlag: true`. Secondary use of existing human data still requires a statement.
  Fixable by citing the source studies' approvals.
- **Sample demographics:** **None** for either dataset (Geirhos n=10 observers; NSD n=8).
- **Constraints-on-generality statement:** **Absent.** Both samples are small, Western and
  convenience-recruited; the paper's claims are about "humans".
- **Sex/gender and race/ethnicity (SAGER):** **n/a as analysed** — no participant variable
  is analysed — but demographics still need reporting.
- **Risk of harm or stigmatisation to a group:** **None.** No group-level claims.
- **Vulnerable populations / sensitive data:** **n/a**; NSD is de-identified neuroimaging
  under its own agreement, which should be cited.

## Specific Improvement Suggestions (ranked, with feasibility)

1. **Cite the paradox.** Name the studies behind "ventral-stream representations are
   texture-like", check that each supports the claim at the stated generality, and narrow
   the abstract's phrasing to what they support. *(1–2 days; no new data. Do this first —
   nothing else in the paper can be assessed until the premise has a source.)*
2. **Write the References section.** ~50–70 checked references. *(1 week; no new data.)*
3. **Define `±`, declare the test family, state the multiplicity control.** Three
   sentences in §4.7 plus a pass over every table legend. *(1 day; removes one of the two
   binding caps.)*
4. **Sweep the fusion weight — the highest value-per-hour item in the paper.** `w` is a
   plain scalar in the forward pass and `cue_conflict.py` already reads it, so evaluating
   cue-conflict shape bias at w ∈ {0, 0.1, …, 1.0} on existing checkpoints is CPU-only
   minutes on `--partition=main`. This instantiates the Discussion's own stated "natural
   next test" — *manipulating the weighting should move the rate without moving any
   component* — as a **dose-response curve**, and converts §2.3 from an arithmetic
   demonstration into a measurement. A `dkl_bcs_mheavy` variant trained at w=0.9 is
   already registered in `standalone_trainer.py`, giving a second arm where the weighting
   differs at *training* time. *(≈2 CPU-hours; no new data, no GPU.)*
5. **Add ethics, data, code and materials statements**, and confirm the Geirhos licence at
   source. *(1–2 days; removes the second binding cap.)*
6. **Add PLAN §72's single-stream control** to §2.1 or §2.3. *(Already computed; hours.)*
7. **Cut to length and build the figures.** Move the six self-correction passages to
   Supplementary (~1,500–2,000 words — they are project history, not results); convert
   §2.1, §2.3, §2.6.2 and §2.6.3 from tables to panels. This addresses the word count,
   the display-item cap and weakness 9 in one pass. *(1–2 weeks.)*
8. **Cut the abstract to ~150 words** and drop most of its numbers. *(Hours.)*
9. **Carry Limitation 9's role-vs-content caveat into the abstract and introduction.**
   *(Hours; materially reduces the overclaim risk in weakness 2.)*

==========================================
## Comments to Authors
==========================================

### Summary of key messages

The manuscript poses a genuine tension — human cue-conflict behaviour is shape-biased
while ventral-stream representations have been characterised as texture-like — and argues
that the tension dissolves if a behavioural rate is a property of a readout over
components rather than a report of any single representation. It supports this with a
two-stream network receiving approximations of the magnocellular and parvocellular
signals: the two streams end training with opposite shape–texture biases (+0.819,
−0.518), the texture-biased stream is the one whose unique variance carries the ventral
end of the hierarchy in eight NSD subjects (−0.862 against −0.101 without the
developmental front end), and the integrated readout lands shape-biased at 0.675 while
the shape-biased stream alone reaches 0.878. Three properties usually treated as
travelling with shape bias — dimensionality, OOD robustness and trial-level agreement
with ten human observers — are shown not to travel with it, each now with an equivalence
bound.

The argument is well organised and the standard of self-scrutiny is unusually high. The
untrained baselines in §2.1, the explicit identification of the two-trunk comparison as
an algebraic identity, the identity-check labelling in §2.6.1, and the *Claims
deliberately not made* section (which retracts three of your own earlier results) are all
to your credit and are not common. My substantive concerns are three: the premise is
uncited, the keystone section is under-specified in Methods and over-claimed in prose,
and the manuscript is roughly twice its target length with almost no figures.

### Major comments

**M1. The claim that human ventral-stream representations are texture-like carries the
entire paper and has no citation.** It appears eight times as "have been described as" or
"well described as". Please name the studies, and state for each the region, species and
stimulus class. My concern is not only attribution but scope: to my reading, the relevant
findings are considerably more specific than the blanket characterisation in your
abstract, and several concern macaque IT rather than human ventral cortex. If so, the
paradox should be posed at that narrower scope. It can survive being narrowed; it cannot
survive being unsourced.

**M2. The model's texture-biased component is chromatic, and the ventral texture finding
is not about chromaticity.** Limitation 9 concedes that the correspondence is "by role,
not by content", which is the right concession — but it appears only in the Limitations.
The title, abstract and introduction present the resolution without it. Please carry the
qualification into the abstract and §1, and state explicitly that the argument establishes
that the reported tension *need not* be a contradiction, rather than that this is how the
human system resolves it. §2.3's closing paragraph already does this well; the front
matter does not.

**M3. §2.3 is the paper's keystone and it is neither fully specified nor correctly
framed.** Methods §4.3 states that streams are read out individually and jointly, and
stops. From the code, `full` is a fixed 50/50 logit-level sum with `stream1_weight = 0.5`
— a hyperparameter that is never learned — and `magno_only`/`parvo_only` are that same
sum with one term zeroed. This must be in Methods; as written the section cannot be
reproduced. It also changes how the result reads. You write that a readout which
"inherited" would sit at 0.878 and one reading the texture-biased stream would sit near
0.418, and that the observed 0.675 "does neither". Those two alternatives correspond to
w = 1 and w = 0. With w fixed at 0.5, an intermediate value is very close to
arithmetically forced, and 0.675 is a deterministic function of the two streams' logits —
there is no residual against which the claim could have failed. This is the same issue you
correctly identified and repaired in §2.6.1's ×1.00 cells.

The remedy is inexpensive and would make this your strongest section rather than your
weakest. `w` is a scalar in the forward pass and your own evaluation code already reads
it. Sweep it: report cue-conflict shape bias at w from 0 to 1 on the existing checkpoints.
Your Discussion already names this as the readout account's central testable prediction —
*"manipulating the weighting should move the rate without moving any component"* — and it
is a few CPU-hours away. A smooth curve from ≈0.418 to ≈0.878 with every component held
fixed would be a direct measurement of the mechanism you are proposing. I would also note
that a `dkl_bcs_mheavy` condition trained at w = 0.9 is already registered in your
training code, which would give a second arm where the weighting differs at training time.

**M4. There is no reference list.** No claim of novelty or priority can be assessed. At a
Nature-portfolio journal this is a return-to-authors condition on its own.

**M5. `±` is not defined anywhere.** It carries §2.5's headline table and §2.3's
0.675 ±0.025. §2.5's argument turns on whether within-arm seed ranges exceed between-arm
gaps, which cannot be evaluated without knowing what the interval is and over which unit
it was computed.

**M6. Multiplicity.** I count 36 reported p-values and roughly forty subject-level
8/8-type claims, with no family defined and no correction stated. Your *Claims
deliberately not made* section makes clear that a screening design was run — which is the
honest thing to disclose, and exactly why the family needs defining. The two-statistics
rule is a valuable robustness check but it is not multiplicity control.

**M7. "Pre-registered" needs re-wording.** The predictions were written into a
version-controlled plan before the analyses ran, and the git history makes that
checkable — which is more than most such claims offer and is worth stating in those terms.
But it is not a registry entry, and the word carries a specific meaning to this
readership. Please describe it as internal pre-specification with commit references.

**M8. Ethics, data, code and materials statements are absent** for two human datasets
(NSD n=8; ten Geirhos observers). Secondary use still requires an ethics statement, and
NSD is restricted-access. Neither sample has reported demographics, and there is no
constraints-on-generality statement. Your own Open Items note that the Geirhos licence is
unverified; please confirm it at source before submission.

**M9. Why two streams?** Your reframed claim is that behaviour reads out from *both*
streams. The obvious control — a single-stream network on the same front end — is not
reported, and I understand from the manuscript's Open Items that the comparison already
exists. Under this framing it is the first thing a reader will ask.

**M10. Length and display items.** Main text is ~9,555 words against roughly 4,000–6,000,
with 11 tables and 2 figures against a cap of 8 display items. The six passages
documenting corrections to your own earlier analyses are, I think, the right material to
move: they are project history rather than results, and moving them to Supplementary
recovers most of the excess without losing anything a reader needs. I would not want the
*disclosures* themselves cut — only relocated.

**M11. The Results are almost entirely tables.** §2.1's two routes, §2.3's mixture,
§2.6.2's degradation curves and §2.6.3's shape-bias-against-κ relationship are all
naturally figures, and as tables they are very hard to read. This is the main obstacle to
a non-specialist following what is otherwise a well-structured argument.

**M12. Several load-bearing cells are n=1** — the `swap` and `parvo_highsf` cortical fits,
the accuracy-matched control, the untrained references, the konio control. Each is
labelled, which I appreciate. But the Open Items state the paper "now has no n=1 claim in
its main results", and that is not quite right as the manuscript stands.

**M13. The developmental framing outruns the manipulation.** The front end is static at a
fixed 48 months, while §3 is headed "What a developmental constraint actually does". The
single trajectory-trained checkpoint is from a different codebase and you describe it as
the least comparable arm. Consider framing the transform as a developmentally *motivated*
low-pass constraint rather than a developmental trajectory.

### Minor comments

- m1. Abstract is ~330–400 words against a ~150 target, and is dense with numbers. Most
  belong in Results.
- m2. "Magno"/"parvo" carry anatomical implications the model does not implement (no
  temporal dynamics, contrast gain, receptive-field size or cortical targets). §3's
  "Against channel essentialism" makes this point while the labels persist throughout.
- m3. Undefined on first use for a general readership: DKL, L−M, participation ratio,
  banded ridge, `nsdgeneral`, b3 betas, Uₘ/Uₚ, VSB.
- m4. Software and package versions are not reported.
- m5. §2.4's post-hoc exclusion of `parvo_highsf` seed 0 from the behavioural row is
  disclosed but was not pre-specified; please say so explicitly at that point.
- m6. Two figure callouts across seven result sections; the existing figures are barely
  integrated into the argument.
- m7. The status header and inline `> TODO` notes should come out before submission.

### Notes by criterion

- **Validity.** No flaw prohibits publication. The analyses I could check against the
  code are as described, and the encoding pipeline's out-of-sample discipline is sound.
  The §2.3 framing issue (M3) is a matter of interpretation and Methods completeness, not
  of a computation being wrong.
- **Originality and significance.** The specific demonstration — a subcortically-split
  model, fitted to human cortex, in which the ventral-explaining component is
  texture-biased while behaviour is shape-biased — is new as far as I know. The
  underlying principle that readout ≠ representation is not new, and the manuscript
  would be strengthened by positioning itself against that literature explicitly rather
  than presenting the inference as generally unexamined.
- **Data and methodology.** Sound and unusually well documented, but not currently
  reproducible from the manuscript alone: the fusion weight, the ± definition, and the
  data/code statements are all missing.
- **Statistics.** Appropriate tests, correct nesting, honest reporting of exact-test
  floors, and four equivalence bounds with their limitations stated. Multiplicity and
  error-bar definition are the gaps.
- **Conclusions.** Generally well calibrated and repeatedly scoped ("existence proof",
  "not a measurement of human cortex"). Two exceptions: the uncited premise (M1) and
  §2.3's "it does neither" (M3).
- **References.** Absent.
- **Clarity and context.** The four-step structure is a real improvement. Length,
  table density and the absence of figures work against it.

==========================================
## Confidential Comments to the Editor
==========================================

**Triage.** Send out. The first version of this manuscript was, in my view, a
computational-vision paper with human data used as a benchmark, and I would have
transferred it. The reframe changes the conclusion to an inferential claim about what
behavioural rates license — which is a claim about how human behavioural science should
interpret its own most-cited alignment measure — and §2.6.3 supports it with human data
(ten observers, error consistency against a human–human ceiling). That clears the scope
gate. The breadth test now passes at two constituencies rather than one, though the
second is thinly served.

**Significance.** Borderline, and honestly so. Against the heuristics it satisfies one
clearly — *it sharply refines a widely held belief with proportionate evidence*: shape
bias is used across vision science and ML as a proxy for shape-like representation, and
this paper separates the two with a mechanism. It partially satisfies a second (*changes
how researchers analyse studies of X*), because the practical upshot — do not infer
representational content from a behavioural rate — is directly actionable. It does not
satisfy the others: no new measurement capability, no policy reach, no preregistered
multi-site test. One-and-a-half of six heuristics is a borderline, not a yes.

**Recommendation.** Major revision. The score sits at 6.4, inside the band, and two hard
overrides cap it there independently: undefined headline error bars, and the missing
ethics statements. I want to be clear with you that **both caps are clerical.** Neither
needs new data, new participants or new compute. A competent revision could clear the
overrides in under a week. The substantive work is the reference list and the figures.

**What I would want to see before acceptance,** in priority order: (i) the paradox's
premise cited and, if necessary, narrowed to what the literature supports — this is the
one item that could change my recommendation downward if it turns out the literature does
not support the claim at the stated generality; (ii) the fusion-weight sweep in M3, which
is a few CPU-hours and would convert the paper's keystone from an arithmetic demonstration
into a measured dose–response curve — if it comes out as predicted, my Advance score would
move up; (iii) references, ± definition, multiplicity, ethics/data/code; (iv) a real
figure programme and a cut to length.

**Revision time:** 2–3 months. **New participants: no. New compute: a few CPU-hours.**

**A note on the authors' conduct.** This manuscript retracts three of its own earlier
results, reports two of four pre-specified predictions as failures, corrects two errors
in its own previous draft in visible footnotes, and maintains a section listing claims it
deliberately does not make. Several of the weaknesses above are ones I found *because the
authors disclosed them*. That should count in their favour, and I would not want the
volume of my comments to obscure it.

**Expertise.** High on visual neuroscience, encoding models and model–human comparison;
moderate on developmental vision. **Conflicts:** none.

## Per-reviewer reviews (JSON)

```json
{
  "Persona": "R1 Subfield",
  "ArticleType": "Article",
  "Summary": "Reframed around a real tension in vision science; the four-step argument works and the two-route asymmetry in 2.1 is the strongest new material. The premise carrying the whole paper is uncited, and the model's texture-biased component is chromatic where the human finding is not.",
  "Strengths": ["Paradox is genuine and the resolution is mechanistic", "2.1's untrained baselines separate input from learning", "Construction guarantee correctly stated as an identity", "2.6.1's identity-cell overclaim repaired", "Three retracted analyses recorded"],
  "Weaknesses": ["[Title/Abstract/1/3]: the texture-like ventral premise has no citation anywhere, in 8 occurrences", "[1,2.2,Lim9]: texture-biased component is chromatic; human finding is not; caveat sits only in Limitations", "[none]: single-stream control absent though computed", "[whole]: no reference list", "[3]: developmental framing outruns a static 48-month front end", "[throughout]: magno/parvo labels imply unmodelled properties"],
  "Questions": ["Which studies, at what region/species/stimulus scope, establish the texture-like ventral claim?", "Does a single-stream network on the same front end behave shape-biased?", "How would you position this against the existing readout-vs-representation literature?"],
  "Advance": 3, "Soundness": 3, "Evidence": 3, "Relevance": 3, "Statistics": 3,
  "OpenScience": 2, "Ethics": 2, "Clarity": 2,
  "Triage": "send out", "SignificanceGate": "borderline",
  "BreadthTest": ["vision science / cognitive neuroscience", "behaviour-to-representation inference"],
  "Recommendation": 3, "Confidence": 4, "EthicsFlag": true,
  "RevisionTime": "2-3 months", "NewDataRequired": false,
  "PreregistrationDeviations": ["Internal pre-specification only; no registry. Deviations themselves disclosed thoroughly."],
  "CommentsToAuthors": "See M1, M2, M9, M13 and the originality note.",
  "ConfidentialToEditor": "Reframe clears the scope gate the first version failed. Major revision; the citation item is the one that could move me down."
}
```

```json
{
  "Persona": "R2 Adjacent-field",
  "ArticleType": "Article",
  "Summary": "The lesson — a behavioural rate does not license a representational description — is portable and I now see why my field should care, which I did not in the previous version. But the manuscript is written for vision-model specialists: 9,555 words, 11 tables, 2 figures, and an abstract that is mostly numbers.",
  "Strengths": ["A portable conclusion, clearly stated", "Four-step structure is followable", "Costs and failures reported openly"],
  "Weaknesses": ["[Format]: 9,555 words vs 4,000-6,000; 13 display items vs 8; abstract ~330-400 vs ~150", "[Results]: eleven tables, two figures, two figure callouts across seven sections", "[Results]: DKL, L-M, participation ratio, banded ridge, nsdgeneral, b3, Um/Up all undefined for a general reader", "[whole]: no reference list", "[1]: the premise a non-specialist most needs to trust is the one with no source"],
  "Questions": ["Can the argument be told in four figures?", "What should a psychologist who does not build vision models do differently after reading this?"],
  "Advance": 3, "Soundness": 3, "Evidence": 3, "Relevance": 2, "Statistics": 3,
  "OpenScience": 2, "Ethics": 2, "Clarity": 2,
  "Triage": "discuss", "SignificanceGate": "borderline",
  "BreadthTest": ["vision science / cognitive neuroscience", "psychology / model-human alignment"],
  "Recommendation": 2, "Confidence": 3, "EthicsFlag": true,
  "RevisionTime": "2-3 months", "NewDataRequired": false,
  "PreregistrationDeviations": [],
  "CommentsToAuthors": "See M10, M11, m1, m3.",
  "ConfidentialToEditor": "Significance moved from no to borderline on the reframe. My recommendation stays at transfer because the delivery, not the idea, is what fails an NHB readership; a figure programme and a length cut would move me to major revision."
}
```

```json
{
  "Persona": "R3 Methods/Stats/Open-science",
  "ArticleType": "Article",
  "Summary": "Statistically among the more disciplined manuscripts I have reviewed: correct nonparametric tests, nesting handled, exact-test floors reported as floors, four pre-fixed equivalence bounds each with its own caveat, and two of four pre-specified predictions reported as failures. The gaps are clerical but two of them are fatal-listed, and the keystone section is under-specified in Methods and over-framed in prose.",
  "Strengths": ["Four TOST bounds with pre-fixed intervals and honest caveats (Welch df 2.3; VO power)", "Subjects and seeds never pooled; seed claims require all three within-seed tests", "Encoding out-of-sample discipline: PCA on train only, shared-515 never used for selection", "Checkpoint-policy sensitivity run and both policies reported", "Kappa pipeline validated against published values before use"],
  "Weaknesses": ["[2.3,2.5,4.7]: plus-minus undefined manuscript-wide on headline tables (fatal-listed)", "[4.7]: 36 p-values, ~40 subject-level claims, no family defined, no FWER/FDR", "[2.3,4.3]: full is a fixed 50/50 logit sum with stream1_weight=0.5 never learned; not stated in Methods; 'it does neither' names w=1 and w=0 alternatives the architecture cannot produce", "[4.7]: 'pre-registered' denotes an internal plan file, not a registry", "[whole]: no data, code, materials, ethics or Reporting Summary statements; Geirhos licence unverified", "[2.2,2.4]: swap/parvo_highsf cortical fits, accuracy-matched control, untrained references and konio control are each n=1"],
  "Questions": ["What does plus-minus denote and over what unit?", "How many tests are in the family and what correction applies?", "What is w, was it learned, and what is shape bias as a function of w?", "Under what approvals were NSD and the Geirhos observer data used?"],
  "Advance": 3, "Soundness": 3, "Evidence": 3, "Relevance": 2, "Statistics": 3,
  "OpenScience": 2, "Ethics": 2, "Clarity": 2,
  "Triage": "send out", "SignificanceGate": "borderline",
  "BreadthTest": ["vision science / cognitive neuroscience", "meta-science of model-human comparison"],
  "Recommendation": 3, "Confidence": 5, "EthicsFlag": true,
  "RevisionTime": "2-3 months", "NewDataRequired": false,
  "PreregistrationDeviations": ["Internal pre-specification, git-timestamped; must not be called preregistration unqualified."],
  "CommentsToAuthors": "See M3, M5, M6, M7, M8, M12.",
  "ConfidentialToEditor": "Validity veto engaged on the error-bar item only; it is clerical and clears in a day. The equivalence work added since the previous version closes my largest prior objection. The w-sweep in M3 is a few CPU-hours and would materially strengthen the paper's central section."
}
```

---

*This review was generated by an LLM-assisted reviewer simulation for the authors' own
self-improvement; the numeric rubric is internal to the simulation and is not an NHB form.*
