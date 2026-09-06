# Reframe proposal — the shape-bias paradox

Written 2026-09-06 after re-reading `BA_Manocchio_Benito.pdf` (49 pp.) in full.
Supersedes the framing decision recorded in PAPER_DRAFT's "Positioning (decided 2026-09-02)".

## 1. What the BA actually asked

The thesis has **two** research questions. The paper inherited the second and dropped the first.

> **RQ1.** "Why is the ventral stream responsive to texture information, while human behavior
> is extensively shape-biased?"
>
> **RQ2.** "Can biologically inspired magnocellular- and parvocellular-driven streams inside an
> Artificial Neural Network diverge in their internal representations, while still showing
> clearly shape-biased behavior?"
> — BA §1.1

RQ1 is a question about **humans**. It names a real, cited contradiction in the human vision
literature: behaviour is shape-biased (Geirhos et al.), while ventral-stream regions carry
texture-like representations (Jagadeesh & Livingstone; Komatsu & Goda; Okazawa et al.).

The current draft answers RQ2 and never states RQ1.

## 2. The BA could not answer RQ1, and said so

> "In order to answer the first research question, **the results of this study are not expressive
> enough to draw conclusions** about why the human object identification behavior is
> predominantly shape-biased, despite different areas of the ventral stream seeming to be
> selective for texture-like features." — BA §5.6

It got as far as an existence proof and flagged it as speculation:

> "The evaluation of the M-heavy DKL B&CS model demonstrated that it is possible to exhibit a
> stream that is representationally biased towards texture information, while the overall
> behavior is mainly shape-biased, due to the influence of other network regions. ... **One
> interpretation of these results, therefore, is that the existence of texture-sensitive regions
> may not necessarily contradict the human shape-biased behavior.** ... While the findings cannot
> be generalized directly to the ventral stream, it raises the question if similar mechanisms
> might be present in the human visual system." — BA §5.3, §5.6

**Why it could not close it.** The BA's neural analysis was layerwise RSA against seven coarse
ROIs, and it returned a global stream preference rather than a regional split:

> "only one of the two streams shows better alignment with both ventral data and parietal data
> ... it is not evident that one model stream aligns more closely with a particular neural
> stream, while the other model stream exhibits a higher correlation with a second neural
> stream." — BA §4.4

That is the general-factor pathology PLAN §8.3 later diagnosed. RSA correlation cannot see a
regional split when one stream dominates everywhere.

## 3. The paper has since built exactly what RQ1 needed

Three things the BA lacked:

1. **Unique-variance partitioning instead of RSA correlation.** Uₘ and Uₚ remove the general
   factor. The draft's own global-preference control proves it: the diet and no-diet models sit
   on *opposite* sides of the global stream-bias axis (−0.495 vs +0.849), yet the regional
   gradient is in the model whose global preference points the wrong way (§2.3b). A global
   preference cannot produce that. **This is the analysis that unblocks the BA's null.**
2. **Per-stream readouts, separately and integrated.** magno-only 0.878–0.903, parvo-only,
   full 0.675.
3. **Seeds, controls, and a second statistic.** 3 seeds, six front-end controls, both statistics
   at the exact floor, cross-codebase replication to 0.0025.

## 4. The frame

> **Shape-biased behaviour does not require shape-biased representations.** A visual system
> whose behaviour is shape-biased can contain a texture-biased component — and in this model,
> that texture-biased component is precisely the one that carries the ventral end of the
> hierarchy. The paradox between shape-biased human behaviour and texture-like ventral
> representations dissolves once behaviour is treated as a readout over streams with opposite
> biases rather than as a report of one representation.

The four-step argument, entirely from results already on disk:

| step | claim | evidence | status |
|---|---|---|---|
| 1 | The system contains a shape-biased component | magno probe **+0.819** [0.014], cue-conflict **0.878–0.903** | 3 seeds, complete separation |
| 2 | It contains a texture-biased component | parvo probe **−0.482**; predicted-cortex VSB **−0.743** | 3 seeds |
| 3 | The **texture-biased** component is the one that explains the ventral end | hV4−V1v **−0.862** vs −0.101; Spearman ρ **−0.964** against hierarchical rank; hV4 the only parvo-preferring ROI of 20 | 3 seeds, both statistics, p=.0078, 8/8, six controls, survives blur removal |
| 4 | The integrated readout is nonetheless shape-biased, and integration **mixes** rather than inherits | full **0.675** against magno-only 0.878 — integration *costs* 0.203 | 3 seeds |

Step 4 is the keystone and is currently filed under *§2.7 What the diet costs* as a liability.
Under this frame it is the load-bearing evidence: if behaviour simply inherited from the
shape-biased stream, integration would be neutral. It is not — the readout demonstrably combines
components with opposite biases, which is the mechanism the frame claims.

## 5. What the reframe fixes

- **NHB scope.** The conclusion becomes a claim about human vision — why a well-documented
  behaviour/representation contradiction is not a contradiction. RQ1 is the paper's question and
  it is a human question.
- **Relevance (scored 1.7 in the referee sim).** Constituencies become human vision neuroscience
  (whose paradox this is) and cognitive psychology / behavioural measurement.
- **Kill risk #1 from the taste check is neutralised.** That learning adds only +0.014 to magno's
  shape bias (`PLAN:3153`, +0.796 untrained → +0.810 trained) is fatal to a title claiming a
  *developmental transform* produces shape bias. It is **irrelevant** to a claim about how a
  multi-stream readout behaves — and arguably supportive, since in the biology the magno/parvo
  split is given by the retina rather than learned. The untrained baseline still gets reported;
  it stops being load-bearing.
- **§2.6 stops being a wound.** It becomes the boundary condition: this account explains the
  shape-bias *rate*, and rate is explicitly not decision structure. The paper says what its
  mechanism does and does not cover.
- **The three dissociations acquire a common parent.** Dimensionality, robustness, and human
  agreement all fail to travel with shape bias because shape bias is a readout property, not a
  representational one. One lesson instead of three.

## 6. What the reframe costs

- **§2.1 and §2.2 demote from headline to mechanism.** They stop being the claim and become the
  answer to "how do you get a shape-biased component in the first place?" (§2.1) and "why does it
  survive training?" (§2.2). Both stay in Results; neither leads.
- **The developmental framing recedes.** The diet becomes the manipulation that produces the
  component, not the paper's subject. This is honest: the front end is static at 48 months.
- **Step 3 must be stated relationally, not absolutely.** hV4's absolute parvo preference is weak
  (−0.012, 2/8, per *Claims deliberately not made*). The strong statistic is the **gradient**
  (ρ = −0.964, hV4−V1v = −0.862), so the claim is "the parvo stream's explanatory share rises
  monotonically toward the ventral end", never "hV4 is parvo".
- **The identification is by role, not by content.** Our texture-biased stream is chromatic
  (L−M); the ventral texture-like representations in the cited human work are not specifically
  chromatic. The mapping is "the component that explains ventral cortex", not "the same features".
  This must be a stated limitation, not a silent equivalence.

## 7. Honest limits that go in the claim wording

- This is a model-based existence proof plus a cortically-grounded demonstration. It shows the
  paradox **need not** be a paradox and gives a mechanism that reproduces it. It does not
  establish that human cortex uses this mechanism. Wording must be "is compatible with" /
  "dissolves under", never "explains why humans".
- The alternative — that ventral texture representations feed shape-biased behaviour directly,
  with no multi-stream readout — is not excluded by these data.
- Parvo's texture organisation is diet-independent (VSB −0.743 vs −0.748), so step 2 is a
  property of the chromatic channel rather than of the diet.
- No time course. fMRI only.

## 8. Citation to verify before writing

The BA cites **Jagadeesh & Livingstone (2024)** for texture-like object representations in human
visual cortex; also Komatsu & Goda (2018) and Okazawa et al. (2015). The year and venue must be
checked at source before this becomes the paper's opening paradox — it is load-bearing under the
new frame, and the draft has no reference list yet.
