# Figure 5 — caption

**Matching the human shape-bias rate is not matching human decisions.**

**(a) The rate moves 3×; trial-level agreement does not.** Each point is one
readout: 21 values of the mixture weight `w`, for 3 seeds × 2 checkpoint
policies of the diet arm and 3 seeds of the no-diet arm — **189 readouts in
total**. The x-axis is cue-conflict shape-bias rate; the y-axis is error
consistency (Cohen's κ) against the same 10 human observers, as a percentage of
the human–human ceiling (κ = 0.234, 45 observer pairs). Because `w` moves the
rate over **frozen** representations (§2.3), the horizontal spread within one
arm is not a difference between models — it is one network being re-read.
Points from a single run are joined so the sweep is visible as a sweep.

The dashed horizontal line is the best of all 189 readouts, **20.7% of the
ceiling** — with `w` chosen post hoc, per seed, to maximise agreement. The
readout weight was given every opportunity to succeed and did not. The three
published CNNs (Geirhos et al. 2020) are drawn on the same axes: varying
architecture, training data and objective lands in the same band that
re-weighting one frozen network already covers. The vertical dashed line is the
human rate (0.958), which no readout reaches.

**No regression line is drawn.** The x-position of every point was set by a free
parameter we chose, so a fitted slope would invite reading the scatter as an
estimated relationship. The claim is a bound, and it is drawn as one.

**(b) The measure does detect a correspondence when one is there.** An
equivalence interval bounds a null but cannot show the measure would have *seen*
an effect had one existed; this panel supplies that positive control. On the
same 1,200 images and the same 10 observers, with shape- and texture-category
effects removed and `w` chosen on half the images and scored on the held-out
half, the same machinery recovers **24.1%** of ceiling on *whether either cue
resolves* against **8.7%** on *which cue is picked* — a 2.8-fold gap (3.6-fold
on ranks), replicated under the other checkpoint policy (22.5% against 11.2%).
Each target is expressed as a percentage of its **own** split-half ceiling
(r = .624 and .728 after category removal, over all 126 exhaustive half-splits),
which is the only unit in which two different targets compare. Dots are seeds;
bars their mean.

**(c) …but the 2×2 withdraws the diet reading, and we draw the control that
kills it.** Panel (b) puts the diet arm above the no-diet arm on difficulty
correspondence with no seed overlap, which invites the claim that the
developmental front end makes the model's difficulty profile more human-like.
**We do not make that claim.** Each arm is evaluated with its own front end, so
the arms differ in test-time input as well as in learned weights. Completing the
2×2 — diet weights without the front end, no-diet weights with it — gives a
train/test **matching** pattern rather than a weights effect: both matched cells
are positive (24.1%, 9.2%), both crossed cells are not (−0.6%, −3.7%), and with
the front end held off the ordering reverses. The crossed cells are in any case
too degraded to arbitrate, resolving to neither cue on 70–85% of images at the
balanced readout against 49–63% for the matched cells. The design cannot
separate the learned representation from the input transform, because separating
them breaks the model.

*The 2×2 itself is clean.* The parvo-stream logits are bit-identical across the
front-end manipulation (max deviation 0.00e+00) while the magno-stream logits
differ by 49–80, so the failure in (c) is one of attribution, not of
implementation.

Built by `tmp_analysis/fig5_human_correspondence.py` from
`results/sweep_error_consistency/` and
`/scratch/byrong/mpk/perimage_behavior/results/2026-09-07_geirhos_perimage/`.
The three published CNN values are the only hardcoded numbers and are labelled
as published.
