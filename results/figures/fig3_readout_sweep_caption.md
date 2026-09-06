# Figure 3 — caption

**Shape bias is set by the readout, over representations that never change.**

**(A)** Cue-conflict shape bias as a function of `w`, the weight on the magno
stream in the readout (`full = classifier(w·y₁ + (1−w)·y₂)`; Methods §4.3). All
21 values of `w`, 3 training seeds, both checkpoint policies. Thin lines are
individual seeds; thick lines the seed mean. **Nothing is retrained, refitted or
reselected between the leftmost and rightmost point**: `y₁` and `y₂` are the two
streams' pre-classifier features and do not depend on `w`, so the sweep moves
the readout while holding both representations exactly fixed. Shape bias runs
from **0.399 to 0.904** (mean of the best saved checkpoints; 0.419 to 0.879 at
the final checkpoints), monotone non-decreasing across all 21 values in 3/3
seeds at the best saved checkpoint and 2/3 at the final, where one seed has a
single local inversion. The dotted rule marks `w` = 0.5, the published model,
which the sweep reproduces **exactly** per seed (0.6922 / 0.6861 / 0.6464 at the
final checkpoint) — the anchor establishing that the sweep and the main
behavioural pipeline are the same computation.

**(B)** The same axis, showing where each model in the comparison sits: one
network per point, each the product of its own training. The shaded band is the
range panel (A) reaches by reweighting alone. Published values for AlexNet,
GoogLeNet and VGG-16 are from the Geirhos et al. (2018a) release; human
observers are the ten who viewed these identical stimuli (mean 0.958, range
0.919–0.987), scored by the identical statistic (§4.3).

**What the figure claims, and what it does not.** It claims that a behavioural
shape-bias rate is a property of a readout over components rather than a report
of either component. It does **not** claim a fitted or preferred `w`: no value
of the weighting was estimated from human data, and the sweep does not identify
which mixture a human readout uses. The human point is drawn because it is the
reference, and because it is the one value the sweep does not reach — 0.904 at
`w` = 1.00 remains below the lowest individual human.

**Endpoints are not the ablations.** `w` = 1.00 and `w` = 0.00 are not formally
identical to the `magno_only` / `parvo_only` ablations, which hold `w` = 0.5 and
zero one term; the classifier is affine, so rescaling features moves the argmax
through the bias term. Empirically the difference is negligible (0.879 against
the ablation's 0.878; 0.419 against 0.418) but they are different quantities and
are not equated. The decided-image count varies across `w` (0.24–0.45 of the
1,200), so the extreme-`w` cells carry the denominator caveat of §2.4; the
monotone trend holds across the well-decided middle of the range regardless.
Both caveats in full: Extended Data §E3.

Built by `tmp_analysis/fig3_readout_sweep.py` from
`results/fusion_sweep/*_wsweep.json` (§2.3, PLAN §52.3, §20.3).
