# Figure 1 — caption

**A developmental visual diet installs stream specialisation; four instruments
agree on the contrast, and one of them is a cost.**

**Read this figure as a convergence on a contrast, not as four views of one
property.** Panels (a) and (c) measure different things and are known to
factorise differently over identical inputs: shape bias requires the
carrier×transform conjunction, while effective dimensionality is additive in the
two factors (PAPER_PLAN §47.4). The decisive case is the `parvo_highsf` front
end, where the parvo stream becomes the *more* dimensional stream while
remaining texture-biased — high dimensionality does not confer shape bias. What
the four panels agree on, each at the n=8 exact-test floor, is the diet-vs-no-diet
contrast specifically.

**(a) Shape bias requires the conjunction of an achromatic carrier and the
developmental transform.** Each line is one carrier; the x-axis is whether the
developmental low-pass transform is applied to it. Non-parallel lines are the
result: neither factor alone moves a stream across the neutral point (dashed),
and only the conjunction does. Points are independent training seeds (n=3, or
n=2 for the achromatic/no-transform cell); the thick line is their mean. Left,
16-way cue-conflict shape bias on the Geirhos set (1,200 images); right, a
linear shape−texture probe on stream features, where zero is the sign flip and
therefore interpretable. Two measurement modalities, four cells each, same
interaction.

*Denominator caveat.* Cue-conflict shape bias is computed only over images the
stream decides as shape *or* texture, and that count varies ~4× across cells
(mean 165/1200 for chromatic+transform, 270 chromatic/none, 583
achromatic/none, 488 for the conjunction). A stream deciding ~13% of the set
returns something near a coin flip by construction, so the three non-conjunction
behavioural cells should be read as "no strong preference" rather than as precise
values. The probe does not have this problem — it classifies every image — and it
gives the same pattern.

**(b) The dissociation is visible through human visual cortex.** Cue-conflict
images passed through per-stream voxelwise encoding models fitted on NSD yield a
per-voxel predicted shape-bias index; plotted is the magno−parvo difference,
median over voxels above a 0.1 noise-ceiling threshold. Grey lines pair the same
subject across the two models (n=8). The effect is carried entirely by the magno
stream (+0.791 with the diet vs −0.638 without — without the diet the magno
stream is *texture*-dominated); the parvo streams are indistinguishable (−0.743
vs −0.748). This is the encoding model's own verdict on (a), using no probe and
no readout.

**(c) The diet fixes which side of parity training ends on.** Effective
dimensionality ratio (participation ratio of magno features ÷ parvo features,
1,000 NSD images per subject). Left: each arm's front end before any training,
marker at the subject mean with the subject range as a whisker. Right: **one
point per training seed**, each the mean over 8 subjects, fanned from the front
end it started at. Counts are subject-seeds on the claimed side of parity
(diet arms above, no diet at or below) out of 24 = 8 subjects × 3 seeds.

The pale blue arm shows the input's asymmetry is not what training carries
forward: its front end is inverted — parvo is the more dimensional stream, below
parity in 8/8 subjects — and after training every seed has flipped above parity,
*overtaking* a model handed 1.40.

**No level, set point or compression factor is quoted anywhere in this panel,
and this is deliberate.** Seed replication (PLAN §58) showed the trained level is
not reproducible: the diet-on-luminance arm spans 1.603–2.507 across seeds and
the diet-on-red−green arm spans 1.995–4.383, a range of 2.39. Earlier versions
of this figure labelled one seed's level per arm and so asserted a set point;
they also supported a "channel-specific" reading that the overlapping seed
ranges cannot establish. Both are withdrawn.

What replicates is the side: **24/24 subject-seeds above parity for every one of
the three diet arms**, from front ends spanning 3.26×, against 20/24 at or below
it for no diet — which enters from the *most* magno-dominant front end in the
set. Seed-level separation between every diet arm and no diet is complete
(p=0.100, the exact two-sided 3×3 floor).

*This panel varies architecture alongside the front end.* The `parvo_highsf`
condition gives its second stream two input channels rather than one, so input
and first-layer fan-in change together and no existing model separates them. The
confound that would fake the result — a weakened second stream — is ruled out
(its 16-way accuracy *rises*, 0.486 vs ~0.379), but the comparison is not
architecture-matched and the claim carries that qualification.

**(d) What the diet costs.** Median r² normalised by the noise ceiling, over
reliable voxels, same 8 subjects paired. **The no-diet model predicts BOLD
better** — 0.464 against 0.405, a ~13% cost in explained variance, p=.0078 on
both statistics, 8/8 subjects. The diet's case rests on stream differentiation
(a–c), not on prediction accuracy, and this panel is included so that a figure
whose first three panels favour the diet does not omit the one measure that does
not.

---

**Statistics.** All tests are paired exact Wilcoxon within subject, n=8, for
which .0078 is the two-sided floor; "8/8" reports subject unanimity. Panel (a)'s
replication unit is the training seed, not the subject — no brain data enters it.
Panels (b)–(d) are within-subject over the same eight NSD subjects. Every claim
is made only where the normalised and raw statistics agree (PAPER_PLAN §8).

**Colour.** Okabe-Ito, validated for colour-vision-deficiency separation and for
contrast against the panel surface. Identity is never carried by colour alone:
every arm in (c) carries its own label and count, and every mean in (b) and (d) is directly labelled.

**Provenance.** `tmp_analysis/fig1_data.py` → `results/figures/fig1_data.npz` →
`tmp_analysis/fig1.py`. Sources: (a) PAPER_PLAN §25 and §51; (b) §43.1;
(c) §47.1, §49.2, §50.2 and §58; (d) §43.3.

**Correction history.** The `swap` cell in (a) is n=3, superseding the n=2 in
§25.1. `dkl_bcs_swap_seed2` was excluded there by the convergence guard while
still training; it was re-run after converging and now decides *more* images
than seeds 0 and 1. Including it moves the cell by +0.011 (behaviour, 0.484 →
0.495) and +0.010 (probe, −0.155 → −0.145) and changes no claim. See §51.
