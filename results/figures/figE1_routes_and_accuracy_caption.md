# Extended Data Figure 1 — caption

**The two routes to stream specialisation (§2.1), and the accuracy control
(§2.4).**

Both panels sit in Extended Data for the same reason: each rests on **one model
per cell**. The untrained references carry no training seed at all, and the
accuracy-matched arm is a single retrained run. The main text states the
corresponding claims at that strength, and this figure is drawn to match.

**(a) The shape-biased component is supplied by the input; the texture-biased
component is built by learning.** Linear-probe shape−texture bias per stream,
before and after training (zero is the sign flip, so it is interpretable as a
side rather than only as a magnitude). With the developmental transform applied,
the magno stream is **already** shape-biased before any weight is updated and
training adds **+0.014**; the parvo stream begins near neutral and acquires its
texture preference over training (**−0.395**). Numbers at the right are each
line's change over training. Shaded bars at the trained end are the 3-seed
range; the untrained points carry no interval because those references have no
seed, and drawing one would manufacture it.

The third line is what makes this causal rather than descriptive. **With the
transform removed, ordinary training does not merely fail to build shape bias —
it destroys the shape bias the input supplied**, carrying the stream from +0.417
across the sign flip to −0.219 (**−0.636**). Nothing in this paper claims that
learning creates the shape-biased component.

**(b, c) Matching accuracy does not recover the shape bias.** The diet-trained
models are less accurate than the no-diet controls (§2.6.1, §2.7), so the
conjunction effect of §2.4 could in principle be an accuracy effect. The fatal
outcome was pre-registered before the run: *a no-diet model at the diet's
accuracy showing a shape-biased magno stream would mean §2.4 is an accuracy
artefact and this paper's central claim is wrong* (PLAN §69.3–§69.4). No saved
no-diet checkpoint was low enough to match, so `dkl_base` was retrained with
per-epoch checkpointing and the epoch nearest the diet arm's 0.556 selected by
rule: epoch 10 at 0.548, matched to within 0.008.

Matching accuracy closes **7.6%** of the probe gap and **7.0%** of the
behavioural gap — two instruments built on different machinery agreeing to
within 0.6 percentage points. The matched model's probe bias remains
**negative**: still on the texture side of the sign flip. Lowering a no-diet
network's accuracy to the diet's moves it about a fourteenth of the way, and the
residual is reported rather than rounded to zero. **The conjunction effect is
not an accuracy effect.**

*Why two panels and not one with twin axes.* The probe (0 = sign flip) and
cue-conflict (0.5 = parity) scales have different zero points and different
units. Overlaying them on one frame would place each cue-conflict point at a
probe value it does not have; they are drawn on separate axes instead. The diet
reference in both panels is the **best-checkpoint** arm, because that is the
policy the matched control was selected under (PLAN §68, §71.6); reading it from
the final-checkpoint files would silently compare across policies.

Built by `tmp_analysis/figE1_routes_and_accuracy.py` from
`results/shape_texture{,_bestep,_matched}/` and
`results/cue_conflict{,_bestep,_matched}/`.
