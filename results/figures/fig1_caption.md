# Figure 1 — caption

**The developmental visual diet decides whether learning builds or erodes
cortical correspondence.**

**(A)** Stream preference contrast between hV4 and V1v (magno-unique minus
parvo-unique variance, normalised by their sum), before and after training.
Thin lines are individual NSD subjects (n=8); thick lines the group mean.
Both untrained networks start at the same value (−0.4877 without the diet,
−0.4691 with it) despite differing input front ends. Training then moves them
in opposite directions: to **−0.117** without the diet and **−0.861** with it
(both p=.0078, paired Wilcoxon, 8/8 subjects). Trained cells average two
independent models — for the diet arm, the thesis Lightning checkpoint and our
independently trained `dkl_bcs_seed0`, which agree to 0.0025.

**The untrained points are a front-end validation, not an architectural
baseline.** A Gabor oriented-energy model computed on each stream's own
transformed input reproduces them, so an untrained network's stream
preference reflects the input filter and not anything the architecture
contributes. Their role here is to establish that neither front end by itself
imposes a cortical gradient — which is what licenses reading the post-training
divergence as an effect of learning.

**(B)** The gradient along the ventral hierarchy, V1v→hV4. Each subject's
profile has its own across-ROI mean removed, so the curves are compared on
*shape*, not level; this is necessary because the two trained models sit on
opposite sides of the global stream-bias axis (no-diet is magno-dominant in 7/8
subjects, diet-trained parvo-dominant in 0/8). Shaded bands are ±1 SEM, n=8
subjects per curve.

The green dashed curve is the channel-swap control — the *same* transform moved
onto red-green — and it is **flat**. Only the diet on luminance produces the
V1→hV4 descent (hV4−V1v **−0.862**, ρ **−0.964**); the swap gives **−0.027**
and ρ **−0.259**, and the two differ in 8/8 subjects (p=.0078, paired
Wilcoxon). **An important caveat belongs with this curve.** The swap does not
merely fail to produce a gradient — it leaves the parvo stream contributing
almost no unique variance at all (median Uₚ **0.006**, against **0.121** for
the diet arm). Because the plotted statistic is a ratio, a condition in which
one stream is ≈0 everywhere has little dynamic range in which a gradient could
appear. The honest reading is therefore not "the swap abolishes the gradient"
but that *only the diet on luminance yields two streams that both carry
substantial unique cortical variance and a hierarchical gradient between them*.
A fourth front end tested in the same runs but not plotted here
(`parvo_highsf`, where parvo receives raw luminance) behaves differently again:
it kills the **magno** stream (Uₘ 0.007) yet retains a real monotonic ordering,
ρ **−0.780**. See §48.

**(C)** The same divergence measured in the networks' own representations.
Linear-probe accuracy for shape label minus texture label, on Geirhos
cue-conflict images where the two are orthogonal by construction (chance
.0625); both streams reduced to matched PCA dimensionality. Three arms: no
diet, the diet on luminance, and the channel-swap control in which the
*identical* transform is applied to red-green instead. Small open markers are
individual models/seeds (n=3 per trained arm, n=4 for the diet arm).

Magno keeps its shape bias only when the diet is on luminance (**+0.82**);
without the diet training destroys it and drives it below zero (**−0.23**).
The swap arm lands on **−0.23** as well — indistinguishable from no diet at
all, which is why the two magno lines are drawn overlapping. Taking the
transform off luminance returns that channel exactly to its undieted state.

Parvo is texture-biased at **−0.50 to −0.52** whether or not luminance is
dieted, so its texture specialisation is produced by learning and is
untouched by what happens to the *other* stream. Applying the diet to parvo's
own channel relieves this only partially, to **−0.15**, and does not cross
zero. Neither manipulation alone produces a shape-biased stream: it takes the
developmental transform *and* an achromatic carrier, which is the conjunction
the swap and high-SF-parvo controls were built to test.

Note that panel C's arms, unlike panel A's, do *not* start together: the
low-pass front end raises magno's shape bias at initialisation (+0.80 vs
+0.42). The input filter therefore changes what magno represents from the
outset, but does **not** change the initial cortical gradient (panel A) — the
diet's cortical effect is entirely a consequence of what it protects during
learning.

**(D)** Control for the general-factor confound that invalidated the original
thesis's RSA. Spearman ρ between hierarchical position (V1→V2→V3→hV4) and
stream preference, computed per subject. Because ρ is rank-based it is
unchanged by any additive global stream bias, so a global factor cannot produce
these values. The diet-trained networks are near-perfectly monotonic
(**ρ = −0.97**, negative in 8/8 subjects for both models); without the diet the
ordering degrades to −0.554. Points are individual subjects across both models
per condition; bars are means.

---

*Numbers regenerate from `tmp_analysis/fig1_data.py` → `fig1.py`, which prints
the panel-A statistics on every run so the caption cannot drift from the data.*

*Statistics: subject is the unit of inference throughout (n=8). Paired Wilcoxon
signed-rank, two-sided; p=.0078 is the floor for n=8.*

*Correction history: an earlier version of this figure reported the no-diet
trained value as −0.149 and panel D's no-diet ρ as −0.56. Those came from a
`dkl_base_seed0_trained` subject-1 file encoded through the wrong front end
(the diet transform applied to a no-diet model). That subject was re-encoded
and the figure regenerated on 2026-09-01; the corrected values are −0.117 and
−0.554. The correction widens the diet/no-diet separation, so it strengthens
rather than qualifies the figure's claim.*
