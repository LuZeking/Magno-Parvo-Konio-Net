# Figure 1 — caption

**The developmental visual diet decides whether learning builds or erodes
cortical correspondence.**

**(A)** Stream preference contrast between hV4 and V1v (magno-unique minus
parvo-unique variance, normalised by their sum), before and after training.
Thin lines are individual NSD subjects (n=8); thick lines the group mean.
Both untrained networks start at the same value (−0.4877 without the diet,
−0.4691 with it) despite differing input front ends. Training then moves them
in opposite directions: to **−0.149** without the diet and **−0.861** with it
(both p=.0078, paired Wilcoxon, 8/8 subjects). Trained cells average two
independent models — for the diet arm, the thesis Lightning checkpoint and our
independently trained `dkl_bcs_seed0`, which agree to 0.0025.

**(B)** The gradient along the ventral hierarchy, V1v→hV4. Each subject's
profile has its own across-ROI mean removed, so the curves are compared on
*shape*, not level; this is necessary because the two trained models sit on
opposite sides of the global stream-bias axis (no-diet is magno-dominant in 7/8
subjects, diet-trained parvo-dominant in 0/8). Shaded bands are ±1 SEM.

**(C)** The same divergence measured in the networks' own representations.
Linear-probe accuracy for shape label minus texture label, on Geirhos
cue-conflict images where the two are orthogonal by construction (chance
.0625); both streams reduced to matched PCA dimensionality. Magno's arms
diverge — the diet preserves its shape bias (+0.82) while training without the
diet destroys it and drives it below zero (−0.23). Parvo's arms coincide,
landing at −0.52 either way: **its texture specialisation is produced by
learning and is unaffected by the diet.** Small open markers are individual
models/seeds.

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
ordering degrades to −0.56. Points are individual subjects across both models
per condition; bars are means.

---

*Numbers regenerate from `tmp_analysis/fig1_data.py` → `fig1.py`, which prints
the panel-A statistics on every run so the caption cannot drift from the data.*

*Statistics: subject is the unit of inference throughout (n=8). Paired Wilcoxon
signed-rank, two-sided; p=.0078 is the floor for n=8.*
