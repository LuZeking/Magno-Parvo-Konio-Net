
**2026-08-28 (later) — statistic harmonized, figure built, training array launched.**

*Reporting-consistency fix.* Two summary statistics were in use across my own
analyses: `stats.py` summarizes an ROI as median(U_m) − median(U_p) (a
difference of medians), while the eccentricity, robustness and figure code
use median(U_m − U_p) (the median of the paired per-voxel difference). They
are not identical. All headline numbers are now reported with the **paired**
statistic (`tmp_analysis/group_interaction_paired.py`), which is the more
natural summary of a within-voxel contrast and the one the figure draws.

Harmonized values, n=8, NC>0.1:

| ROI | median(U_m−U_p) | p | subjects | dz |
|---|---|---|---|---|
| early | **+0.0556** | .0078 | 8/8 | +3.65 |
| midparietal | **+0.0286** | .0078 | 8/8 | +1.75 |
| lateral | +0.0174 | .023 | 7/8 | +1.29 |
| parietal | +0.0106 | .078 | 6/8 | +0.81 |
| ventral | +0.0065 | .109 | 6/8 | +0.74 |
| midlateral | −0.0033 | .055 | 1/8 | −0.79 |
| midventral | **−0.0162** | .039 | 1/8 | −0.89 |

Pre-declared interaction: **+0.0245, p_perm = .0079, 8/8 subjects, dz = 1.46**
(was +0.0266 under the difference-of-medians statistic; the conclusion does
not move). Simple effects: dorsal **+0.0196, p = .016**; ventral **−0.0049,
p = .31**.

*The double-dissociation caveat gets stronger, not weaker.* Under the paired
statistic the ventral simple effect falls from p = .078 to **p = .31**. The
pre-declared pooling is the reason: `midventral` on its own is
parvo-preferring (−0.0162, 7/8 subjects) but `ventral` leans very slightly
the other way (+0.0065), and averaging them cancels most of the effect. The
pooled contrast was fixed in advance and is reported as primary; the
per-ROI breakdown showing that the parvo effect is specifically
**midventral** is exploratory and must be labelled as such.

*Figure.* `tmp_analysis/fig_streams.py` → `results/figures/fig_streams.{png,pdf}`.
Three panels (eccentricity, V1→hV4, the interaction), all 8 subjects drawn
individually since the subject is the unit of inference. `--compare` overlays
a second model, which is how the blur control will be shown: if the gradient
is a restatement of the low-pass front end, the red lines in panel A flatten.

**Training array 26373625 launched: 5 conditions x 3 seeds = 15 runs.**
Conditions: dkl_bcs, dkl_base, rgb_tsn, dkl_bcs_swap, dkl_bcs_parvo_highsf.
Deferred: dkl_b, dkl_cs, dkl_bcs_mheavy, dkl_bcs_inter, rgb_tsn_blur.

Measured first-epoch times (8 jobs, 4 per node, g010/g011): **583–734 s**,
i.e. 359–452 img/s — better than the 130–200 img/s I projected, so
**~24–31 h per run and ~2.6 days for all 15**. Two checks that the
performance model is right: `dkl_base` (585 s) and `dkl_bcs` (734 s) differ
by 149 s, almost exactly the profiled DVD front-end cost (143 ms × 1028
steps = 147 s); and val top-1 after one epoch is 0.07–0.23 against 1/112 =
0.009 chance, so every condition is learning. `dkl_base` learns fastest
(0.18–0.23) and `dkl_bcs` slowest (0.07–0.14), as expected when blur+CSF
removes information.

*Self-imposed constraint while the array runs:* no further edits to
`standalone_trainer.py`. A job that resumes from `checkpoint_last.pt` after
an edit would run different code than its own earlier epochs. The
RandomEqualize optimization (a ~20% win) waits until the array drains.

**Still the gating item:** the `--no-lum-blur --no-lum-cs` control array
(26373186) has not returned. Until it does, none of the three neural results
above can be written up as biology rather than as a consequence of the magno
stream being low-pass by construction.
