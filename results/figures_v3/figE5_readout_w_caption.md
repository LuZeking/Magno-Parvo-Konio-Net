# Extended Data Figure 5 — caption

**The readout weight cortex prefers descends the hierarchy.** For each mixing weight w the classifier's own input z(w) = w·y₁ + (1 − w)·y₂ is the design of a ridge encoding model of every nsdgeneral voxel (Methods 4.5); a region's w* is the w that maximises its median test r² over reliable voxels (mean over seeds per subject).

**(a)** The model's median r² against w, normalised to each region's best, for V1, hV4 and the ventral and dorsal groups (mean ± s.e.m. over subjects). Every ROI is within 2% of its best r² at w = 0.5: the panel shows an ordering of optima, not an r² difference.

**(b)** w* per region and subject for the model and its control (bar: median). In the model w* falls from 0.65 in V1 to 0.30–0.35 in every extrastriate group, below V1 in 8/8 subjects (Δ = −0.323 [−0.371, −0.269], exact p = .0078); in the control the gradient reverses (0.43 in V1 to 0.62–0.78 beyond hV4; higher in 8/8 subjects, Δ = +0.308 [+0.194, +0.425], p = .0078; three seeds). Bottom row: the weight a trial-by-trial fit to each of the ten observers prefers (median 0.79; not identified by these data, see Results 2.4).

**(c)** The model's shape-bias rate against w (the control's beneath it), with each region's and each observer's w* marked: at the extrastriate weight the model's rate would be 0.60, at V1's 0.75.

Sources: `results/figures_v3/readout_w_cortex.json` and Extended Data Table 2 in `results/figures_v3/ed_tables.md` (`tmp_analysis/figs_v2/readout_w_cortex.py --set v3`, from `/scratch/byrong/mpk/readout_w/results/2026-10-16_fs2_w21`).
