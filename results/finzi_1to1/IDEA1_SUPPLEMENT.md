# Idea-1 supplement: does a spatial wiring cost make the dorsal stream magno-like?

Hand-written; the generated readouts are `SPATIAL_PILOT_REPORT.md` (`spatial_pilot_stats.json`)
and `MAGNO_PROBE_REPORT.md`. Plan: `docs/PLAN_TSN_SPATIAL_PILOT_2026-09-25.md` (outcome row
"A1 DV NOT MAGNO-LIKE -> stop; one paragraph in the idea-1 supplement").

## Spatial-loss pilot (2026-09-26)

We trained a fused M/P ResNet-18 (separate magno and parvo trunks through `layer2`, shared
`layer3`-`layer4`, diet front end) with and without Finzi's spatial loss (A1 alpha = 2.5 per
depth, A0 alpha = 0; one seed, 150 epochs) and ran the idea-1 1-to-1 mapping and the TDANN
magno probe on its `layer4.1` sheet. All gates passed:
- G-train: final val top-1 A0 0.524, A1 0.482 (0.92x).
- G-SL: layer4.1 SL A1 0.194 vs A0 0.487.
- G-probe: trunk median ACH magno 0.98-0.99, parvo 0.00.

The spatial loss built real topography. Units that won the same stream clustered on the sheet
(T2 excess +0.062, z +20.5, vs A0 +0.007, z +2.3; paired p = .0078, 8/8). Magno reliance formed
a map (T1 r +0.142 vs shuffle null +0.052, p = .005; A0 +0.022 vs +0.019, p = .09). The share of
magno-dominated units rose from 0% to 12.6% (MR > 0.8).

The map's dorsal region was nevertheless **not magno-like** (primary: A1 dorsal - ventral, SD
units, n = 8, SESOI 0.2):
- ACH: -0.072 [-0.101, -0.042], 1/8 subjects > 0.
- LSF: -0.165 [-0.187, -0.142], 0/8.
Both are SMALL and point away from magno. This matches TDANN's dorsal winners (ACH -0.146,
LSF -0.107).

Compared with A0, the spatial loss moved the dorsal units further from magno on ACH (-0.147,
0/8) and MR (-0.127, 0/8). It thereby erased the small magno lean A0 showed (ACH +0.075, 8/8;
MR +0.083, 7/8).

Across two unrelated topographic networks, then, the dorsal stream that a wiring cost
produces is slightly anti-magno. The magno -> dorsal routing of idea 1 (+0.125) comes from
keeping the M and P pathways separate behind the diet front end. A spatial constraint on
shared layers does not recreate it and even dilutes it.

**Limitations:**
- One seed.
- The fused networks fit cortex about half as well as TDANN (r/sqrt(NC) ventral / lateral /
  dorsal: A0 0.22 / 0.20 / 0.18, A1 0.19 / 0.18 / 0.15, TDANN alpha = 0.25 0.37 / 0.31 / 0.32).
- Both arms overfit after mid-training (peak val 0.604 / 0.565). The pre-registered final
  checkpoint was used.
- The architecture is not the manuscript TSN.

Per plan section 5 this line stops here; no seed-1 run.

## Post hoc: training sets the direction (2026-09-27)

Not pre-registered. Numbers: `python -m src.analysis.finzi_1to1.seed_subject_split`; reasoning
in `docs/NOTE_TRAINING_X_DIET_2026-09-27.md`.

About half of the idea-1 magno -> dorsal preference is present before training (untrained net,
diet front end, +0.064). Splitting the registered diet - no-diet contrast (+0.160) into the
effects of training shows that training moves the magno stream in opposite directions depending
on the front end:
- With the diet front end, training moves it toward dorsal: +0.087 / +0.042 / +0.054 in seeds
  0 / 1 / 2 (subject level +0.061 [-0.001, +0.122], 7/8).
- Without it, training moves it away: -0.069 / -0.076 / -0.066 (subject level -0.070
  [-0.136, -0.004], 1/8 > 0).
- Before training, the two front ends differ by only +0.030 (7/8).

As a property of the networks this is consistent in 3/3 seeds (seed SD <= 0.023). The front end
sets the direction in which learning moves the magno stream; it both adds to and protects the
input-given lean. As a statement about human cortex it holds in 7/8 subjects for each effect,
with one exception that repeats in every seed (subject 8 without the diet, subject 4 with it).
With 8 NSD subjects, more seeds cannot resolve the subject-level tests (exact Wilcoxon p = .078
and .195), so no confirmatory run was made.
