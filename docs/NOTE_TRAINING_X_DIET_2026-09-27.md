# Training × diet: registration stopped at the attainability check

Written 2026-09-27 on branch `finzi-1to1`. **Status: not registered, no compute launched.
Decision 2026-09-27: option 1 (report §4 post hoc; paragraph in `results/finzi_1to1/IDEA1_SUPPLEMENT.md`).**
Numbers: `python -m src.analysis.finzi_1to1.seed_subject_split` (reads the idea-1 mapping outputs).

## 1. Origin (post hoc)

The idea-1 result (`PLAN_FINZI_1TO1_2026-09-24.md` §6) was diet ΔDV +0.125 (8/8), of which an
untrained net with the diet front end already gives +0.064. After seeing that, the registered
diet − no-diet contrast (+0.160) was split into the simple effects of training. Only
diet − untrained_diet and diet − nodiet were in the plan; the rest is post hoc.

Subject level (seed-averaged, n = 8, rule of plan §6, δ = 0.05):

| contrast | mean | 90 % CI | > 0 | p | label |
|---|---|---|---|---|---|
| T_diet = diet − untrained diet | +0.061 | [−0.001, +0.122] | 7/8 | .195 | UNRESOLVED |
| T_nodiet = no diet − untrained no diet | −0.070 | [−0.136, −0.004] | 1/8 | .078 | UNRESOLVED |
| untrained: diet − no diet front end | +0.030 | [+0.015, +0.044] | 7/8 | .016 | SMALL |
| interaction = T_diet − T_nodiet | +0.131 | [+0.071, +0.191] | 8/8 | .0078 | POSITIVE |

## 2. The variance is in the subjects, not the seeds

| contrast | seed 0 | seed 1 | seed 2 | SD over subjects | SD seed × subject |
|---|---|---|---|---|---|
| T_diet | +0.087 | +0.042 | +0.054 | 0.092 | 0.018 |
| T_nodiet | −0.069 | −0.076 | −0.066 | 0.099 | 0.011 |
| interaction | +0.156 | +0.118 | +0.119 | 0.089 | 0.014 |

Each subject's value repeats across seeds. Subject 8's T_nodiet is +0.106 / +0.111 / +0.133 in
seeds 0 / 1 / 2 (the only positive subject); subject 4's T_diet is −0.106 / −0.162 / −0.178 (the
only negative subject). Seeds of a trained arm and its untrained control are paired by index only.

## 3. Attainability of a new-seed replication

The plan was: train new seeds (3+) of `dkl_bcs` and `dkl_base` and apply the same rule to them
alone. Measured cost from seeds 0–2: about 20 h (no diet) and 27 h (diet) per run on one A5000.

- **Interaction: PASS is certain and says little.** It is the registered diet − no-diet (+0.160)
  minus a small untrained difference, and it is positive whether the diet *creates* the dorsal
  preference or *protects* it from training.
- **T_nodiet and T_diet, the contrasts that separate those readings, cannot pass.** With the exact
  Wilcoxon at n = 8, one opposite-sign subject at magnitude rank 5 gives p = .078; p < .05 needs
  that subject at rank ≤ 3. Subject 8 would have to move about 10 seed × subject SDs; subject 4
  sits at rank 8 (p = .195). New seeds reproduce both exceptions, so both contrasts stay
  UNRESOLVED however many seeds are added.
- **Binding constraint:** NSD's 8 subjects. A rule with one branch that can only pass and one that
  can only fail does not discriminate, so it was not registered.

## 4. What the existing data support

- **As a property of the networks** (unit = training run; 3/3 seeds, seed SD ≤ 0.023 against effects
  of 0.04–0.08): training with the diet front end moves the magno stream toward dorsal-like voxels
  (T_diet +0.042 to +0.087); training without it moves the stream away (T_nodiet −0.066 to −0.076).
  The front ends barely differ before training (+0.030). The front end sets the direction in which
  learning moves the magno stream. Both "protect" and "add" happen; neither alone describes it.
- **As a statement about human cortex** (unit = subject): each simple effect holds in 7/8 subjects
  with one stable exception (subject 8, subject 4). Post hoc; no registered label.

## 5. Options

1. **Report §4 as is, marked post hoc.** No compute. Recommended.
2. **Trajectory over existing checkpoints** (epochs 25–125 of the 6 trained runs): shows when the
   drift happens. CPU only; at the pilot's measured 12–45 min per network per subject, 30 networks
   × 8 subjects is about 1–2 days on `main` (about half with 2 epochs). Does not lift the subject bound.
3. **More subjects** (a second fMRI dataset with dorsal and ventral stream ROIs): the only way to test
   the subject-level claim. New pipeline, several days.
