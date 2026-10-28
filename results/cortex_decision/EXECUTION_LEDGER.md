# Cortex-to-decision test — execution ledger

Plan: `docs/PLAN_CORTEX_DECISION_2026-10-26.md`. Paper: PAPER_v3 Results 2.5, Methods 4.8, Fig. 6.

## 1. Registration and code (2026-10-26) — NOT LAUNCHED

Why: the manuscript was refocused on its question (how object recognition can be shape-biased over a
texture-sensitive cortex). The former Fig. 6 strand (frozen single-trunk networks under subspace gains, the
model zoo) was removed from `docs/PAPER_v3.md` (1,123 → 847 lines; main text abstract–Discussion 5,407
words, unchanged in length) and this test was added in its place as the link the paper lacked:
predicted cortex ↔ the observers' decisions on the same images.

What exists before any run:

| item | file |
|---|---|
| registered design (P-C1–P-C5, decision rules, floors, nulls, sensitivity) | `docs/PLAN_CORTEX_DECISION_2026-10-26.md` |
| pipeline flags, all default-off: `--save-predictions`, `--template-design miniecoset16`, `--save-encoder`, `--reproduce-check` | `src/analysis/encoding/run_encoding.py`, `banded_ridge.py` (`band_stats`, `fit_banded(save_encoder=)`) |
| extra designs: 800 MiniEcoset templates + 80 same-cue images | `src/analysis/encoding/template_design.py` |
| analysis (margins, rates, correspondence, exact tests, permutation null, cluster bootstrap, readings) | `src/analysis/encoding/cortex_decision.py` |
| tests (synthetic; saved-encoder path checked against himalaya's own predictions) | `tmp_analysis/test_cortex_decision.py` |
| Slurm: 11 runs × 8 subjects = 88 refits, main/standard, 44 G, 14 h, %16; then the analysis + 3 sensitivity runs | `slurm/ctd_encode.sh`, `slurm/ctd_analysis.sh` |
| Fig. 6 caption skeleton | `results/figures_v3/fig6_cortex_decision_caption.md` |

Inputs verified on disk: stored fits for all 10 trained/untrained runs × 8 subjects in
`/scratch/byrong/mpk/encoding/results/2026-10-16_fs2/` (summaries only — the per-image predictions and the
estimators were discarded, which is why a refit with `--save-predictions --save-encoder` is needed);
feature caches in `/scratch/byrong/mpk/encoding/cache/2026-10-16_fs2/` (so each refit is the ridge only);
1,280 cue-conflict PNGs (1,200 + 80 same-cue) under `/scratch/byrong/encoding/datasets/geirhos_cue_conflict/`;
MiniEcoset `val` 5,600 images (50 per class); observers' per-image profiles and ceilings in
`/scratch/byrong/mpk/perimage_behavior/results/2026-09-07_geirhos_perimage/` (residual ceilings .624 / .728).

Checks before launch (2026-10-26, in-session, mpk env): `tmp_analysis/test_cortex_decision.py` 12/12 passed (synthetic end-to-end reading `PARADOX_REPRODUCED_IN_PREDICTED_CORTEX` as constructed; saved-encoder path reproduces himalaya's predictions to 1e-4; default `fit_banded` call unchanged); `tmp_analysis/verify_v3_claims.py` on the edited manuscript: v3 FAILURES none, 1,471 OK, 44 retired (the former Fig. 6 strand), BA comparison block pass; abstract 149 words; 25 condensed-paragraph pairs consistent. Slurm scripts `bash -n` clean; `--mem` 44G/16G.

Launch (requires the user's explicit "launch"):

```
cd <worktree fusion-stage1>; SHA=$(git rev-parse --short HEAD)
# pinned snapshot: git archive $SHA | tar -x -C /home/byrong/projects/TSN/fusion_stage1_code/$SHA ; echo "$SHA fusion-stage1 (Magno-Parvo-Konio-Net)" > .../COMMIT
mkdir -p /scratch/byrong/mpk/encoding/{logs,results,cache}/2026-10-26_ctd
# pin the result/cache dirs to a healthy OST (lfs setstripe -i <OST>) after probing
E=--export=ALL,CODE=/home/byrong/projects/TSN/fusion_stage1_code/$SHA
ENC=$(env -u TMPDIR sbatch --parsable $E slurm/ctd_encode.sh)
env -u TMPDIR sbatch $E --dependency=afterok:$ENC slurm/ctd_analysis.sh
```

Budget: ≈ 450 CPU-h, ≈ 1.5 days wall at %16; ≈ 40 GB on scratch.

## 2. Launch (2026-10-26 17:52 +01:00, on the user's "launch")

- Code: pinned snapshot `/home/byrong/projects/TSN/fusion_stage1_code/376fb05` = worktree HEAD `376fb05`
  (`diff -rq` of `slurm/` and `src/analysis/encoding/` against the worktree: identical).
- Pre-launch checks: all 80 stored fits and 80 feature caches of `2026-10-16_fs2`, the 9 checkpoints,
  the observers' profiles, the 1,280 cue-conflict PNGs (16 class directories) and the MiniEcoset h5 are on
  disk; `--mem` 44G / 16G; `/tmp` absent; EXIT trap removes `$TMPDIR`; QOS standard MaxJobsPU 45 (array %16 + 1).
- Lustre (42-OST parallel write probe, 16 MB + fsync, 20 s cap, 17:51): OST 3, 12–17 and 23 timed out,
  OST 28 was redirected to OST 4, the rest wrote in 0.8–3.3 s. Pinned with `lfs setstripe -c 1 -i` and
  verified with a probe file: results → OST 36, cache → OST 30, logs → OST 9 (new subdirectories such as
  `results/2026-10-26_ctd/analysis` inherit). Probe files sit in `cache/2026-10-26_ctd/_ostprobe/`; eight
  `dd` writers were still blocked in D on the bad OSTs at submit time — delete that directory once they return.
- Submitted from the snapshot directory with `env -u TMPDIR sbatch --export=ALL,CODE=<snapshot>`:

  | job | script | resources | id |
  |---|---|---|---|
  | encode | `slurm/ctd_encode.sh` | array 0–87 %16, main/standard, 8 CPU, 44 G, 14 h | **26713943** |
  | analysis | `slurm/ctd_analysis.sh`, `--dependency=afterok:26713943` | main/standard, 4 CPU, 16 G, 4 h | **26713944** |

- Outputs: `/scratch/byrong/mpk/encoding/results/2026-10-26_ctd/<run>_subjNN.npz` (88 files),
  `.../results/2026-10-26_ctd/analysis/{ctd_analysis.json,ctd_summary.md}` plus `sens_spearman/`,
  `sens_v1_streams/`, `sens_all_voxels/`; logs `/scratch/byrong/mpk/encoding/logs/2026-10-26_ctd/enc_26713943_<task>.out`
  and `analysis_26713944.out`; the Gabor arm's feature caches in `.../cache/2026-10-26_ctd/`.
- Gate in flight: a trained or untrained task whose refit deviates from its stored fit by more than 1e-3 saves
  its file and exits 3 (`afterok` then holds the analysis); `cortex_decision.load_fit` also drops such files.
  A failed cell is reported as missing, never refilled by hand or rerun with other settings.
- Next, once 26713944 has ended: copy `ctd_analysis.json` and `ctd_summary.md` into `results/cortex_decision/`,
  write §3 here, fill the ⟦PENDING⟧ slots of PAPER_v3 (Results 2.5, abstract, Discussion), write
  `tmp_analysis/figs_v2/fig6_cortex_decision.py`, add the new numbers to `tmp_analysis/verify_v3_claims.py`.

### 2a. In flight, 2026-10-27 15:05 — two time-outs from Lustre stalls; chain re-built

- State at 14:55: 27 of 88 refits COMPLETED, 11 RUNNING, 2 TIMEOUT (tasks 3 = `fs2_dvd300…seed0` subj04 on c116,
  5 = same run subj06 on c042), 48 not started. Every completed refit reproduces its stored 2026-10-16_fs2 fit
  **exactly** (reproduce-check max |Δ| = 0.0 on r2_joint, unique_magno, unique_parvo; 27/27). No Python error in any log.
- Cause of the time-outs: storage, not the fit. Normal elements spend 20–40 min before the ridge (betas 25 min,
  cue-conflict + template features 3 min) and 3.1–7.1 h in the ridge (total 3.6–7.6 h). Six elements stalled in the
  read stages: task 5 took 10 h to read the 1,280 cue-conflict PNGs (its ridge then ran out of time at 65%), task 3
  never left "design + betas" in 14 h and could not be killed ("JOB NOT ENDING WITH SIGNALS", the Lustre-wait
  signature), tasks 7/13/16/21 lost 6–10 h in the same stages (7 and 13 still finished, 44,553–48,792 s). Today's
  42-OST probe (15:01): OST 0, 12, 13, 16, 18–23, 25 stalled; yesterday it was 3, 12–17, 23. The inputs are spread over
  every OST (cue-conflict PNGs ≈ 30 per OST; each subject's 30–40 betas sessions include 3–9 on yesterday's bad set),
  so some elements will always hit a stalled OST; the output directories (OST 36/30/9) were healthy both days.
- Actions (same script, same settings, only the wall limit and node list changed on the sbatch command line;
  no refit output was touched):
  1. `scancel 26713944` — its `afterok:26713943_*` can never be satisfied after a TIMEOUT.
  2. Rerun of tasks 3 and 5: **26719969** (`--array=3,5 --time=30:00:00 --exclude=c116,c042,c162,c012`).
  3. The 48 elements that had not started (`26713943_[40-87]`) were cancelled and resubmitted as
     **26719997** (`--array=40-87%16 --time=30:00:00`, same exclusions), so a stall no longer costs a day.
     The 10–11 running elements of 26713943 (14 h limit) run on; task 21 (subj06, seed2) spent 10 h in the read stage
     and may time out at ≈ 18:20 — if so it is rerun the same way and the gate dependency extended.
  4. New `slurm/ctd_gate.sh` (commit 4e85f06): exits 0 only when all 88 refit files exist. Chain now:
     gate **26719988** `afterany:26713943:26719969:26719997` → analysis **26719989** `afterok:26719988`
     (CODE = snapshot 376fb05 as before). If the gate fails it lists the missing files; rerun those elements and
     resubmit gate + analysis.
- MaxRSS of finished elements 33.5–39.2 GB (includes page cache) against the 44 G request.

### 2b. 2026-10-27 18:30 — third time-out (task 21); rerun 26721875, gate dependency extended

- `26713943_21` (`fs2_dvd300_avg_w050_ls` seed2, subj06, c012) TIMEOUT at 14:00:14 (ended 18:23:58). Log: 1.6 h in
  "design + betas", then 8.5 h reading the 1,280 cue-conflict PNGs + 880 MiniEcoset templates (stage +5,744 s →
  +36,184 s), ridge reached 80 % (12,019 s) at the limit — ≈ 50 min short. No output file was written (DEST absent,
  the script's refuse-overwrite rule is therefore not triggered), no Python error. Same storage signature as tasks 3 and 5.
- Action (script and settings unchanged, only wall limit / node list on the sbatch line): rerun **26721875**
  (`--array=21 --time=30:00:00 --exclude=c116,c042,c162,c012`, CODE = snapshot 376fb05).
  Gate 26719988 dependency extended to `afterany:26713943:26719969:26719997:26721875`
  (`scontrol update`, verified with `scontrol show job`); analysis 26719989 unchanged (`afterok:26719988`).
- State 18:30: 32/88 refit files, all 32 reproduce the stored fits exactly (0.0). Running: 26713943 tasks 25, 28, 37,
  38, 39 (14 h limit; 4–12 h elapsed), 26719969 tasks 3, 5 (c161, 3 h), 26719997 tasks 40–55 (≈ 3 h). Pending:
  26719997_[56–87], 26721875_21. Any further time-out among 25/28/37/38/39 is handled the same way.
- Rerun recipe is now a helper, `slurm/ctd_rerun_element.sh <task>` (commit d8cc66a): refuses while the task is
  live in the gate's chain or its output exists, submits with `--time=30:00:00` and extends the gate's afterany list.
- Queue note (18:40): the standard QOS cap of 45 running jobs is full (23 CTD + 22 `cat_ctl` from another lane).
  Freed slots go by priority: `fs2_nlfeat` ×5 (277065) and `tuned_gen` ×6 (276069, both queued since 10-23) >
  26719997 (270940, %16) > `cat_ctl` (270841, 162 pending, %30) > **26721875_21 (270739)**. So the task-21 rerun
  (and a task-25 rerun, if 25 times out at 20:01 — its ridge was at 45 % with 1.5 h left) starts only when 26719997
  is in its last wave; expected chain end ≈ 2026-10-28 afternoon. Not changed: QOS is the user's policy
  (`scontrol update jobid=26721875 qos=prio` would start it at once; prio has 10/20 running).

### 2c. 2026-10-27 20:10 — fourth time-out (task 25); rerun 26722389

- `26713943_25` (`fs2_dvdfix300_avg_w050_ls` seed0, subj02, c054) TIMEOUT at 14:00:20 (ended 20:01:30): 9.0 h in the
  read stages before the ridge, ridge at 70 % (14,361 s) when the limit hit. No output file, no Python error — the
  same Lustre signature as tasks 3, 5 and 21.
- Action: `slurm/ctd_rerun_element.sh 25` → **26722389** (`--array=25 --time=30:00:00`, same exclusions, CODE 376fb05).
  Gate 26719988 now `afterany:26713943:26719969:26719997:26721875:26722389` (verified); analysis 26719989 unchanged.
- State 20:10: 42/88 refit files, 42/42 reproduce the stored fits exactly (0.0). One element left on the 14 h limit:
  task 28 (ridge 90 %, 1.5 h left, expected to finish). Running 18, pending 26 (26719997, %16 throttle) + 2 (reruns
  21 and 25, behind the standard-QOS cap as noted in §2b).

### 3. 2026-10-28 — chain complete; reading of record `CORTEX_NOT_TEXTURE_BIASED_ON_THESE_IMAGES`

- **Chain.** Reruns 21 (26721875, c149, 4.5 h) and 25 (26722389, c106, 6.6 h) started 21:39/21:40 on 10-27 and
  finished by 04:19; the last 26719997 elements finished 13:37; gate 26719988 ran 13:37:32–:36 ("88/88 refit files
  present"); analysis 26719989 ran 13:39:33–14:18:40 on c055 (39 min, MaxRSS 16.0 GB of the 16 G requested — raise to
  24 G for any rerun). Accounting: `chain_accounting_2026-10-28.txt` in this directory (sacct of all seven jobs).
- **Encode accounting.** 88 elements COMPLETED (3.2–13.6 h each, mean 5.9 h, 520 element-hours = 4,163 CPU-core-hours
  at 8 cores; MaxRSS max 39.2 GB, 26713943_0, of 44 G); 4 TIMEOUTs (tasks 3, 5, 21, 25; Lustre read stalls, §2a–2c),
  each rerun unchanged at 30 h. 1 cancelled range (26713943_[40–87], resubmitted as 26719997, §2a).
- **Reproduction gate.** All 80 refits with a stored 2026-10-16_fs2 reference reproduce it exactly (max |Δ| = 0.0 on
  r2_joint, unique_magno, unique_parvo; `reproduce-check vs` lines of the 80 logs; the 8 `gabor_dvd300` files have no
  reference — new feature set). The analysis excluded 0 files; its three unit tests passed in the job.
- **Readings** (`ctd_summary.md`; identical in the three registered sensitivity runs, Spearman / V1 = streams early /
  all voxels): P-C1 **NOT MET**, P-C2 **NOT MET**, P-C2b **NOT MET**, P-C3 **MET**, P-C4 **NOT MET** → by the plan's
  decision table (PLAN §3) the reading of record is `CORTEX_NOT_TEXTURE_BIASED_ON_THESE_IMAGES`: the paradox is not
  reproduced in predicted cortex; stated as a failure. P-C3 and P-C4 qualify it, they do not change it.
- **Numbers of record** (model, joint band, seeds averaged per subject, mean over 8 subjects; `ctd_analysis.json`):

  | ROI | B joint [min–max] | B magno | B parvo | % ceiling shape, model | input floor (gabor) | n(model > floor) | % ceiling resolution, model |
  |---|---|---|---|---|---|---|---|
  | V1 | 0.603 [0.56–0.65] | 0.660 | 0.506 | +13.9 | +10.2 | 7/8 | +9.5 |
  | V2 | 0.594 [0.56–0.63] | 0.664 | 0.483 | +14.7 | +9.7 | 8/8 | +11.3 |
  | V3 | 0.574 [0.53–0.61] | 0.679 | 0.463 | +15.5 | +10.9 | 8/8 | +10.9 |
  | hV4 | 0.532 [0.51–0.56] | 0.697 | 0.454 | +15.8 | +11.5 | 7/8 | +9.5 |
  | ventral | 0.471 [0.45–0.49] | 0.722 | 0.434 | +9.4 | +13.0 | 0/8 | +8.7 |
  | lateral | 0.496 [0.47–0.52] | 0.728 | 0.442 | +10.3 | +13.8 | 1/8 | +9.2 |
  | parietal | 0.468 [0.45–0.48] | 0.708 | 0.419 | +10.8 | +13.4 | 1/8 | +7.1 |

  - P-C1: B < 0.5 in 8/8 subjects only in ventral (every seed); lateral 5/8; hV4 0/8. B < 0.682 (the behavioural rate)
    in every ROI of every subject: true. The control (B 0.40–0.51) and the swap (0.39–0.51) sit lower than the model in
    every ROI; untrained 0.49–0.55; gabor 0.48–0.52.
  - P-C2: r_h > 0 in 8/8 subjects in hV4, ventral and lateral (exact Wilcoxon p = .0039); above the input floor in
    7/8 (hV4; Wilcoxon vs floor p = .0078), 0/8 (ventral), 1/8 (lateral) — the floor carries as much or more than the
    model outside V1–hV4. Same-cue gate 8/8 in every ROI for model, control and swap (untrained 5–8/8, gabor 6–8/8;
    no NOEST cell). The permutation p on the shape target reported in the json is vacuous — see the defect below.
  - P-C2b (resolution target v): r_v > 0 in 8/8 (p = .0039) in the three ROIs, but the permutation null is not
    beaten (p = .12–.18; 97.5th percentile ≈ 11.8 % of ceiling) and the floor is above the model in ventral (0/8).
  - P-C3: B(magno) > B(parvo) in 8/8 subjects and 24/24 seed-cells in every one of the 7 ROIs, with B(joint) between
    them in 8/8: magno 0.66–0.73, parvo 0.42–0.51, joint 0.47–0.60.
  - P-C4 (extrastriate mean, % ceiling): on v the model seeds (7.0, 8.0, 9.3) lie **below** every control seed
    (9.5, 10.7, 13.1) and every swap seed (10.0, 11.8, 12.0) — exact Mann–Whitney U = 0, p_less = .05, the opposite of
    the predicted direction; on h interleaved (model 8.4, 9.7, 15.2 v control 7.8, 10.7, 11.1 v swap 8.9, 12.0, 12.4;
    p = 1.0). Not met.
  - P-C5 profile (model, % ceiling on h): rises V1 13.9 → hV4 15.8, then 9.4 ventral, 10.3 lateral, 10.8 parietal,
    against median r² 0.19 → 0.09; the floor's profile is the reverse (9.7–11.5 in V1–hV4, 13.0–13.8 beyond).
- **Files.** Copied into this directory: `ctd_analysis.json`, `ctd_summary.md`, `sens_spearman/`, `sens_v1_streams/`,
  `sens_all_voxels/` (json + md each; byte-identical to `/scratch/byrong/mpk/encoding/results/2026-10-26_ctd/analysis/`),
  `chain_accounting_2026-10-28.txt`; `ctd_perimage_hV4.json` (per-image joint-band margins of hV4 for Fig. 6e, model
  and floor, recomputed by `tmp_analysis/ctd_perimage_margins.py` from the pinned code — its per-subject r_h and B
  equal the record's to 1e-6). The 88 refit files (41–51 GB) stay on /scratch.
- **Defect found after the run** (reading of record unaffected): six of the 1,200 cue-conflict images have no resolved
  observer trial, so the observers' shape-rate profile h (and h_resid) is NaN there. The observed r_h masks non-finite
  images (`pearson`), but `pearson_rows` (permutation null) and `cluster_bootstrap_rs` (exemplar CI) did not, so every
  permuted r on the shape target is NaN and the json reports p_h = 1/2001 = .0005 for every arm × ROI, null_h_97.5 =
  NaN and r_h CIs of (nan, nan). The resolution target v has no NaN image: its null, p and CIs are valid. B, r_h, the
  Wilcoxon counts, the floor comparison, P-C1, P-C3, P-C4 and the reading are untouched. Corrected null and CIs for
  r_h: `tmp_analysis/ctd_h_null_fix.py` → `ctd_h_null_fix.json` (§3b); the module is fixed for future runs with a test.

### 3b. 2026-10-28 — corrected shape-target null; Fig. 6; manuscript slots filled

- **Corrected null** (`tmp_analysis/ctd_h_null_fix.py` → `ctd_h_null_fix.json`; 1,194 finite images; same 2,000
  within-cell permutations, seed 0; same exemplar bootstrap seed; every recomputed r_h equals the record's to 4e-17 and
  every recomputed p_v equals the record's exactly, which checks the masked estimator on the finite target). Model,
  shape target, subject-mean r as % of ceiling / permutation p / null 97.5th percentile: V1 13.9 / .0010 / 9.7;
  V2 14.7 / .0005 / 9.4; V3 15.5 / .0005 / 9.9; hV4 15.8 / .0015 / 11.0; ventral 9.4 / .029 / 9.5;
  lateral 10.3 / .013 / 9.3; parietal 10.8 / .0055 / 8.7. Floor (gabor): p = .0095, .0055, .0045, .0005, .0030,
  .0015, .0030. Control and swap p ≤ .005 everywhere; untrained p = .05–.29. So P-C2's permutation criterion
  (p < .01 in each of hV4, ventral, lateral) is **also** not met once corrected (ventral .029, lateral .013); the
  record's P-C2 verdict (not met, on the floor criterion) and the reading of record stand. Exemplar CIs of r_h for
  the model in hV4: lower bounds 0.027–0.058 (all > 0).
- **Module fixed** (`src/analysis/encoding/cortex_decision.py`: `pearson_rows` and `cluster_bootstrap_rs` mask
  non-finite entries as `pearson` does; `tmp_analysis/test_cortex_decision.py::test_nan_images_in_target`; suite
  13/13). The pinned snapshot 376fb05 that produced the record is unchanged; a future full rerun of the analysis
  (CPU, ~40 min, `slurm/ctd_analysis.sh` with `--mem=24G`) would reproduce the record except for the shape-target
  nulls and CIs, which would then equal `ctd_h_null_fix.json`. Not rerun (a new Slurm stage; user's call).
- **Fig. 6** `results/figures_v3/fig6_cortex_decision.{pdf,png}` from `tmp_analysis/figs_v2/fig6_cortex_decision.py`
  (panels a–e; caption `fig6_cortex_decision_caption.md`). Fig. 6e uses `ctd_perimage_hV4.json` (rerun with the
  masked exemplar CI: subject-mean margin r = 0.108 [0.050, 0.155], 17.3 % of ceiling; floor 0.079 [0.017, 0.131]).
- **Manuscript** (`docs/PAPER_v3.md`): header note, abstract (re-cut to 146 words), Introduction (one sentence),
  Results 2.5 (heading, registration note, methods paragraph, four results paragraphs; 430 words in the main text,
  the full paragraphs in Supplementary Note S6 under the journal's 5,000-word rule, the predictions paragraph in
  Methods 4.8), Discussion ("Where the account fails" replaces "Texture-biased cortex that withholds nothing"; the
  "What the readout account licenses" sentence; Limitations) with their Supplementary Note S6 copies, Methods 4.8
  (predictions + defect paragraphs). No ⟦PENDING⟧ marker left for this test; the Supplementary-rows marker of the
  header note remains. `tmp_analysis/verify_v3_claims.py::check_cortex_decision` re-derives every number stated for
  the test from the three json files and the caption; the full verifier passes (1,587 checks, 0 failures, BA block
  pass; main text 4,989 words).
