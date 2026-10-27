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
