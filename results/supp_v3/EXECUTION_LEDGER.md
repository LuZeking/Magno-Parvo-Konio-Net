# Execution ledger — Supplementary S1–S10 for the model of record

Plan: `docs/PLAN_SUPP_MODEL_2026-11-03.md` (plan commit a42bfeb). Launched on the user's "launch", 2026-11-03.

## 1. Stage A — submission (2026-11-03)

- Code: pinned snapshot `/home/byrong/projects/TSN/fusion_stage1_code/057f2fc` = worktree HEAD `057f2fc` (generator
  `tmp_analysis/figs_v3/supp_tables_model.py`, tests `tmp_analysis/test_supp_tables_model.py`, job `slurm/supp_tables_model.sh`).
- Tests before submission: 7/7 in the session (synthetic fixtures with the real dtypes; S7 reimplementation reproduces PAPER_v2's
  Supplementary Table S7 on the BA files: floors +0.0375 / +0.0511, slopes 1.210 / 1.281, real contrasts −0.8335 / −0.0785, all six
  variants to three decimals with the published subject counts).
- Local smoke (session, before the S3 format fix): S3, S5, S6, S9, S7 generated; S9 reproduces Table 1's forced estimator
  (0.730 / 0.648 / 0.520; control 0.393 / 0.383 / 0.471); S3 reproduces the 2×2 (+0.316, −0.177, −0.069, −0.016; interaction +0.44 / +0.73);
  S7's floor/displacement block equals `results/figures_v3/cortex_controls.json` to 5 decimals.
- Job: `26764879`, `--partition=main --qos=standard --cpus-per-task=4 --mem=16G --time=02:00:00`, submitted with
  `env -u TMPDIR sbatch --export=ALL,CODE=<snapshot>`; writes `/scratch/byrong/mpk/encoding/results/2026-11-03_supp_v3/`
  (log `/scratch/byrong/mpk/encoding/logs/2026-11-03_supp_v3/supp_26764879.out`); both directories pinned to OST 36
  (8 MB fsync probes: OST 36 0.65 s, 30 0.65 s, 9 0.57 s).
- Inputs: encoding fits `/scratch/byrong/mpk/encoding/results/2026-10-16_fs2` (model, control, swap × 3 seeds; untrained_dvd300_seed0),
  `results/vp_recovery_realnc_v3/` (16), `results/{ood,sweep_error_consistency,fusion_sweep,cue_conflict,shape_texture}/fs2_*`,
  `results/figures_v2/exemplar_streams.json`, `results/figures_v3/{cortex_controls,ext_gradient}.json`, run histories under
  `docs/slides_2026-10-12_ba_vs_fusion/figures_ba/`, final checkpoints (read-only, parameter count) and NSD pRF/atlas files.

### Noted while writing the generator (not acted on)

- PAPER_v2's Supplementary Table S7 row "per-ROI normalised floor … (ρ vs rank +0.54, p = .22)" is hand-entered in
  `tmp_analysis/figs_v2/ed_tables.py`; the no-diet files give 0.719 for the mean per-subject ρ and 0.964 for the ρ of the mean
  profile. The model's row uses the mean per-subject ρ with an exact Wilcoxon p over the eight subjects, and its caption says so;
  the BA row is left as published and flagged here for the user.

## 2. Stage A — completion and gate (2026-11-03)

- Job `26764879` COMPLETED, exit 0:0, elapsed 00:02:35, MaxRSS 2.0 GB (of 16G requested), node c085; log
  `/scratch/byrong/mpk/encoding/logs/2026-11-03_supp_v3/supp_26764879.out`. Tests in the job: 7/7. Ten tables, 165 input files
  tracked with sha256 in `provenance.json`. The three files were copied verbatim into `results/supp_v3/` (`SHA256SUMS`, verified).
- Consistency gate (plan §3.2; `tmp_analysis/figs_v3/supp_gate.py`): **147/147 comparisons agree, 0 mismatches** — S1 ρ and gradients
  per seed equal the committed cortex summary and Extended Data Table 1; S2 equals the summary's in-silico prediction and Results 2.3;
  S3 reproduces the 2×2 and Table 1; S4 and S9 reproduce Table 1 and the forced estimator; S5 the seven OOD differences of Results 2.4;
  S6 the κ percentages of Results 2.4; S7 equals `cortex_controls.json`'s estimator-bias block; S8 equals its per-seed regressions;
  S10's recomputed fs2 arms reproduce `ext_gradient.json` (max |Δ| < 1e-6 over every cell).
- Two things the gate surfaced that are **not** plan items, disclosed rather than acted on:
  1. The stored affine sweep (`results/fusion_sweep/*_final_wsweep.json`) and the stored per-condition evaluation
     (`results/cue_conflict/*.json`) agree exactly at w = 0.5 and within 0.002 at w = 1, but differ by 0.009–0.026 at the parvo-alone
     point of the control and the swap (e.g. control seed 0: 0.388 over 224 decided images against 0.380 over 216), where only ~17% of
     images are decided. Table S4 and Table S9 report the per-condition evaluation (Table 1's); Table S6's "rate at w = 1" is the
     sweep's, as in PAPER_v2. The gate keeps the comparison as informational, bounded at 0.03.
  2. The κ sweep's w = 1 rate (0.869 / 0.845 / 0.837) is not the per-seed magno rate Results 2.4 quotes (0.871 / 0.842 / 0.838, the
     `magno_only` evaluation); the gate compares the κ percentages, which agree.
- Parameter count: 22,512,848 per network (two trunks of 11,170,240, the joint classifier 57,456, two per-stream heads 57,456 each),
  identical across the three arms; PAPER_v2's BA count (22,397,936) excluded the per-stream heads.

## 3. Stage B — manuscript (2026-11-03)

- `tmp_analysis/figs_v3/patch_supp_model.py` (exact-count assertions, all or nothing; dry run then applied): the ten
  "*Model of record.*" blocks (145 table lines, verbatim from `supp_tables_model.md`) inserted above the BA rows, each BA block labelled
  "*BA network (PAPER_v2).*"; header line 3 (pending sentence removed, 2026-11-03 entry added); Supplementary preface rewritten;
  Note S1 given the model's list of claims not made above the BA list; Note S2 labelled as the BA front end's; Note S4 retitled as
  PAPER_v2's; Note S5 names this plan; Table S4's caption carries the parameter breakdown. `docs/PAPER_v3.md` 869 → 1,088 lines; main
  text unchanged (4,992 words), abstract unchanged (146 words); the only ⟦PENDING⟧ left is the 2026-10-26 changelog sentence, resolved
  on 2026-10-28.
- `tmp_analysis/verify_v3_claims.py` gained `check_supp_model()`: the plan's gate (`supp_gate.gate`, 147 comparisons), every record
  line verbatim in the manuscript, the ten blocks and labels, the marker count, Note S1. Full run: **0 failures, 1,739 OK, BA comparison
  block pass** (`/home/byrong/.claude/jobs/ee1e6055/tmp/verify_v3_supp2.log`). A first run failed on every new check because
  `holds()` requires its phrase to be in the manuscript; the block now anchors on the Supplement preface's sentence.
- Not done (plan §6): no training, refit, CTD rerun or new stimuli; `figs_v2/` scripts, `fs2_cortex_controls.py`, `cortex_controls.json`
  and every v1/v2 result file untouched; not pushed.
