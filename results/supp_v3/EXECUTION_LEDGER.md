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
