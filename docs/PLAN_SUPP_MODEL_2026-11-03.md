# PLAN — Supplementary Tables S1–S10 and Notes S1–S5 for the model of record

**Date:** 2026-11-03. **Branch:** `fusion-stage1` (base `de87e7a`). **Status:** plan only; nothing runs until the user says "launch" / "执行".

## 0. What this stage closes

`docs/PAPER_v3.md` carries two ⟦PENDING⟧ markers (header line 3; Supplementary preface line 480): Supplementary Notes S1–S5 and Tables S1–S10 are PAPER_v2's and describe the BA network (`dkl_bcs`), while the paper's model of record is `fs2_dvd300_avg_w050_ls` (seeds 0–2). This stage gives every table its model-of-record rows, computed with the table's own statistic from outputs that already exist, and keeps the BA rows beneath them as the comparison the main text cites. It adds no claim, no test and no number to the main text; it is a descriptive completion of the Supplement plus consistency checks.

Arms, wherever a table has a column for them (names as stored):

| arm | run names | seeds |
|---|---|---|
| model | `fs2_dvd300_avg_w050_ls_seed{0,1,2}` | 3 |
| control (adult input throughout) | `fs2_dvdfix300_avg_w050_ls_seed{0,1,2}` | 3 |
| channel swap (ramp on L−M) | `fs2_dvd300swap_avg_w050_ls_seed{0,1,2}` | 3 |
| untrained, shared front end | `untrained_dvd300_seed0` | 1 (cortical tables only) |

The BA network's own arms (diet / no diet / swap / parvo + high-SF / thesis checkpoint) stay in the BA rows and nowhere else. The gabor floor exists only as CTD refits on cue-conflict features (`2026-10-26_ctd/gabor_dvd300`), not as NSD-image fits, so it enters no table here.

## 1. Inputs (all exist; nothing is trained, refit or re-encoded; no sealed pixels are read)

| table | statistic (unchanged from the BA table) | model-of-record source | status |
|---|---|---|---|
| S1 monotonicity + exact 630-labelling null, per seed, both statistics | Spearman ρ of the 7-ROI profile vs rank [1,1,2,2,3,3,4]; per-voxel index and ratio of medians | `/scratch/byrong/mpk/encoding/results/2026-10-16_fs2/<run>_subj0[1-8].npz` (`unique_magno`, `unique_parvo`, `reliable_mask`, `noise_ceiling`, `roi_prf-visualrois`) via `tmp_analysis/hierarchy_permutation_null.spi_profile(model, stat, src)` | compute (minutes) |
| S2 in-silico cue conflict through the fitted encoders, per seed | median over nc > 0.1 voxels of `cc_magno_vsb`, `cc_parvo_vsb`; magno − parvo; model − control paired per subject, exact Wilcoxon | same npz files (`cc_*` keys are stored in the fs2 fits) | compute (minutes). Caption states this is PAPER_v2's in-silico index on the NSD-image fits, not the Results 2.5 CTD rate, which refits on cue-conflict features |
| S3 carrier × ramp 2×2 on the probe and on behaviour | probe bias (stratified folds), probe bias with exemplars held out, cue-conflict rate, decided / 1,200; mean [seed range] | `results/shape_texture/fs2_*.json`, `results/figures_v2/exemplar_streams.json['two_by_two_v3']` (cells `achromatic_transform` = model magno, `achromatic_none` = control magno, `chromatic_transform` = swap L−M, `chromatic_none` = the cell Results 2.2 uses; the generator reads the cell map from the JSON, never re-assigns it), `results/cue_conflict/fs2_*.json` | compute (seconds); must reproduce Results 2.2's +0.316 / −0.156 and the +0.44 interaction |
| S4 readout ladder | magno alone (w = 1), both (w = 0.5), parvo alone (w = 0): shape bias [range], decided, accuracy at the scored checkpoint, parameter count from the checkpoint's state_dict | `results/fusion_sweep/fs2_*_final_wsweep.json`, run histories (`verify_v3_claims.hist`), `/scratch/byrong/TSN/fusion_stage1/checkpoints/2026-10-12_stage2/<run>/checkpoint_final.pt` (read-only) | compute (seconds). **Single-stream rows are not regenerated**: no single-trunk network was trained on the dvd300 schedule; the table says so and Note S1 lists it among claims not made. Training one is a separate decision, not this stage's |
| S5 OOD relative-accuracy AUC per family | mean [range] over 3 seeds; model − control; "sep." = complete 3 v 3 seed separation (exact floor p = .100); ‡ families declared trained-in keep the BA run's declaration | `results/ood/fs2_*.json` (`families`) via the AUC of `tmp_analysis/figs_v2/ed_tables.auc` | compute (seconds) |
| S6 error consistency κ with the ten observers | κ and % of the human–human ceiling (0.234) at w = 0.5, at w = 1, best over the 21 w; per run | `results/sweep_error_consistency/fs2_*_final_kappa_sweep.json` | compute (seconds) |
| S7 estimator's own bias + residual-misfit variants | floor (normalised, raw), real contrast seed 0, displacement, recovered slope; the six floor-pairing × residual-removal variants | `results/vp_recovery_realnc_v3/vp_recovery_<model|control seed0>_subj0[1-8].json` (keys identical to the BA files: `band_corr`, `cca`, `conditions{A_recovery,B_manufacture}`) + the fs2 npz | compute (minutes). `tmp_analysis/{diet_floor_paired,floor_pairing,form_check,residual_shape,slope_check}.py` hard-code `results/vp_recovery_realnc` and `results/encoding_cc`: they are **imported, not edited**, with the two paths passed in (or, where a path is a module constant, forked into `tmp_analysis/figs_v3/` with the path as the only change, stated in the file docstring). Seeds 1–2 and the swap were not simulated, as in the BA table |
| S8 stream preference ~ log pRF eccentricity + log pRF size | partial slopes (ecc \| size, size \| ecc), mean over subjects, n on the stated side, exact Wilcoxon p; VIF; trained − untrained paired rows (seed 0 vs `untrained_dvd300_seed0`) | `results/figures_v3/cortex_controls.json['per_seed_regression']` (10 networks × 8 subjects × norm/raw: `ecc_partial`, `size_partial`, `n_vox`, `r_logecc_logsize`, `vif`) | from stored record. The univariate "ecc alone" / "size alone" slopes are not stored: computed by the generator from the npz + NSD pRF maps with `fs2_cortex_controls.py`'s voxel selection imported (pRF R² > 5%, finite ecc/size, ≥ 200 voxels); `fs2_cortex_controls.py` and `cortex_controls.json` are not edited |
| S9 forced two-way estimator | restricted (primary) rate, decided, forced rate; mean [range] over 3 seeds; rows magno alone / both / parvo alone for model, control, swap (final checkpoint) | `results/cue_conflict/fs2_*_logits.npz` with `tmp_analysis/figs_v2/forced_choice.py` | compute (seconds); must reproduce Table 1 (0.850 / 0.682 / 0.477; forced 0.730 / 0.648 / 0.520; control 0.393 / 0.383 / 0.471) |
| S10 stream preference beyond hV4 on every atlas | per-voxel index, median over reliable voxels, mean over 8 subjects, seeds averaged per subject; Kastner2015 beyond hV4, `streams`, fLoc | `results/figures_v3/ext_gradient.json` (untrained, control, model, BA, BA no diet) **plus the swap arm**, computed by the generator with `ext_gradient.profile()` on `FS_ENC` swap runs (ARMS_V3 extended inside the generator, `ext_gradient.py` untouched) | from record + minutes. The thesis and parvo + high-SF columns are BA-only and stay in the BA rows |

Row layout in every table: a sub-heading **Model of record** (rows per §1) above a sub-heading **BA network (PAPER_v2)** holding the present rows verbatim. Captions gain one sentence naming the model rows' source.

## 2. Conventions (fixed; copied from the BA tables and `fs2_cortex_controls.py`)

- Voxels: noise ceiling > 0.1, U_m + U_p > 10⁻³; an ROI cell needs ≥ 20 voxels; both the normalised index (U_m − U_p)/(U_m + U_p) and the raw U_m − U_p are reported where the BA table reports both.
- Trained arms are seed-averaged per subject before any subject-level test; tests are exact two-sided Wilcoxon over 8 subjects (floor p = .0078); seed-level separation is the exact 3 v 3 floor (p = .100); the S1 null enumerates all 630 distinct labellings (floor 1/630 = .0016).
- Behavioural rates: the restricted estimator over decided images, and the forced two-way estimator over all 1,200, exactly as Methods 4.3 / Table S9 define them; checkpoints are the final ones for every fs2 run (no epoch selection).
- Precision: three decimals for indices and rates, two for atlas tables, as the BA rows.

## 3. Reading rule (no reinterpretation at execution)

1. The rows are descriptive. No prediction is attached, no new test enters the main text, no sentence of Results, Discussion or Methods changes because of a Supplementary row.
2. **Consistency gate.** Every model-row number that has a counterpart in the main text, Table 1, Extended Data Tables 1–3 or `results/figures_v3/*.json` must equal it at the stated precision. A disagreement is a **FAIL** of this stage: the ledger records it, the Supplement is not patched, and the discrepancy is raised to the user before anything else moves. The main text is not edited to match.
3. Rows that cannot be regenerated (S4 single-stream; S7 seeds 1–2 and swap; the gabor floor; the thesis checkpoint) are stated as not regenerated, with the reason, in the table itself. No substitute is invented.
4. Note S1 (claims deliberately not made) is rewritten for the model from statements the main text already makes; it may not introduce a claim the main text does not make. Draft list: no dorsal/ventral or anatomical M/P mapping; the two-stream split is not shown necessary (no single-trunk network was trained on the dvd300 schedule); no capacity-matched or task-load-matched control, so the cortical division does not separate task load or the carrier from texture bias (Results 2.3, third qualification); predicted cortex is not texture-biased on the cue-conflict images (Results 2.5 reading of record) and no human readout is measured; the model's shape bias is not human-like in rate or in error consistency (Results 2.4); the model does not buy out-of-distribution robustness; the chromatic stream is not claimed to predict colour-selective cortex; the resolution-tracking claim rests on three seeds against three (p = .10 floor) and its predicted-cortex counterpart points the other way (Results 2.5, P-C4); the CTD instrument does not exceed its input floor beyond hV4.

## 4. Edits to `docs/PAPER_v3.md` (Stage B; patch script, exact-count assertions, all-or-nothing)

- Header (line 3): the ⟦PENDING⟧ sentence is replaced by a dated note naming this plan and `results/supp_v3/`.
- Preface (line 480): rewritten: each table gives the model of record's rows first, then PAPER_v2's rows for the BA network as the comparison; Notes S2 and S4 are the BA network's and are labelled so.
- Note S1: the model's list (§3.4) first; the BA list kept under a sub-heading "The BA network's list (PAPER_v2)".
- Note S2: first sentence labels it the BA front end's koniocellular control (unchanged otherwise). Note S3: already model-first (2026-10-24); unchanged. Note S4: retitled "PAPER_v2's passages on the BA network, condensed in its main text". Note S5: adds the v3 ledgers (`docs/PAPER_v3_PLAN_dvd300.md`, `docs/PLAN_CORTEX_DECISION_2026-10-26.md`, this plan, `results/cortex_decision/EXECUTION_LEDGER.md`).
- Tables S1–S10: model rows inserted per §1, from `results/supp_v3/supp_tables_model.md` verbatim.
- Note S6 (the model's condensed passages) is **not touched** (the verifier's partner rule). The Supplement lies outside the 5,000-word main-text budget; the abstract is untouched.
- `tmp_analysis/verify_v3_claims.py`: new `check_supp_model()` that re-derives every number in the model rows from the §1 sources and applies the §3.2 consistency gate; the two `pending()` markers flip to landed. Full run must end with 0 failures before the commit.

## 5. Execution (CPU only; one Slurm job; 0 GPU-h)

**Stage A — generator.** New file `tmp_analysis/figs_v3/supp_tables_model.py` (new directory; `figs_v2/` scripts are imported, not edited). It writes, into the new directory `results/supp_v3/`:
`supp_tables_model.json` (every number, per seed and per subject where the table aggregates), `supp_tables_model.md` (the rows in the paper's table syntax, one block per table), and `provenance.json` (commit, input paths with sizes and mtimes, the arm map of §0). Unit tests (`tmp_analysis/test_supp_tables_model.py`, synthetic fixtures carrying the real dtypes) cover the exact null, the forced estimator, the AUC and the cell map of S3, and run before submission.

Slurm: `slurm/supp_tables_model.sh`, static, `--partition=main --qos=standard --cpus-per-task=4 --mem=16G --time=02:00:00`, submitted as `env -u TMPDIR sbatch --export=ALL,CODE=<pinned code dir>` from a code snapshot whose `COMMIT` file names the commit; EXIT trap removes `$TMPDIR`; nothing under `/tmp`; log under `/scratch/byrong/mpk/encoding/logs/2026-11-03_supp_v3/`, which is created and pinned to a probed healthy OST first (`lfs setstripe -i <ost>`), as for the CTD run. Reads: ~104 fit files (2.9 MB each) + 16 simulation JSONs + NSD pRF maps and atlas labels; expected wall time ≤ 30 min. One job, not an array (N = 1).

**Stage B — paper patch** (in session, after the job's `provenance.json` exists and the §3.2 gate passes): patch script → verifier in the background → one commit, local, message "Supplementary S1–S10: model-of-record rows (results/supp_v3)".

**Ledger.** `results/supp_v3/EXECUTION_LEDGER.md`: submission (job id, commit, OST), completion (wall time, MaxRSS), gate outcome (§3.2) with every compared pair, and any row declared not regenerated.

**Budget.** ~2 h writing the generator and tests; ≤ 30 min job; ~1.5 h patch + verifier + commit. Half a day end to end; no GPU.

## 6. Not done by this stage

No training (single-trunk dvd300 controls; capacity-matched trunk), no NSD refit, no CTD rerun, no new stimuli, no epoch selection, no edit to any v1/v2 result file, frozen manifest, `figs_v2/` script, `fs2_cortex_controls.py` or `cortex_controls.json`; no push. The premise-3 budget and the capacity-matched control remain the user's separate decisions.
