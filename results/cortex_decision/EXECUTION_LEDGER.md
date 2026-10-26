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
