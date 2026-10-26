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
