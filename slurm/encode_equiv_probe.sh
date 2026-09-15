#!/bin/bash
# W2a equivalence probe (docs/PLAN_NMI9_2026-09-15.md, W2a; 2026-09-15).
#
# Refit the stored diet seed-0, subject-01 encoding model on the memory-bounded
# path (--ridge-batched, which since 2026-09-15 also batches the banded random
# search over alphas: the (n_alphas x n_test x n_train) intermediate of
# himalaya's group-ridge solver is the 30 GB peak that pushed the original fits
# to 160 GB and killed the 48 GB control probes 26484409/26484845 at 0% of the
# search). Alpha batching solves each alpha independently, so the estimates must
# reproduce results/encoding/dkl_bcs_seed0_trained_subj01.npz; the check script
# applies the plan's acceptance rule (voxelwise r >= 0.99, per-ROI index within
# 0.02) and prints PASS/FAIL. One heterogeneous job, deliberately not an array.
#
#   sbatch slurm/encode_equiv_probe.sh
#   sacct -j <jobid> --format=JobID,Elapsed,MaxRSS     # size the control array from this
#SBATCH --job-name=mpk_equiv
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/equiv_%j.out

set -euo pipefail
mkdir -p slurm/logs
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

PY=/scratch/byrong/anaconda3/envs/mpk/bin/python
REPO=$SLURM_SUBMIT_DIR
OUT=/scratch/byrong/mpk/encoding/results/2026-09-15_equiv
CACHE=/scratch/byrong/VST_data/mpk_feature_cache/dkl_bcs_seed0_trained_subj01.npz
CKPT=/scratch/byrong/VST_data/mpk_runs/dkl_bcs_seed0/checkpoint_last.pt

[[ -f "$CKPT" ]] || { echo "FATAL: checkpoint missing: $CKPT" >&2; exit 1; }
[[ -f "$CACHE" ]] || { echo "FATAL: feature cache missing: $CACHE" >&2; exit 1; }
mkdir -p "$OUT"
cd "$REPO"
echo "[equiv] dkl_bcs_seed0_trained subj01, batched path, out=$OUT"
$PY -m src.analysis.encoding.run_encoding \
    --subject 1 --model-name dkl_bcs_seed0_trained --out-dir "$OUT" \
    --feature-cache "$CACHE" --checkpoint "$CKPT" --ridge-batched
$PY tmp_analysis/equiv_check.py \
    --new "$OUT/dkl_bcs_seed0_trained_subj01.npz" \
    --ref results/encoding/dkl_bcs_seed0_trained_subj01.npz
echo "[equiv] done"
