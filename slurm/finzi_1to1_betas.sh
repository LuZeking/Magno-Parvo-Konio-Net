#!/bin/bash
# Stream-ROI beta cache for the 1-to-1 test (docs/PLAN_FINZI_1TO1_2026-09-24.md §2).
#
#   sbatch slurm/finzi_1to1_betas.sh
#
# CPU only (h5py + numpy). Peak ~ n_trials x n_voxels float32 twice (~3 GB for
# 30,000 trials x 11k voxels); 16G leaves room for the per-session float64 buffer.
#SBATCH --job-name=f1to1_betas
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-7
#SBATCH --exclude=c025
#SBATCH --output=slurm/logs/f1to1_betas_%A_%a.out

set -euo pipefail
unset SLURM_CPU_BIND SLURM_CPU_BIND_LIST SLURM_CPU_BIND_TYPE SLURM_CPU_BIND_VERBOSE
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

SUBJECT=$(( SLURM_ARRAY_TASK_ID + 1 ))
PY=/scratch/byrong/anaconda3/envs/mpk/bin/python
cd "$SLURM_SUBMIT_DIR"
echo "[f1to1_betas] subject=$SUBJECT host=$(hostname) start=$(date -Is)"
$PY -m src.analysis.finzi_1to1.betas --subject "$SUBJECT"
echo "[f1to1_betas] done $(date -Is)"
