#!/bin/bash
# CPU variant of slurm/finzi_1to1_extract.sh, for when every usable GPU is busy.
# Identical code path (extract.py falls back to CPU); ~1 h per subject on 16 cores
# against ~21 min on one GPU, but subjects run in parallel on `main`.
#
#   sbatch --array=0-3 slurm/finzi_1to1_extract_cpu.sh
#SBATCH --job-name=f1to1_extract
#SBATCH --partition=main
#SBATCH --qos=inefficient
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --exclude=c025
#SBATCH --output=slurm/logs/f1to1_extract_cpu_%A_%a.out

set -euo pipefail
unset SLURM_CPU_BIND SLURM_CPU_BIND_LIST SLURM_CPU_BIND_TYPE SLURM_CPU_BIND_VERBOSE
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl_$SLURM_JOB_ID"

SUBJECT=$(( SLURM_ARRAY_TASK_ID + 1 ))
# Home copy of the mpk env (same pins): the /scratch copy spent up to ~2.5 h
# page-faulting its CUDA .so files at import on 2026-09-25 (task 26530341_1).
PY=/home/byrong/anaconda3/envs/mpk/bin/python
cd "$SLURM_SUBMIT_DIR"
echo "[f1to1_extract_cpu] subject=$SUBJECT host=$(hostname) start=$(date -Is)"
$PY -m src.analysis.finzi_1to1.extract --subject "$SUBJECT" --networks "${NETWORKS:-all}"
echo "[f1to1_extract_cpu] done $(date -Is)"
