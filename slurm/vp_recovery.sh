#!/bin/bash
# Um/Up variance-partitioning recovery analysis (PLAN §82).
#
# Validates the instrument §2.2 rests on: can `variance_partition` recover a
# known unique-variance split, and can it MANUFACTURE a gradient out of
# correlated bands plus ROI-varying SNR? Synthetic voxels, real feature bases,
# production fit_banded/variance_partition.
#
# Pure CPU -- himalaya runs on the numpy backend here, no CUDA in this path --
# so `main`, no --gres. One task per NSD subject: 8 tasks, one array job.
#SBATCH --job-name=mpk_vprec
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=1-8%4
#SBATCH --output=slurm/logs/vprec_%A_%a.out

set -euo pipefail
mkdir -p slurm/logs results/vp_recovery
export OPENBLAS_NUM_THREADS=16 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
export PYTHONUNBUFFERED=1
REPO=$SLURM_SUBMIT_DIR
PY=/scratch/byrong/anaconda3/envs/mpk/bin/python
cd "$REPO"

SUBJ=$SLURM_ARRAY_TASK_ID
echo "[$SLURM_ARRAY_TASK_ID] vp recovery, subject $SUBJ"

PYTHONPATH="$REPO" $PY tmp_analysis/vp_recovery.py \
    --subject "$SUBJ" \
    --model dkl_bcs_seed0_trained \
    --n-comp 1000 \
    --n-vox 200 \
    --backend numpy \
    --out-dir "$REPO/results/vp_recovery"
