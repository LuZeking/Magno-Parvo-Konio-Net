#!/bin/bash
# Cue-conflict shape bias vs the readout fusion weight w, on existing
# checkpoints. CPU only -- no CUDA in this path -- so `main`, no --gres.
#
# 3 diet seeds x 2 checkpoint policies = 6 tasks, one array job.
# Best-saved epochs for the diet arm are 125 / 75 / 50 (PLAN §71.8, §2.6.2);
# "final" is checkpoint_last.pt, the weights §2.3's published numbers use.
#SBATCH --job-name=mpk_wsweep
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-5
#SBATCH --output=slurm/logs/wsweep_%A_%a.out

set -euo pipefail
mkdir -p slurm/logs results/fusion_sweep
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1
REPO=$SLURM_SUBMIT_DIR
PY=/scratch/byrong/anaconda3/envs/mpk/bin/python
RUNS=/scratch/byrong/VST_data/mpk_runs
cd "$REPO"

CONFIG=(
  "dkl_bcs_seed0_final|$RUNS/dkl_bcs_seed0/checkpoint_last.pt"
  "dkl_bcs_seed1_final|$RUNS/dkl_bcs_seed1/checkpoint_last.pt"
  "dkl_bcs_seed2_final|$RUNS/dkl_bcs_seed2/checkpoint_last.pt"
  "dkl_bcs_seed0_best|$RUNS/dkl_bcs_seed0/checkpoint_125.pt"
  "dkl_bcs_seed1_best|$RUNS/dkl_bcs_seed1/checkpoint_075.pt"
  "dkl_bcs_seed2_best|$RUNS/dkl_bcs_seed2/checkpoint_050.pt"
)
IFS='|' read -r NAME CK <<< "${CONFIG[$SLURM_ARRAY_TASK_ID]}"
echo "[$SLURM_ARRAY_TASK_ID] $NAME <- $CK"

PYTHONPATH="$REPO" $PY tmp_analysis/fusion_sweep.py \
    --model-name "$NAME" --checkpoint "$CK" \
    --out-dir "$REPO/results/fusion_sweep" \
    --w-step 0.05 --allow-partial
