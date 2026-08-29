#!/bin/bash
# stream1_weight 2x2 cross-check.  Nothing in the Lightning checkpoints records
# the fusion weight -- the s1_90 -> 0.9 mapping rests on the filename, the
# model_zoo registration and the thesis figure legend.  Accuracy is the only
# measure that runs through the classifier, so evaluating each checkpoint under
# BOTH weights tests that mapping directly: each model should peak at the weight
# it was trained with.  Off-diagonal cells only.
#SBATCH --job-name=mpk_ckptw
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-1
#SBATCH --output=slurm/logs/ckptw_%A_%a.out

set -euo pipefail
mkdir -p slurm/logs
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1
REPO=$SLURM_SUBMIT_DIR
PY=/scratch/byrong/anaconda3/envs/mpk/bin/python
CKPT=/scratch/byrong/tsn_checkpoints
cd "$REPO"

CONFIG=(
  "dvd_trans_at_w0.9|$CKPT/dkl_blur_cs_dvd_trans/epoch=147-step=152292.ckpt|0.9"
  "s1_90_at_w0.5|$CKPT/dkl_blur_cs_s1_90/epoch=147-step=152292.ckpt|0.5"
)
IFS='|' read -r NAME CK W <<< "${CONFIG[$SLURM_ARRAY_TASK_ID]}"
echo "[$SLURM_ARRAY_TASK_ID] $NAME w=$W"
PYTHONPATH="$REPO" $PY tmp_analysis/thesis_ckpt_accuracy.py \
    --model-name "$NAME" --checkpoint "$CK" --condition dkl_bcs \
    --stream1-weight "$W" --out "$REPO/results/thesis_ckpt_accuracy"
