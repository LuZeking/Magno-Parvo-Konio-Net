#!/bin/bash
# PLAN_SUBSPACE D4 (pre-registered 2026-09-15): penultimate features of every NSD
# image any subject saw for the two ImageNet networks added to the cortex analysis.
#
#   sbatch slurm/subspace_nsd.sh
#
# ViT-B/16 and ConvNeXt-B on CPU, ~73k images at 224 px: a few hours each.
#SBATCH --job-name=mpk_sub_nsd
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-1
#SBATCH --output=slurm/logs/sub_nsd_%A_%a.out

set -euo pipefail
mkdir -p slurm/logs
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export TORCH_HOME=/scratch/byrong/caches/torch

MODELS=(tv:vit_b_16 tv:convnext_base)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
PY=/scratch/byrong/anaconda3/envs/mpk/bin/python
RUNID=${GEN_RUNID:-2026-09-14_gen}
FEAT=/scratch/byrong/mpk/generality/features/$RUNID

echo "[sub_nsd] task=$SLURM_ARRAY_TASK_ID model=$MODEL"
cd "$SLURM_SUBMIT_DIR"
$PY -m src.analysis.generality.extract --model "$MODEL" --set nsd --out-dir "$FEAT"
echo "[sub_nsd] done $MODEL"
