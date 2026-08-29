#!/bin/bash
# Stream-level shape/texture representational probe, one array element per
# model. Pure CPU work (1280 images through two ResNet18 streams plus small
# logistic probes), so per project policy this runs on `main` with no --gres.
#
# The model list pairs each trained model with the untrained control that
# shares its front end, because that is the only comparison in which "what
# training added" is separable from "what the input transform did".
#SBATCH --job-name=mpk_shptex
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-6%4
#SBATCH --output=slurm/logs/shptex_%A_%a.out

set -euo pipefail
mkdir -p slurm/logs
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

REPO=$SLURM_SUBMIT_DIR
PY=/scratch/byrong/anaconda3/envs/mpk/bin/python
RUNS=/scratch/byrong/VST_data/mpk_runs
cd "$REPO"

# name | checkpoint (- for untrained) | extra front-end args
CONFIG=(
  "untrained_blur_cs|-|"
  "untrained_noblur|-|--no-lum-blur --no-lum-cs"
  "dkl_base_seed0_trained|$RUNS/dkl_base_seed0/checkpoint_last.pt|--no-lum-blur --no-lum-cs"
  "dkl_base_seed1_trained|$RUNS/dkl_base_seed1/checkpoint_last.pt|--no-lum-blur --no-lum-cs"
  "dkl_base_seed2_trained|$RUNS/dkl_base_seed2/checkpoint_last.pt|--no-lum-blur --no-lum-cs"
  "dkl_bcs_swap_seed0_trained|$RUNS/dkl_bcs_swap_seed0/checkpoint_last.pt|--transform-target parvo"
  "dkl_bcs_swap_seed1_trained|$RUNS/dkl_bcs_swap_seed1/checkpoint_last.pt|--transform-target parvo"
)

IFS='|' read -r NAME CKPT EXTRA <<< "${CONFIG[$SLURM_ARRAY_TASK_ID]}"
CKPT_ARG=""
[[ "$CKPT" != "-" ]] && CKPT_ARG="--checkpoint $CKPT"

echo "[$SLURM_ARRAY_TASK_ID] $NAME  ckpt=$CKPT  extra=$EXTRA"
$PY -m src.analysis.behavior.stream_shape_texture \
    --model-name "$NAME" \
    --out "$REPO/results/shape_texture" \
    $CKPT_ARG $EXTRA
