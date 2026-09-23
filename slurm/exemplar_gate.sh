#!/bin/bash
# PLAN_EXEMPLAR addendum C (post hoc, 2026-09-23), array C: the accuracy gate recomputed on the held-out
# (E1 fold) bases for the 42 zoo networks. Stored features only; one network per element.
#
#   sbatch slurm/exemplar_gate.sh
#SBATCH --job-name=mpk_exgate
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --array=0-41%14
#SBATCH --output=/scratch/byrong/mpk/exemplar/logs/exgate_%A_%a.out

set -euo pipefail
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1

ZOO=(
  tv_resnet50 tv_resnet152 tv_vgg16 tv_vit_b_16 tv_convnext_small tv_convnext_base
  tv_alexnet tv_vgg11_bn tv_vgg16_bn tv_vgg19_bn tv_googlenet tv_inception_v3
  tv_resnet18 tv_resnet34 tv_resnet101 tv_resnext50_32x4d tv_resnext101_32x8d
  tv_wide_resnet50_2 tv_wide_resnet101_2 tv_densenet121 tv_densenet201
  tv_mobilenet_v2 tv_mobilenet_v3_large tv_shufflenet_v2_x1_0 tv_mnasnet1_0
  tv_efficientnet_b0 tv_efficientnet_b3 tv_efficientnet_v2_s
  tv_regnet_y_8gf tv_regnet_y_16gf tv_convnext_tiny tv_convnext_large
  tv_vit_b_32 tv_vit_l_16 tv_swin_t tv_swin_s tv_swin_b tv_maxvit_t
  tv_resnet50_IMAGENET1K_V2 hub_resnet50_sin hub_resnet50_sin_in hub_resnet50_sin_in_ft
)
TAG=${ZOO[$SLURM_ARRAY_TASK_ID]}
PY=/home/byrong/anaconda3/envs/mpk/bin/python
OUT=/scratch/byrong/mpk/exemplar/results/2026-09-23_gate
mkdir -p "$OUT"
cd "$SLURM_SUBMIT_DIR"
echo "[exgate] task=$SLURM_ARRAY_TASK_ID tag=$TAG"
PYTHONPATH="$SLURM_SUBMIT_DIR" $PY -m src.analysis.generality.exemplar_gate --tag "$TAG" \
    --feat-dir /scratch/byrong/mpk/generality/features/2026-09-15_zoo \
    --beh-dir /scratch/byrong/mpk/generality/behaviour/2026-09-15_zoo \
    --exemplar-dir /scratch/byrong/mpk/exemplar/results/2026-09-21_exemplar --out-dir "$OUT"
echo "[exgate] done $TAG"
