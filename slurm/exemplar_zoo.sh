#!/bin/bash
# PLAN_EXEMPLAR (docs/PLAN_EXEMPLAR_2026-09-21.md, pre-registered 2026-09-21), array A: exemplar-held-out
# spans (E0 reference, E1 fully exemplar-disjoint, E2 texture-, E3 object-exemplar folds) and probes
# (Q0 reference, Q1 balanced, Q1b, Q2 grouped) on stored features -- the 42 zoo networks and our 12
# trunk runs. Nothing is extracted or retrained. One network per element.
#
#   sbatch slurm/exemplar_zoo.sh                    # all 54
#   sbatch --array=12 slurm/exemplar_zoo.sh         # one element (test)
#SBATCH --job-name=mpk_exemplar
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --array=0-53%14
#SBATCH --output=/scratch/byrong/mpk/exemplar/logs/exemplar_%A_%a.out

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
TRUNKS=(
  concat_dkl_bcs_seed0 concat_dkl_bcs_seed1 concat_dkl_bcs_seed2
  run_dkl_bcs_single_seed0 run_dkl_bcs_single_seed1 run_dkl_bcs_single_seed2
  run_dkl_bcs_single34_seed0 run_dkl_bcs_single34_seed1 run_dkl_bcs_single34_seed2
  run_rgb_tsn_seed0 run_rgb_tsn_seed1 run_rgb_tsn_seed2
)
ALL=("${ZOO[@]}" "${TRUNKS[@]}")
TAG=${ALL[$SLURM_ARRAY_TASK_ID]}
if (( SLURM_ARRAY_TASK_ID < ${#ZOO[@]} )); then RUN=2026-09-15_zoo; else RUN=2026-09-14_gen; fi

PY=/home/byrong/anaconda3/envs/mpk/bin/python
FEAT=/scratch/byrong/mpk/generality/features/$RUN
BEH=/scratch/byrong/mpk/generality/behaviour/$RUN
OUT=/scratch/byrong/mpk/exemplar/results/2026-09-21_exemplar
mkdir -p "$OUT"
cd "$SLURM_SUBMIT_DIR"
echo "[exemplar] task=$SLURM_ARRAY_TASK_ID tag=$TAG run=$RUN"
PYTHONPATH="$SLURM_SUBMIT_DIR" $PY -m src.analysis.generality.exemplar \
    --features "$FEAT/${TAG}_cue_conflict.npz" --behaviour-dir "$BEH" --out-dir "$OUT"
echo "[exemplar] done $TAG"
