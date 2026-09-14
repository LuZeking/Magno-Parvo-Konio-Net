#!/bin/bash
# Generality test, behaviour (PLAN_GENERALITY D1-D2): cue-conflict features of
# 18 frozen networks and the unit factorisation + fixed-head gain sweep.
#
#   sbatch slurm/generality_cc.sh
#
# Each element: one network, 1,200 cue-conflict images through the frozen
# trunk (CPU), then src.analysis.generality.unit_factor. Minutes per element.
#SBATCH --job-name=mpk_gen_cc
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-17
#SBATCH --output=slurm/logs/gen_cc_%A_%a.out

set -euo pipefail
mkdir -p slurm/logs
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export TORCH_HOME=/scratch/byrong/caches/torch

MODELS=(run:rgb_tsn_seed0 run:rgb_tsn_seed1 run:rgb_tsn_seed2
        run:dkl_bcs_single_seed0 run:dkl_bcs_single_seed1 run:dkl_bcs_single_seed2
        run:dkl_bcs_single34_seed0 run:dkl_bcs_single34_seed1 run:dkl_bcs_single34_seed2
        concat:dkl_bcs_seed0 concat:dkl_bcs_seed1 concat:dkl_bcs_seed2
        tv:resnet50 tv:resnet152 tv:vgg16 tv:vit_b_16 tv:convnext_small tv:convnext_base)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

PY=/scratch/byrong/anaconda3/envs/mpk/bin/python
RUNID=${GEN_RUNID:-2026-09-14_gen}
FEAT=/scratch/byrong/mpk/generality/features/$RUNID
BEH=/scratch/byrong/mpk/generality/behaviour/$RUNID
TAG=$(echo "$MODEL" | tr ':' '_')

echo "[gen_cc] task=$SLURM_ARRAY_TASK_ID model=$MODEL tag=$TAG"
cd "$SLURM_SUBMIT_DIR"
$PY -m src.analysis.generality.extract --model "$MODEL" --set cue_conflict --out-dir "$FEAT"
$PY -m src.analysis.generality.unit_factor --features "$FEAT/${TAG}_cue_conflict.npz" --out-dir "$BEH" --n-splits 10
echo "[gen_cc] done $MODEL"
