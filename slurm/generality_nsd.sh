#!/bin/bash
# Generality test, cortex step 1 (PLAN_GENERALITY D4): penultimate features of
# every NSD image any subject saw (union over subjects 1-8, ~73k images) for
# the seven networks that go to cortex.
#
#   sbatch slurm/generality_nsd.sh
#SBATCH --job-name=mpk_gen_nsd
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-6
#SBATCH --output=slurm/logs/gen_nsd_%A_%a.out

set -euo pipefail
mkdir -p slurm/logs
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export TORCH_HOME=/scratch/byrong/caches/torch

MODELS=(run:rgb_tsn_seed0 run:rgb_tsn_seed1 run:rgb_tsn_seed2
        run:dkl_bcs_single34_seed0 run:dkl_bcs_single34_seed1 run:dkl_bcs_single34_seed2
        tv:resnet50)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

PY=/scratch/byrong/anaconda3/envs/mpk/bin/python
RUNID=${GEN_RUNID:-2026-09-14_gen}
FEAT=/scratch/byrong/mpk/generality/features/$RUNID

echo "[gen_nsd] task=$SLURM_ARRAY_TASK_ID model=$MODEL"
cd "$SLURM_SUBMIT_DIR"
$PY -m src.analysis.generality.extract --model "$MODEL" --set nsd --out-dir "$FEAT"
echo "[gen_nsd] done $MODEL"
