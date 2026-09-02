#!/bin/bash
# OOD robustness: one array element per trained model (PAPER_PLAN §53).
#
# CPU partition, no --gres. The forward pass would use CUDA if it were there,
# but it does not NEED it: resnet18 inference over 230k image-passes finishes
# inside the walltime on 16 CPU threads, and the GPU queue is currently full of
# the other session's training arrays. Per the project Slurm rules, anything
# that does not need CUDA goes to `main`.
#SBATCH --job-name=mpk_ood
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-11%6
#SBATCH --output=slurm/logs/ood_%A_%a.out

set -euo pipefail
mkdir -p slurm/logs
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16

PY=/scratch/byrong/anaconda3/envs/mpk/bin/python
REPO=$SLURM_SUBMIT_DIR
RUNS=/scratch/byrong/VST_data/mpk_runs
cd "$REPO"

CELL=$(( SLURM_ARRAY_TASK_ID / 3 ))
SEED=$(( SLURM_ARRAY_TASK_ID % 3 ))

case $CELL in
  0) NAME=dkl_bcs_seed${SEED};              EXTRA=() ;;
  1) NAME=dkl_base_seed${SEED};             EXTRA=(--no-lum-blur --no-lum-cs) ;;
  2) NAME=dkl_bcs_swap_seed${SEED};         EXTRA=(--transform-target parvo) ;;
  3) NAME=dkl_bcs_parvo_highsf_seed${SEED}; EXTRA=(--parvo-highsf) ;;
esac

PYTHONPATH=. $PY tmp_analysis/ood_robustness.py \
    --model-name "$NAME" \
    --checkpoint "$RUNS/${NAME}/checkpoint_last.pt" \
    --out-dir "$REPO/results/ood" \
    "${EXTRA[@]}"
