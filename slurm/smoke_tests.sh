#!/bin/bash
# Pipeline smoke tests on a compute node.
#
# The login node has RLIMIT_NPROC=4096 and is frequently saturated, which
# makes BLAS thread spawning fail mid-import (surfacing as a confusing
# KeyboardInterrupt during `import scipy`). Always smoke-test through Slurm.
#
#SBATCH --job-name=mpk_smoke
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/smoke_%j.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

PY=/scratch/byrong/anaconda3/envs/mpk/bin/python
TMP=/scratch/byrong/anaconda3/fake_home/.claude/jobs/dc3b9010/tmp
SRC=/scratch/byrong/VST_data/VST_datasets/texture2shape_miniecoset_256px.h5

echo "########## 1. encoding solver (primal banded ridge) ##########"
$PY "$TMP/smoke_solver.py" 2>&1 | grep -vE "sampling with cv" | tail -4

echo "########## 2. trainer conditions ##########"
for COND in dkl_bcs dkl_bcs_inter dkl_bcs_parvo_highsf dkl_bcs_swap; do
  echo "--- $COND ---"
  $PY -m src.training.standalone_trainer \
      --dataset "$SRC" --condition "$COND" --seed 0 \
      --out-dir "$TMP/smoke_runs" --epochs 1 --max-steps 2 \
      --batch-size 4 --num-workers 0 --val-batches 1 --ckpt-every 1 \
      2>&1 | grep -E "^\[|Error|Traceback|error" | tail -4
done

echo "########## smoke tests finished ##########"
