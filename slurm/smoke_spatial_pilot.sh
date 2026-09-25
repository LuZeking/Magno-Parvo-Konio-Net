#!/bin/bash
# GPU smoke test for the TDANN spatial-loss pilot (PLAN_TSN_SPATIAL_PILOT_2026-09-25).
# Task 0 = dkl_bcs_fused_a0, task 1 = dkl_bcs_fused_a2p5. Each runs 150 steps as
# epoch 0, then resumes into epoch 1 for 150 more (checks the sheet reload and the
# optimizer-state resume), and prints throughput and the per-depth spatial loss.
#
# Runs on g017 (8x B200; the agcichy A5000s are held by a long array of another
# user). g017 has no /scratch: the dataset comes from the /home staging copy
# (slurm/spatial_pilot_stage_data.sh) and is cached in RAM; outputs go to
# /home/byrong/_staging. --mem is sized for the 48 GiB train cache + ~8 GiB process
# and is re-measured here (sacct MaxRSS) before the training runs are sized.
#SBATCH --job-name=sl_smoke
#SBATCH --partition=agcichy
#SBATCH --qos=prio
#SBATCH --nodelist=g017
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=72G
#SBATCH --time=01:00:00
#SBATCH --array=0-1
#SBATCH --output=slurm/logs/sl_smoke_%A_%a.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

PY=/home/byrong/anaconda3/envs/mpk/bin/python
DKL=/home/byrong/_staging/TSN/spatial_pilot_data/miniecoset_256px_dkl_uint8.h5
OUT=/home/byrong/_staging/TSN/spatial_pilot/smoke/$SLURM_ARRAY_JOB_ID
CONDS=(dkl_bcs_fused_a0 dkl_bcs_fused_a2p5)
COND=${CONDS[$SLURM_ARRAY_TASK_ID]}
echo "task $SLURM_ARRAY_TASK_ID -> $COND on $(hostname)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'bf16', torch.cuda.is_bf16_supported())"

COMMON="--dataset $DKL --condition $COND --seed 0 --out-dir $OUT --batch-size 256 \
        --num-workers 0 --cache-ram --val-batches 5 --ckpt-every 100 --resume"
echo "=== run 1: epoch 0, 150 steps ==="
$PY -m src.training.standalone_trainer $COMMON --epochs 1 --max-steps 150
echo "=== run 2: resume, epoch 1, steps 150-300 ==="
$PY -m src.training.standalone_trainer $COMMON --epochs 2 --max-steps 300
echo "=== history ==="
cat "$OUT/${COND}_seed0/history.json"
du -sh "$OUT/${COND}_seed0"
