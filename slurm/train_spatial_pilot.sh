#!/bin/bash
# TDANN spatial-loss pilot training (PLAN_TSN_SPATIAL_PILOT_2026-09-25 §1).
# Task 0 = A0 dkl_bcs_fused_a0 (alpha 0), task 1 = A1 dkl_bcs_fused_a2p5 (alpha 2.5).
#
# g017 (B200): the agcichy A5000s are held by another user's ~190-task array. g017 has
# no /scratch, so the dataset is read from the verified /home staging copy
# (slurm/spatial_pilot_stage_data.sh) into RAM (--cache-ram), and outputs are staged
# under /home/byrong/_staging; slurm/spatial_pilot_migrate.sh (chained, main) moves
# them to /scratch/byrong/TSN/spatial_pilot/runs and removes both staged copies.
# --mem: smoke test 26532505 (both arms, incl. resume) peaked at MaxRSS = its 72G
# request, i.e. the cgroup also counted the NFS page cache from reading the dataset
# into RAM (48.2 GiB train + 1.0 GiB val cache, ~6 GiB process). 80G adds headroom
# over the reclaimable cache; g017 is exempt from the 64G cap, fair share ~258G/GPU.
# Smoke speed: ~0.43 s/step -> ~440 s/epoch -> ~18.5 h for 150 epochs.
#SBATCH --job-name=sl_train
#SBATCH --partition=agcichy
#SBATCH --qos=standard
#SBATCH --nodelist=g017
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-1
#SBATCH --output=slurm/logs/sl_train_%A_%a.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

PY=/home/byrong/anaconda3/envs/mpk/bin/python
DKL=/home/byrong/_staging/TSN/spatial_pilot_data/miniecoset_256px_dkl_uint8.h5
OUT=/home/byrong/_staging/TSN/spatial_pilot/runs
CONDS=(dkl_bcs_fused_a0 dkl_bcs_fused_a2p5)
COND=${CONDS[$SLURM_ARRAY_TASK_ID]}
echo "task $SLURM_ARRAY_TASK_ID -> $COND on $(hostname)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

$PY -m src.training.standalone_trainer --dataset "$DKL" --condition "$COND" \
    --seed 0 --out-dir "$OUT" --epochs 150 --batch-size 256 --num-workers 0 \
    --cache-ram --ckpt-every 25 --resume
rc=$?
# /home truncates silently when full: fail the task unless the final checkpoint
# exists and loads.
$PY -c "
import torch, sys
ck = torch.load('$OUT/${COND}_seed0/checkpoint_final.pt', map_location='cpu', weights_only=False)
print('final checkpoint ok:', len(ck['state_dict']), 'tensors')" || rc=1
exit $rc
