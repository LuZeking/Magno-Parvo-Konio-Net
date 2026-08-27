#!/bin/bash
# Measure real GPU throughput for one TSN condition before sizing the
# seed x condition array. Runs a bounded number of steps and reports
# images/s, so --time for the full 150-epoch runs comes from evidence
# rather than a guess.
#
#SBATCH --job-name=mpk_train_probe
# The `gpu` partition is GTX 1080 Ti (sm_61), which torch 2.13+cu130 no
# longer ships kernels for: any run there dies with
# 'CUDA error: no kernel image is available for execution on the device'.
# a5000 (sm_86) on agcichy is supported.
#SBATCH --partition=agcichy
#SBATCH --qos=standard
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/train_probe_%j.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1

PY=/scratch/byrong/anaconda3/envs/mpk/bin/python
DKL=/scratch/byrong/VST_data/VST_datasets/miniecoset_256px_dkl_uint8.h5
OUT=$SLURM_SUBMIT_DIR/tmp_analysis/train_timing

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 60 steps at the production batch size; the trainer prints wall seconds per
# epoch, and with --max-steps the "epoch" is exactly those steps.
$PY -m src.training.standalone_trainer \
    --dataset "$DKL" --condition dkl_bcs --seed 0 --out-dir "$OUT" \
    --epochs 1 --max-steps 60 --batch-size 256 --num-workers 8 \
    --val-batches 2 --ckpt-every 1000 2>&1 | grep -aE "^\[|Traceback|Error"

echo "=== steps/epoch at bs=256: $((263200 / 256)) ==="
echo "=== extrapolate: sec_per_epoch = sec_60steps * (1028/60) ==="
sacct -j "$SLURM_JOB_ID" --format=JobID,State,Elapsed,MaxRSS
