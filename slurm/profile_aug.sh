#!/bin/bash
# Split the training step into dataloader / DVD front end / model, so the
# 57 img/s measured by the timing probe gets attributed before we try to
# fix it. Runs the same profile at two worker counts: if throughput tracks
# workers, the h5 dataloader is the bottleneck.
#
#SBATCH --job-name=mpk_aug
# NOT the `gpu` partition: those are GTX 1080 Ti (sm_61), unsupported by
# torch 2.13+cu130.
#SBATCH --partition=agcichy
#SBATCH --qos=standard
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --output=slurm/logs/profile_aug_%j.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1 PYTHONPATH=.

PY=/scratch/byrong/anaconda3/envs/mpk/bin/python
DKL=/scratch/byrong/VST_data/VST_datasets/miniecoset_256px_dkl_uint8.h5

nvidia-smi --query-gpu=name,memory.total,utilization.gpu --format=csv,noheader
echo "--- other jobs sharing this node ---"
squeue -w "$SLURMD_NODENAME" -o "%.12i %.10u %.14j %.6D" | head

PYTHONPATH=. $PY tmp_analysis/profile_aug.py --batch-size 256 --steps 20
