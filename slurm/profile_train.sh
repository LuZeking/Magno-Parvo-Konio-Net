#!/bin/bash
# Split the training step into dataloader / DVD front end / model, so the
# 57 img/s measured by the timing probe gets attributed before we try to
# fix it. Runs the same profile at two worker counts: if throughput tracks
# workers, the h5 dataloader is the bottleneck.
#
#SBATCH --job-name=mpk_prof
# NOT the `gpu` partition: those are GTX 1080 Ti (sm_61), unsupported by
# torch 2.13+cu130.
#SBATCH --partition=agcichy
#SBATCH --qos=standard
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=slurm/logs/profile_train_%j.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1 PYTHONPATH=.

PY=/scratch/byrong/anaconda3/envs/mpk/bin/python
DKL=/scratch/byrong/VST_data/VST_datasets/miniecoset_256px_dkl_uint8.h5

nvidia-smi --query-gpu=name,memory.total,utilization.gpu --format=csv,noheader
echo "--- other jobs sharing this node ---"
squeue -w "$SLURMD_NODENAME" -o "%.12i %.10u %.14j %.6D" | head

for W in 8 16; do
  echo ""
  echo "================ num_workers=$W ================"
  $PY tmp_analysis/profile_train.py --dataset "$DKL" \
      --condition dkl_bcs --batch-size 256 --num-workers "$W" --steps 30
done
