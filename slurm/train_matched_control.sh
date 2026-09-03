#!/bin/bash
# §67 Test B: the ACCURACY-MATCHED no-diet control.
#
#   sbatch slurm/train_matched_control.sh
#
# WHY. The NMI simulation's major comment M1: the diet arm reaches 0.445 final
# accuracy against 0.701, no per-condition hyperparameter search was run, and so
# the dissociations may follow from an accuracy gap rather than from the diet.
# §67.1 then found the gap is partly a checkpoint artefact -- the diet arm peaks
# at 0.592 (seed range 0.008) and decays to 0.445 (range 0.180) -- but a real
# 0.592-vs-0.709 gap remains, and §67 Test C found the PR ratio is ordered with
# accuracy at r=-0.99 within the diet arm. That measure cannot be cleared
# without a no-diet model held at the diet's accuracy.
#
# No SAVED no-diet checkpoint is low enough: the earliest is ep25 at 0.658.
# From history.json the no-diet arm passes 0.556 at epoch 10-11 and 0.592 at
# epoch 13-16, so 20 epochs with per-epoch checkpointing brackets both targets.
#
# Same seed, same data, same schedule as the original runs, so epochs 1-20 here
# reproduce the first 20 epochs of the published dkl_base runs; history.json is
# checked against the original as a correctness test before anything is claimed.
#
# OUT IS A SEPARATE ROOT. The trainer writes to <out-dir>/<condition>_seed<N>
# (standalone_trainer.py:294), so pointing at mpk_runs would overwrite the
# published dkl_base runs. It points at mpk_runs_matched instead. --resume is
# NOT passed: this must be a fresh run, not a continuation.
#
# COST. ~580 s/epoch x 20 = ~3.2 h per seed (measured from the original runs'
# history.json `sec` field). 3 seeds.
#
# GPU is genuinely required here (this is training, not analysis), so this is
# one of the few jobs that belongs off `main`. QOS standard per project policy;
# `prio` caps at 8 GPUs and the queue already carries other work.
#SBATCH --job-name=mpk_matched
#SBATCH --partition=agcichy
#SBATCH --qos=standard
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=22G
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=slurm/logs/matched_%A_%a.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1

PY=/scratch/byrong/anaconda3/envs/mpk/bin/python
DKL=/scratch/byrong/VST_data/VST_datasets/miniecoset_256px_dkl_uint8.h5
OUT=/scratch/byrong/VST_data/mpk_runs_matched
SEED=$SLURM_ARRAY_TASK_ID

mkdir -p "$OUT" slurm/logs
echo "[matched] seed=$SEED out=$OUT on $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader

$PY -m src.training.standalone_trainer \
    --dataset "$DKL" --condition dkl_base --seed "$SEED" --out-dir "$OUT" \
    --epochs 20 --batch-size 256 --num-workers 8 \
    --ckpt-every 1

echo "[matched] seed=$SEED finished with status $?"
