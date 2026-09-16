#!/bin/bash
# W2b (docs/PLAN_NMI9_2026-09-15.md): the accuracy-matched no-diet control at seeds 1-2, so that
# Fig. 2e-f rests on three seeds instead of one.
#
#   sbatch slurm/matched_seeds.sh
#
# The retraining already exists: slurm/train_matched_control.sh (job 26423530) trained dkl_base
# seeds 0-2 for 20 epochs with per-epoch checkpoints under /scratch/byrong/VST_data/mpk_runs_matched
# (seeds 1-2 completed 2026-09-07); only seed 0's epoch-10 analysis was run. The selection rule of
# Results 2.2 / Methods 4.7 -- the epoch whose validation top-1 is nearest the diet arm's 0.556 --
# gives epoch 10 for seed 1 (0.546) and epoch 11 for seed 2 (0.554), read from history.json.
# Same flags as the seed-0 analysis (results/*_matched/dkl_base_matched_seed0_ep010.json):
# no-diet front end (--no-lum-blur --no-lum-cs), --allow-partial because a mid-training
# checkpoint is used on purpose. CPU only, minutes each.
#
#   task 0  probe        seed 1 ep 10        task 2  probe        seed 2 ep 11
#   task 1  cue-conflict seed 1 ep 10        task 3  cue-conflict seed 2 ep 11
#SBATCH --job-name=mpk_matched_seeds
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#SBATCH --output=slurm/logs/matched_seeds_%A_%a.out

set -euo pipefail
mkdir -p slurm/logs
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1
PY=/scratch/byrong/anaconda3/envs/mpk/bin/python
RUNS=/scratch/byrong/VST_data/mpk_runs_matched
cd "$SLURM_SUBMIT_DIR"

TASK=$SLURM_ARRAY_TASK_ID
case $TASK in
  0|1) SEED=1; EP=010 ;;
  2|3) SEED=2; EP=011 ;;
  *) echo "bad task $TASK" >&2; exit 1 ;;
esac
NAME=dkl_base_matched_seed${SEED}_ep${EP}
CKPT=$RUNS/dkl_base_seed${SEED}/checkpoint_${EP}.pt
[[ -f "$CKPT" ]] || { echo "FATAL: checkpoint missing: $CKPT" >&2; exit 1; }
echo "[matched_seeds] task=$TASK name=$NAME ckpt=$CKPT"

if (( TASK % 2 == 0 )); then
    PYTHONPATH=. $PY src/analysis/behavior/stream_shape_texture.py \
        --model-name "$NAME" --checkpoint "$CKPT" --no-lum-blur --no-lum-cs --allow-partial \
        --out results/shape_texture_matched
else
    PYTHONPATH=. $PY src/analysis/behavior/cue_conflict.py \
        --model-name "$NAME" --checkpoint "$CKPT" --no-lum-blur --no-lum-cs --allow-partial \
        --out-dir results/cue_conflict_matched
fi
echo "[matched_seeds] done $NAME"
