#!/bin/bash
#SBATCH --job-name=nl_feat
#SBATCH --partition=agcichy
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --array=0-5
#SBATCH --output=/home/byrong/projects/TSN/Magno-Parvo-Konio-Net/.claude/worktrees/mpk-dev/slurm/logs/nlfeat_%A_%a.out
# §104 T1 step 1: pooled per-stream features for the nonlinear readout test.
# 24G is sized to the a5000 fair share (~23G per GPU on a 95G node) so this
# does not strand the node's other three GPUs; the job holds ~0.6 GB of
# features plus batches.
set -euo pipefail
cd /home/byrong/projects/TSN/Magno-Parvo-Konio-Net/.claude/worktrees/mpk-dev
export PYTHONPATH=.:tmp_analysis
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/mpl_$SLURM_JOB_ID"
PY=/scratch/byrong/anaconda3/envs/mpk/bin/python
RUNS=/scratch/byrong/VST_data/mpk_runs
OUT=/scratch/byrong/TSN/nonlinear_readout/features/2026-09-10

# best saved checkpoints are epochs 125/75/50 (§S1.5); final is 150.
SEEDS=(0 1 2 0 1 2)
CKPTS=(checkpoint_125.pt checkpoint_075.pt checkpoint_050.pt \
       checkpoint_150.pt checkpoint_150.pt checkpoint_150.pt)
POLICY=(best best best final final final)
i=$SLURM_ARRAY_TASK_ID
S=${SEEDS[$i]}; C=${CKPTS[$i]}; P=${POLICY[$i]}
EXTRA=""
if [ "${SMOKE:-0}" = "1" ]; then EXTRA="--train-limit 512"; OUT=$OUT/smoke; fi

$PY tmp_analysis/nonlinear_readout_features.py \
    --run-dir "$RUNS/dkl_bcs_seed$S" \
    --checkpoint-name "$C" \
    --tag "dkl_bcs_seed${S}_${P}" \
    --out-dir "$OUT" $EXTRA
