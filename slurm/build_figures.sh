#!/bin/bash
# Build the three missing paper figures (PAPER_DRAFT §2.6.1, §2.6.2, §2.1/§2.4).
#
# Plotting only -- no CUDA, no training. `main`, no --gres, per the Slurm rules.
# 3 scripts = ONE array job, not three sbatch calls.
#SBATCH --job-name=mpk_figs
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --array=0-2
#SBATCH --output=slurm/logs/figs_%A_%a.out

set -euo pipefail
mkdir -p slurm/logs results/figures
export MPLCONFIGDIR=/scratch/byrong/mpk/.mplcache
mkdir -p "$MPLCONFIGDIR"
export PYTHONUNBUFFERED=1
REPO=$SLURM_SUBMIT_DIR
PY=/scratch/byrong/anaconda3/envs/mpk/bin/python
cd "$REPO"

SCRIPTS=(
  "tmp_analysis/fig4_ood_degradation.py"
  "tmp_analysis/fig5_human_correspondence.py"
  "tmp_analysis/figE1_routes_and_accuracy.py"
)
S="${SCRIPTS[$SLURM_ARRAY_TASK_ID]}"
echo "[$SLURM_ARRAY_TASK_ID] $S"
PYTHONPATH="$REPO:$REPO/tmp_analysis" $PY "$S"
