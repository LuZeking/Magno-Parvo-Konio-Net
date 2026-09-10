#!/bin/bash
# Fit the readout weight `w` to each of the ten Geirhos observers (PLAN §106).
#
# Closes Limitation 14's own admission -- "no w was fitted to human data and
# nothing here estimates the weighting a human readout would use". §2.3 has a
# sweep and §2.6.2 has kappa at 21 weights; neither is a fit, and a fit can come
# back UNATTAINABLE, which is a statement a range cannot make.
#
# Same 9 runs, same checkpoints, same front-end flags as
# `sweep_error_consistency.sh`, so the fine grid and the published coarse grid
# describe the same models. The script asserts its vectorised statistics against
# the validated scalar ones at every point of that coarse grid before reporting
# anything, so the two cannot silently diverge.
#
# CPU only -- `main`, no --gres, per the project Slurm rules. One array, not a
# loop of sbatch calls. 32G is the proven request for this exact pipeline; the
# fine grid adds a (1001 x 1200 x 16) float32 buffer (~77 MB) and a
# (1200 x 1000) bootstrap multiplicity matrix (~5 MB) on top of it.
#SBATCH --job-name=mpk_wfit
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-8%5
#SBATCH --output=slurm/logs/wfit_%A_%a.out

set -euo pipefail
mkdir -p slurm/logs results/fit_w_observers
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1
REPO=$SLURM_SUBMIT_DIR
PY=/scratch/byrong/anaconda3/envs/mpk/bin/python
RUNS=/scratch/byrong/VST_data/mpk_runs
cd "$REPO"

# name | checkpoint | extra front-end flags
# Copied verbatim from slurm/sweep_error_consistency.sh -- the checkpoint
# resolution is NOT re-derived here, because a different best-epoch choice
# would make the fine grid describe different weights than the published sweep.
CONFIG=(
  "dkl_bcs_seed0_final|$RUNS/dkl_bcs_seed0/checkpoint_last.pt|"
  "dkl_bcs_seed1_final|$RUNS/dkl_bcs_seed1/checkpoint_last.pt|"
  "dkl_bcs_seed2_final|$RUNS/dkl_bcs_seed2/checkpoint_last.pt|"
  "dkl_bcs_seed0_best|$RUNS/dkl_bcs_seed0/checkpoint_125.pt|"
  "dkl_bcs_seed1_best|$RUNS/dkl_bcs_seed1/checkpoint_075.pt|"
  "dkl_bcs_seed2_best|$RUNS/dkl_bcs_seed2/checkpoint_050.pt|"
  "dkl_base_seed0_final|$RUNS/dkl_base_seed0/checkpoint_last.pt|--no-lum-blur --no-lum-cs"
  "dkl_base_seed1_final|$RUNS/dkl_base_seed1/checkpoint_last.pt|--no-lum-blur --no-lum-cs"
  "dkl_base_seed2_final|$RUNS/dkl_base_seed2/checkpoint_last.pt|--no-lum-blur --no-lum-cs"
)
IFS='|' read -r NAME CK EXTRA <<< "${CONFIG[$SLURM_ARRAY_TASK_ID]}"
echo "[$SLURM_ARRAY_TASK_ID] $NAME <- $CK  ${EXTRA:-(diet front end)}"

[[ -f "$CK" ]] || { echo "FATAL: checkpoint missing: $CK"; exit 1; }

PYTHONPATH="$REPO:$REPO/tmp_analysis" $PY tmp_analysis/fit_w_observers.py \
    --model-name "$NAME" --checkpoint "$CK" \
    --out-dir "$REPO/results/fit_w_observers" \
    --w-step 0.001 --n-boot 1000 --allow-partial ${EXTRA}
