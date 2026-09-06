#!/bin/bash
# Error consistency vs the 10 Geirhos observers across the whole readout sweep.
#
# §2.6.2 bounds trial-level agreement for two models. §2.3 shows the readout
# weight spans the entire behavioural range over FROZEN representations, so this
# asks the sharper question: does agreement track the rate as w is swept?
#
# 3 diet seeds x 2 checkpoint policies + 3 no-diet seeds (whose best saved
# checkpoint IS epoch 150) = 9 tasks, ONE array job.
# CPU only -- `main`, no --gres.
#SBATCH --job-name=mpk_ksweep
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-8%5
#SBATCH --output=slurm/logs/ksweep_%A_%a.out

set -euo pipefail
mkdir -p slurm/logs results/sweep_error_consistency
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1
REPO=$SLURM_SUBMIT_DIR
PY=/scratch/byrong/anaconda3/envs/mpk/bin/python
RUNS=/scratch/byrong/VST_data/mpk_runs
cd "$REPO"

# name | checkpoint | extra front-end flags
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

PYTHONPATH="$REPO:$REPO/tmp_analysis" $PY tmp_analysis/sweep_error_consistency.py \
    --model-name "$NAME" --checkpoint "$CK" \
    --out-dir "$REPO/results/sweep_error_consistency" \
    --w-step 0.05 --allow-partial ${EXTRA}
