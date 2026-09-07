#!/bin/bash
# Per-image, per-stream classifier logits on the Geirhos cue-conflict set.
#
# The same 9 configurations as slurm/sweep_error_consistency.sh (3 diet seeds x
# 2 checkpoint policies + 3 no-diet seeds), so the per-image data lines up
# exactly with 2.6.2's kappa sweep. ONE array job.
#
# Storing per-STREAM logits rather than per-w decisions makes the whole readout
# sweep exact offline arithmetic -- see tmp_analysis/perimage_logits.py.
# CPU only: `main`, no --gres.
#SBATCH --job-name=mpk_pilogits
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-8%5
#SBATCH --output=slurm/logs/pilogits_%A_%a.out

set -euo pipefail
mkdir -p slurm/logs
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1
REPO=$SLURM_SUBMIT_DIR
PY=/scratch/byrong/anaconda3/envs/mpk/bin/python
RUNS=/scratch/byrong/VST_data/mpk_runs
OUT=/scratch/byrong/mpk/perimage_behavior/results/2026-09-07_geirhos_perimage/logits
cd "$REPO"
mkdir -p "$OUT"

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

PYTHONPATH="$REPO:$REPO/tmp_analysis" $PY tmp_analysis/perimage_logits.py \
    --model-name "$NAME" --checkpoint "$CK" --out-dir "$OUT" \
    --allow-partial ${EXTRA}
