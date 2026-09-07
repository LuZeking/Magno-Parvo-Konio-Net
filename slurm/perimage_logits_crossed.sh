#!/bin/bash
# Crossed front-end cells for the per-image analysis.
#
# The matched comparison (slurm/perimage_logits.sh) evaluates each arm with its
# own front end, so the diet and no-diet arms differ BOTH in learned weights and
# in the test-time input transform. Any difference between them is therefore
# unattributable. These six runs complete the 2x2 used in 2.4:
#
#   dkl_bcs_*_nofe   diet weights,    no front end   -> isolates the representation
#   dkl_base_*_fe    no-diet weights, diet front end -> isolates the transform
#
# ONE array job. CPU only: `main`, no --gres.
#SBATCH --job-name=mpk_pixfe
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-5%6
#SBATCH --output=slurm/logs/pixfe_%A_%a.out

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

CONFIG=(
  "dkl_bcs_seed0_final_nofe|$RUNS/dkl_bcs_seed0/checkpoint_last.pt|--no-lum-blur --no-lum-cs"
  "dkl_bcs_seed1_final_nofe|$RUNS/dkl_bcs_seed1/checkpoint_last.pt|--no-lum-blur --no-lum-cs"
  "dkl_bcs_seed2_final_nofe|$RUNS/dkl_bcs_seed2/checkpoint_last.pt|--no-lum-blur --no-lum-cs"
  "dkl_base_seed0_final_fe|$RUNS/dkl_base_seed0/checkpoint_last.pt|"
  "dkl_base_seed1_final_fe|$RUNS/dkl_base_seed1/checkpoint_last.pt|"
  "dkl_base_seed2_final_fe|$RUNS/dkl_base_seed2/checkpoint_last.pt|"
)
IFS='|' read -r NAME CK EXTRA <<< "${CONFIG[$SLURM_ARRAY_TASK_ID]}"
echo "[$SLURM_ARRAY_TASK_ID] $NAME <- $CK  ${EXTRA:-(diet front end)}"

PYTHONPATH="$REPO:$REPO/tmp_analysis" $PY tmp_analysis/perimage_logits.py \
    --model-name "$NAME" --checkpoint "$CK" --out-dir "$OUT" \
    --allow-partial ${EXTRA}
