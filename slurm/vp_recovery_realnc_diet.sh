#!/bin/bash
# Re-run of the DIET arm of the real-noise-ceiling recovery analysis.
#
# Job 26440398 ran both arms over the same eight subjects, but
# vp_recovery_realnc.py named its output by subject only, so tasks 8-15
# (dkl_base) overwrote tasks 0-7 (dkl_bcs). The script now puts the model in
# the filename; this recovers the lost diet half. The eight surviving
# dkl_base files were renamed in place, not recomputed.
#
# Pure CPU (himalaya numpy backend) -- `main`, no --gres.
#SBATCH --job-name=mpk_vprec_diet
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-7%6
#SBATCH --output=slurm/logs/vprec_diet_%A_%a.out

set -euo pipefail
mkdir -p slurm/logs results/vp_recovery_realnc
export OPENBLAS_NUM_THREADS=16 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
export PYTHONUNBUFFERED=1
REPO=$SLURM_SUBMIT_DIR
PY=/scratch/byrong/anaconda3/envs/mpk/bin/python
cd "$REPO"

SUBJ=$(( SLURM_ARRAY_TASK_ID + 1 ))
echo "[$SLURM_ARRAY_TASK_ID] model=dkl_bcs_seed0_trained subject=$SUBJ"

PYTHONPATH="$REPO" $PY tmp_analysis/vp_recovery_realnc.py \
    --subject "$SUBJ" \
    --model dkl_bcs_seed0_trained \
    --n-comp 1000 \
    --n-vox 200 \
    --backend numpy \
    --out-dir "$REPO/results/vp_recovery_realnc"
