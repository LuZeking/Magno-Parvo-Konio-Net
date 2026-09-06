#!/bin/bash
# Condition B of the Um/Up recovery analysis, re-run with the REAL per-ROI
# noise-ceiling profile instead of vp_recovery.py's hardcoded ascending ramp.
#
# The original ramp (0.15 -> 0.45) runs the nuisance BACKWARDS: measured mean
# noise ceiling declines V1v -> hV4 (rho = -0.893, n=8), so the manufactured
# -0.351 is the artefact for a nuisance pointing the wrong way. This asks what
# the artefact floor is under the real profile.
#
# Both arms are run because the artefact depends on the feature geometry, and
# the diet and no-diet models differ there (the no-diet model predicts BOLD
# better, §2.7). 2 models x 8 subjects = 16 tasks, ONE array job.
#
# Pure CPU (himalaya numpy backend) -- `main`, no --gres.
#SBATCH --job-name=mpk_vprec_realnc
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-15%6
#SBATCH --output=slurm/logs/vprec_realnc_%A_%a.out

set -euo pipefail
mkdir -p slurm/logs results/vp_recovery_realnc
export OPENBLAS_NUM_THREADS=16 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
export PYTHONUNBUFFERED=1
REPO=$SLURM_SUBMIT_DIR
PY=/scratch/byrong/anaconda3/envs/mpk/bin/python
cd "$REPO"

MODELS=(dkl_bcs_seed0_trained dkl_base_seed0_trained)
MODEL=${MODELS[$(( SLURM_ARRAY_TASK_ID / 8 ))]}
SUBJ=$(( SLURM_ARRAY_TASK_ID % 8 + 1 ))
echo "[$SLURM_ARRAY_TASK_ID] model=$MODEL subject=$SUBJ"

PYTHONPATH="$REPO" $PY tmp_analysis/vp_recovery_realnc.py \
    --subject "$SUBJ" \
    --model "$MODEL" \
    --n-comp 1000 \
    --n-vox 200 \
    --backend numpy \
    --out-dir "$REPO/results/vp_recovery_realnc"
