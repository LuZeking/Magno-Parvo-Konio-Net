#!/bin/bash
# Thesis-checkpoint bring-up: the two Osnabrueck checkpoints that landed
# 2026-08-29 (DKL B&CS = dkl_blur_cs_dvd_trans, M-heavy DKL B&CS =
# dkl_blur_cs_s1_90).  Two things are measured, in one array:
#
#   tasks 0-1  stream shape/texture probe (PAPER_PLAN Sec 7.3a / 9.2).  This
#              resolves the Sec 9.3 pre-declaration on a CONVERGED dkl_bcs,
#              which the local retrains (26373625 tasks 0-2) cannot yet supply.
#              Front end is the default blur+CS-on-magno, so the comparison is
#              directly against the existing `untrained_blur_cs` result.
#   tasks 2-3  top-1 accuracy positive control against the thesis's own
#              reported numbers (0.469 / 0.524).  Accuracy is the only measure
#              that goes through the classifier, so it is the one check that is
#              sensitive to stream1_weight.
#
# Heterogeneous by design (two different entry points), dispatched from one
# CONFIG table rather than two submissions, per the array-job policy.
#
# Pure CPU (ResNet18 x2 over 1200 cue-conflict / 5600 val images), so `main`
# with no --gres.
#SBATCH --job-name=mpk_thesisckpt
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-3
#SBATCH --output=slurm/logs/thesisckpt_%A_%a.out

set -euo pipefail
mkdir -p slurm/logs
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1

REPO=$SLURM_SUBMIT_DIR
PY=/scratch/byrong/anaconda3/envs/mpk/bin/python
CKPT=/scratch/byrong/tsn_checkpoints
cd "$REPO"

DVD="$CKPT/dkl_blur_cs_dvd_trans/epoch=147-step=152292.ckpt"
S190="$CKPT/dkl_blur_cs_s1_90/epoch=147-step=152292.ckpt"

# kind | model-name | checkpoint | stream1_weight
CONFIG=(
  "probe|dkl_bcs_thesis_trained|$DVD|0.5"
  "probe|dkl_bcs_mheavy_thesis_trained|$S190|0.9"
  "acc|dkl_bcs_thesis_trained|$DVD|0.5"
  "acc|dkl_bcs_mheavy_thesis_trained|$S190|0.9"
)

IFS='|' read -r KIND NAME CK W <<< "${CONFIG[$SLURM_ARRAY_TASK_ID]}"
echo "[$SLURM_ARRAY_TASK_ID] kind=$KIND name=$NAME w=$W ckpt=$CK on $(hostname)"

if [[ "$KIND" == "probe" ]]; then
  $PY -m src.analysis.behavior.stream_shape_texture \
      --model-name "$NAME" \
      --checkpoint "$CK" \
      --stream1-weight "$W" \
      --out "$REPO/results/shape_texture"
else
  PYTHONPATH="$REPO" $PY tmp_analysis/thesis_ckpt_accuracy.py \
      --model-name "$NAME" \
      --checkpoint "$CK" \
      --condition dkl_bcs \
      --stream1-weight "$W" \
      --out "$REPO/results/thesis_ckpt_accuracy"
fi
