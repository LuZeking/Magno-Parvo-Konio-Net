#!/bin/bash
# Cortex-optimal readout weight (Results 2.4 extension).
#
#   sbatch slurm/readout_w_cortex.sh
#
# For every (model, subject) the classifier's own mixture z(w) = w*y1 + (1-w)*y2
# of the two 512-d stream outputs is used as the design of a ridge encoding
# model of all nsdgeneral voxels, for 21 values of w, plus the concatenation
# [y1, y2] as ceiling. See src/analysis/encoding/readout_w_cortex.py.
#
# 6 models (diet x 3 seeds, no diet x 3 seeds; checkpoint_last.pt, the same
# checkpoints as results/encoding_cc) x 8 subjects = 48 tasks. CPU only:
# ResNet-18 forward over ~10k images per subject plus 22 primal ridge fits with
# 512-1024 features. Betas for one subject are ~2 GB float32; himalaya holds a
# float64 copy plus fold copies, so 32G is generous and under the 64G cap.
#SBATCH --job-name=mpk_readout_w
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --array=0-47%12
#SBATCH --output=slurm/logs/readout_w_%A_%a.out

set -euo pipefail
mkdir -p slurm/logs
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

TASK=$SLURM_ARRAY_TASK_ID
SUBJECT=$(( TASK % 8 + 1 ))
GROUP=$(( TASK / 8 ))

case $GROUP in
  0) RUN=dkl_bcs_seed0  ; FRONTEND="" ;;
  1) RUN=dkl_bcs_seed1  ; FRONTEND="" ;;
  2) RUN=dkl_bcs_seed2  ; FRONTEND="" ;;
  3) RUN=dkl_base_seed0 ; FRONTEND="--no-lum-blur --no-lum-cs" ;;
  4) RUN=dkl_base_seed1 ; FRONTEND="--no-lum-blur --no-lum-cs" ;;
  5) RUN=dkl_base_seed2 ; FRONTEND="--no-lum-blur --no-lum-cs" ;;
  *) echo "bad group $GROUP" >&2; exit 1 ;;
esac

MODEL_NAME=${RUN}_trained
CKPT=/scratch/byrong/VST_data/mpk_runs/${RUN}/checkpoint_last.pt
PY=/scratch/byrong/anaconda3/envs/mpk/bin/python
REPO=$SLURM_SUBMIT_DIR
OUT=/scratch/byrong/mpk/readout_w/results/2026-09-13_w21_last
FEAT=/scratch/byrong/mpk/readout_w/features

echo "[readout_w] task=$TASK model=$MODEL_NAME subj=$SUBJECT frontend='${FRONTEND:-blur+CS}'"
[[ -f "$CKPT" ]] || { echo "FATAL: checkpoint missing: $CKPT" >&2; exit 1; }

cd "$REPO"
$PY -m src.analysis.encoding.readout_w_cortex \
    --subject "$SUBJECT" \
    --checkpoint "$CKPT" \
    --model-name "$MODEL_NAME" \
    --out-dir "$OUT" \
    --feature-dir "$FEAT" \
    $FRONTEND

echo "[readout_w] done $MODEL_NAME subj$SUBJECT"
