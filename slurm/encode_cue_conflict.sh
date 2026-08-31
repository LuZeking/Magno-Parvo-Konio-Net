#!/bin/bash
# PAPER_PLAN §7.3b / §21: refit the encoding models and additionally push the
# Geirhos cue-conflict set through each fitted model, producing a per-voxel
# shape-bias index (VSB) alongside the usual variance partition.
#
# One array over (model x subject). The three models need DIFFERENT front-end
# flags, so the flags are selected from the task id rather than passed in --
# this is still one array job, not a loop of sbatch calls.
#
#   task 0-7   dkl_bcs_seed0_trained   blur + CS   (diet)
#   task 8-15  thesis_dvd_trans        blur + CS   (diet, thesis checkpoint)
#   task 16-23 dkl_base_seed0_trained  no blur, no CS (no diet)
#
# Output goes to results/encoding_cc, NOT results/encoding: the existing fits
# stay untouched, and reproducing their r2 in a separate directory doubles as
# a determinism check. The dkl_base_seed0 tasks here also REPAIR the
# contaminated subject 1 of job 26386734 (PAPER_PLAN §22).
#
# CPU partition per project policy: no CUDA is required (himalaya runs the
# primal solver on torch CPU; feature extraction is a few thousand forward
# passes). Memory and time mirror slurm/encode_subjects.sh, whose measured
# peak was 137G; the cue-conflict predictions add only ~226 MB.
#SBATCH --job-name=mpk_enc_cc
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --array=0-23%6
#SBATCH --output=slurm/logs/enccc_%A_%a.out

set -euo pipefail
mkdir -p slurm/logs
export OPENBLAS_NUM_THREADS=16 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16

TASK=$SLURM_ARRAY_TASK_ID
SUBJECT=$(( TASK % 8 + 1 ))
GROUP=$(( TASK / 8 ))

case $GROUP in
  0) MODEL_NAME=dkl_bcs_seed0_trained
     CKPT=/scratch/byrong/VST_data/mpk_runs/dkl_bcs_seed0/checkpoint_last.pt
     FRONTEND=""
     ;;
  1) MODEL_NAME=thesis_dvd_trans
     CKPT=/scratch/byrong/tsn_checkpoints/dkl_blur_cs_dvd_trans/epoch=147-step=152292.ckpt
     FRONTEND=""
     ;;
  2) MODEL_NAME=dkl_base_seed0_trained
     CKPT=/scratch/byrong/VST_data/mpk_runs/dkl_base_seed0/checkpoint_last.pt
     FRONTEND="--no-lum-blur --no-lum-cs"
     ;;
  *) echo "bad group $GROUP" >&2; exit 1 ;;
esac

PY=/scratch/byrong/anaconda3/envs/mpk/bin/python
REPO=$SLURM_SUBMIT_DIR
OUT=$REPO/results/encoding_cc
CACHE_DIR=${MPK_FEATURE_CACHE:-/scratch/byrong/VST_data/mpk_feature_cache}
CACHE=$CACHE_DIR/${MODEL_NAME}_subj$(printf %02d "$SUBJECT").npz

cd "$REPO"
echo "task=$TASK model=$MODEL_NAME subject=$SUBJECT frontend='${FRONTEND:-blur+CS}'"

$PY -m src.analysis.encoding.run_encoding \
    --subject "$SUBJECT" \
    --model-name "$MODEL_NAME" \
    --out-dir "$OUT" \
    --feature-cache "$CACHE" \
    --checkpoint "$CKPT" \
    --cue-conflict \
    $FRONTEND
