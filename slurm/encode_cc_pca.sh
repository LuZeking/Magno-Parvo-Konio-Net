#!/bin/bash
# PAPER_PLAN §27.1: capacity-matched control for the §25 null.
#
# The Gabor bank has ~15 effective dimensions (participation ratio) against the
# trained network's ~270/160. If the null fails to reproduce the hV4/LO step,
# that could be a capacity limit rather than a front-end fact. This refits the
# TRAINED model with its PCA truncated toward the bank's capacity, giving a
# ladder: 16 dims (matched) and 64 dims (intermediate) against the 1000 already
# run.
#
# One array over (pca-dim x subject):
#   tasks 0-7   --pca-dim 16   -> dkl_bcs_seed0_pca16
#   tasks 8-15  --pca-dim 64   -> dkl_bcs_seed0_pca64
#
# CACHE_NAME is deliberately NOT the output model name: the features are
# identical to dkl_bcs_seed0_trained (PCA happens after caching), so this reuses
# that provenance-stamped cache and skips the ~25 min extraction. The ridge over
# 16-64 features is cheap, so these are far shorter than the parent runs.
#
# CPU partition, no GRES: no CUDA in this path.
#SBATCH --job-name=mpk_cc_pca
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH --array=0-15%6
#SBATCH --output=slurm/logs/ccpca_%A_%a.out

set -euo pipefail
mkdir -p slurm/logs
export OPENBLAS_NUM_THREADS=16 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16

TASK=$SLURM_ARRAY_TASK_ID
SUBJECT=$(( TASK % 8 + 1 ))
case $(( TASK / 8 )) in
  0) PCA=16 ;;
  1) PCA=64 ;;
  *) echo "bad task $TASK" >&2; exit 1 ;;
esac

MODEL_NAME=dkl_bcs_seed0_pca${PCA}
CACHE_NAME=dkl_bcs_seed0_trained
CKPT=/scratch/byrong/VST_data/mpk_runs/dkl_bcs_seed0/checkpoint_last.pt

PY=/scratch/byrong/anaconda3/envs/mpk/bin/python
REPO=$SLURM_SUBMIT_DIR
OUT=$REPO/results/encoding_cc
CACHE_DIR=${MPK_FEATURE_CACHE:-/scratch/byrong/VST_data/mpk_feature_cache}
CACHE=$CACHE_DIR/${CACHE_NAME}_subj$(printf %02d "$SUBJECT").npz

cd "$REPO"
echo "task=$TASK subject=$SUBJECT pca_dim=$PCA out=$MODEL_NAME cache=$CACHE"

$PY -m src.analysis.encoding.run_encoding \
    --subject "$SUBJECT" \
    --model-name "$MODEL_NAME" \
    --out-dir "$OUT" \
    --feature-cache "$CACHE" \
    --checkpoint "$CKPT" \
    --pca-dim "$PCA" \
    --cue-conflict
