#!/bin/bash
# PAPER_PLAN §25: input-level null for the §7.3b cue-conflict profile.
# Gabor oriented-energy features on each stream's own transformed input,
# blur+CS front end to match dkl_bcs_seed0, with --cue-conflict so the
# per-voxel shape-bias index is computed for the null exactly as for the
# trained model. No checkpoint and no network: --features gabor.
#
# CPU partition, no GRES: no CUDA is used anywhere in this path.
#SBATCH --job-name=mpk_cc_gab
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --array=0-7%4
#SBATCH --output=slurm/logs/ccgab_%A_%a.out

set -euo pipefail
mkdir -p slurm/logs
export OPENBLAS_NUM_THREADS=16 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16

SUBJECT=$((SLURM_ARRAY_TASK_ID + 1))
MODEL_NAME=gabor_blur_cs
PY=/scratch/byrong/anaconda3/envs/mpk/bin/python
REPO=$SLURM_SUBMIT_DIR
OUT=$REPO/results/encoding_cc
CACHE_DIR=${MPK_FEATURE_CACHE:-/scratch/byrong/VST_data/mpk_feature_cache}
CACHE=$CACHE_DIR/${MODEL_NAME}_subj$(printf %02d "$SUBJECT").npz

cd "$REPO"
echo "model=$MODEL_NAME subject=$SUBJECT features=gabor frontend=blur+CS"

$PY -m src.analysis.encoding.run_encoding \
    --subject "$SUBJECT" \
    --model-name "$MODEL_NAME" \
    --out-dir "$OUT" \
    --feature-cache "$CACHE" \
    --features gabor \
    --cue-conflict
