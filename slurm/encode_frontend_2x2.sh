#!/bin/bash
# PAPER_PLAN §45: does the magno/parvo participation-ratio asymmetry follow the
# CARRIER (luminance vs red-green) or the developmental TRANSFORM?
#
# §41.3 measured the asymmetry on two front ends only (diet blur+CS, and the
# no-diet control). This fills the remaining two cells of the same input-factor
# 2x2 that §25 established behaviourally, so the two modalities can be compared
# on identical factors:
#
#   front end       magno stream gets        parvo stream gets
#   blur_cs         lum, transformed         rg,     untransformed   (have)
#   noblur          lum, untransformed       rg,     untransformed   (have)
#   swap            lum, untransformed       rg,     TRANSFORMED     (this job)
#   parvo_highsf    lum, transformed         lum+rg, untransformed   (this job)
#
# Run learning-free (gabor) and untrained-network, exactly the two references
# §41.3 used, so the new cells are commensurable with the published ones.
#
# --cache-only: the PR analysis reads the feature cache and nothing else, so
# betas (~25 min/subject) and the banded ridge (~3.5 h/subject) are skipped.
#
# CPU partition, no GRES: no CUDA is used anywhere in this path.
#SBATCH --job-name=mpk_fe2x2
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=8:00:00
#SBATCH --array=0-31%8
#SBATCH --output=slurm/logs/fe2x2_%A_%a.out

set -euo pipefail
mkdir -p slurm/logs
export OPENBLAS_NUM_THREADS=16 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16

CELL=$((SLURM_ARRAY_TASK_ID / 8))
SUBJECT=$((SLURM_ARRAY_TASK_ID % 8 + 1))

case $CELL in
  0) MODEL_NAME=gabor_swap;             EXTRA=(--features gabor --transform-target parvo) ;;
  1) MODEL_NAME=gabor_parvo_highsf;     EXTRA=(--features gabor --parvo-highsf) ;;
  2) MODEL_NAME=untrained_swap;         EXTRA=(--transform-target parvo) ;;
  3) MODEL_NAME=untrained_parvo_highsf; EXTRA=(--parvo-highsf) ;;
  *) echo "bad cell $CELL" >&2; exit 1 ;;
esac

PY=/scratch/byrong/anaconda3/envs/mpk/bin/python
REPO=$SLURM_SUBMIT_DIR
CACHE_DIR=${MPK_FEATURE_CACHE:-/scratch/byrong/VST_data/mpk_feature_cache}
CACHE=$CACHE_DIR/${MODEL_NAME}_subj$(printf %02d "$SUBJECT").npz

cd "$REPO"
echo "cell=$CELL model=$MODEL_NAME subject=$SUBJECT extra=${EXTRA[*]}"

$PY -m src.analysis.encoding.run_encoding \
    --subject "$SUBJECT" \
    --model-name "$MODEL_NAME" \
    --out-dir "$REPO/results/encoding_fe" \
    --feature-cache "$CACHE" \
    --cache-only \
    --seed 0 \
    "${EXTRA[@]}"
