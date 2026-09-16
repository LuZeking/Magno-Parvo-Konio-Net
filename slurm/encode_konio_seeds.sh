#!/bin/bash
# W2c (docs/PLAN_NMI9_2026-09-15.md): seeds 1-2 of the koniocellular control, so that
# Supplementary Note S2 rests on three untrained networks instead of one.
#
#   sbatch slurm/encode_konio_seeds.sh
#
# Each seed needs BOTH arms, because the verdict (tmp_analysis/konio_verdict.py) is the
# contrast between the konio arm and the untrained L-M arm at the SAME seed: --seed sets the
# untrained network's initialisation, the PCA and the GroupKFold folds together, so within a
# seed the magno band is bit-identical across the two arms (the gate r2_magno == r2_magno).
#
#   task 0-7    untrained_seed1                 rg (L-M) stream 2          -- baseline
#   task 8-15   untrained_konio_matched_seed1   by (S-(L+M)), RMS-matched  -- primary konio arm
#   task 16-23  untrained_seed2
#   task 24-31  untrained_konio_matched_seed2
#
# No feature cache exists for these seeds; run_encoding extracts the features in the job
# (~10k NSD images through the front end and the untrained network, CPU) and writes the cache
# under $MPK_FEATURE_CACHE. The memory-bounded path (--ridge-batched, alpha-batched banded
# search) measured 26.7-31 GB on the trained controls (probe 26492385, array 26492597).
#SBATCH --job-name=mpk_konio_seeds
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=34G
#SBATCH --time=16:00:00
#SBATCH --array=0-31%8
#SBATCH --output=slurm/logs/konioseeds_%A_%a.out

set -euo pipefail
mkdir -p slurm/logs
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

TASK=$SLURM_ARRAY_TASK_ID
SUBJECT=$(( TASK % 8 + 1 ))
GROUP=$(( TASK / 8 ))
case $GROUP in
  0) SEED=1; MODEL_NAME=untrained_seed1;               EXTRA=() ;;
  1) SEED=1; MODEL_NAME=untrained_konio_matched_seed1; EXTRA=(--chroma-channel by --chroma-rescale match-rg) ;;
  2) SEED=2; MODEL_NAME=untrained_seed2;               EXTRA=() ;;
  3) SEED=2; MODEL_NAME=untrained_konio_matched_seed2; EXTRA=(--chroma-channel by --chroma-rescale match-rg) ;;
  *) echo "bad group $GROUP" >&2; exit 1 ;;
esac

PY=/scratch/byrong/anaconda3/envs/mpk/bin/python
REPO=$SLURM_SUBMIT_DIR
OUT=$REPO/results/encoding
CACHE_DIR=${MPK_FEATURE_CACHE:-/scratch/byrong/VST_data/mpk_feature_cache}
CACHE=$CACHE_DIR/${MODEL_NAME}_subj$(printf %02d "$SUBJECT").npz

cd "$REPO"
echo "[konio_seeds] task=$TASK model=$MODEL_NAME seed=$SEED subj=$SUBJECT on $(hostname)"
# No --checkpoint: untrained on purpose, as in slurm/encode_konio.sh and encode_subjects.sh untrained_seed0.
$PY -m src.analysis.encoding.run_encoding \
    --subject "$SUBJECT" --model-name "$MODEL_NAME" --out-dir "$OUT" \
    --feature-cache "$CACHE" --seed "$SEED" --ridge-batched \
    "${EXTRA[@]}"
echo "[konio_seeds] done $MODEL_NAME subj$SUBJECT"
