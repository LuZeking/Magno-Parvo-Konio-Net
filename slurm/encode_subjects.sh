#!/bin/bash
# Voxelwise encoding: one Slurm array element per NSD subject.
#
# Usage:
#   sbatch slurm/encode_subjects.sh <model-name> [checkpoint-path] [extra args]
# e.g.
#   sbatch slurm/encode_subjects.sh untrained_seed0
#   sbatch slurm/encode_subjects.sh dkl_bcs /path/to/ckpt
#
# GPU is used for feature extraction and himalaya (torch_cuda).
# prio QOS caps gres/gpu=8/user -> throttle to 4 and use standard QOS so
# other latency-critical jobs keep prio headroom.
#
# CPU partition: the fits are primal banded ridge over ~1k features per band
# (cheap), and per project policy anything that does not need CUDA runs on
# `main` with no --gres. This also avoids the GPU queue, where an earlier
# attempt sat pending behind contention. Note gpu nodes g[001-008] only have
# 95250 MB RealMemory, so a >93G request there is unschedulable, not queued.
#SBATCH --job-name=mpk_encode
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --array=0-7%4
#SBATCH --output=slurm/logs/encode_%A_%a.out

set -euo pipefail
mkdir -p slurm/logs
export OPENBLAS_NUM_THREADS=16 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16

MODEL_NAME=${1:?model name required}
CHECKPOINT=${2:-}
shift $(( $# >= 2 ? 2 : 1 )) || true

SUBJECT=$((SLURM_ARRAY_TASK_ID + 1))
PY=/scratch/byrong/anaconda3/envs/mpk/bin/python
# Run from the directory the job was submitted from, so a worktree checkout
# uses its own code and writes its own results (a hard-coded main-checkout
# path made `src.analysis.encoding` unimportable).
REPO=$SLURM_SUBMIT_DIR
OUT=$REPO/results/encoding
CACHE=$REPO/results/encoding/feature_cache/${MODEL_NAME}_subj$(printf %02d "$SUBJECT").npz

cd "$REPO"
CKPT_ARG=""
if [[ -n "$CHECKPOINT" ]]; then
    CKPT_ARG="--checkpoint $CHECKPOINT"
fi

$PY -m src.analysis.encoding.run_encoding \
    --subject "$SUBJECT" \
    --model-name "$MODEL_NAME" \
    --out-dir "$OUT" \
    --feature-cache "$CACHE" \
    $CKPT_ARG "$@"
