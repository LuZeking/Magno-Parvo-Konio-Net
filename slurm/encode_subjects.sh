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
#SBATCH --job-name=mpk_encode
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
# gpu nodes g[001-008] have 95250 MB RealMemory; asking for >93G makes the job
# unschedulable (Reason=BadConstraints) rather than merely queued.
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --array=0-7%4
#SBATCH --output=slurm/logs/encode_%A_%a.out

set -euo pipefail
mkdir -p slurm/logs

MODEL_NAME=${1:?model name required}
CHECKPOINT=${2:-}
shift $(( $# >= 2 ? 2 : 1 )) || true

SUBJECT=$((SLURM_ARRAY_TASK_ID + 1))
PY=/scratch/byrong/anaconda3/envs/mpk/bin/python
REPO=/home/byrong/projects/TSN/Magno-Parvo-Konio-Net
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
