#!/bin/bash
# TSN developmental-EEG RSA: extract stream RDMs on the EEG stimuli, then
# run the 7-sub-age-bin temporal RSA. CPU-only (48 images, ResNet18).
#
# Usage:
#   sbatch slurm/eeg_rsa_tsn.sh <model-name> [checkpoint-path] [extra extractor args]
# e.g.
#   sbatch slurm/eeg_rsa_tsn.sh tsn_untrained_seed0
#   sbatch slurm/eeg_rsa_tsn.sh tsn_dkl_bcs /path/to/ckpt
#   sbatch slurm/eeg_rsa_tsn.sh tsn_dkl_base /path/to/ckpt --no-lum-blur --no-lum-cs
#
#SBATCH --job-name=mpk_eeg_rsa
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/eeg_rsa_%j.out

set -euo pipefail
mkdir -p slurm/logs
cd "$SLURM_SUBMIT_DIR"

MODEL_NAME=${1:?model name required}
CHECKPOINT=${2:-}
shift $(( $# >= 2 ? 2 : 1 )) || true

PY=/scratch/byrong/anaconda3/envs/mpk/bin/python
CACHE_P=/scratch/byrong/infant/tsn_rdms_orig_input_v1
CACHE_S=/scratch/byrong/infant/tsn_rdms_orig_input_v1_spearman

CKPT_ARG=""
if [[ -n "$CHECKPOINT" ]]; then
    CKPT_ARG="--checkpoint $CHECKPOINT"
fi

$PY -m src.analysis.eeg.tsn_rdm_extractor \
    --model-name "$MODEL_NAME" \
    --out-root "$CACHE_P" \
    --out-root-spearman "$CACHE_S" \
    $CKPT_ARG "$@"

$PY -m src.analysis.eeg.temporal_rsa_tsn \
    --model-rdm-dir "$CACHE_S" \
    --model-name "$MODEL_NAME" \
    --rdm-metric no_zscore \
    --sources occipital,wholebrain \
    --out-dir "results/eeg_rsa/$MODEL_NAME"
