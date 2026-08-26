#!/bin/bash
# EEG-RSA control battery for the untrained TSN (no checkpoints needed).
#
# Conditions (array index):
#   0,1,2  bcs_seed{0,1,2}   B&CS config, 3 random seeds -> seed robustness
#   3      base_seed0        no blur / no CS            -> is the effect driven
#                                                          by the transforms?
#   4      swap_seed0        blur+CS on the rg channel  -> transformation vs
#                                                          channel identity
# Every run also emits input-level pseudo-layers (magno_input / parvo_input),
# i.e. RDMs of the stimuli themselves, which separate input statistics from
# network features as the source of alignment.
#
#SBATCH --job-name=mpk_eeg_ctrl
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-4
#SBATCH --output=slurm/logs/eeg_ctrl_%A_%a.out

set -euo pipefail
mkdir -p slurm/logs
cd "$SLURM_SUBMIT_DIR"

PY=/scratch/byrong/anaconda3/envs/mpk/bin/python
CACHE_P=/scratch/byrong/infant/tsn_rdms_orig_input_v1
CACHE_S=/scratch/byrong/infant/tsn_rdms_orig_input_v1_spearman

case "$SLURM_ARRAY_TASK_ID" in
  0) NAME=tsn_untrained_bcs_seed0;  ARGS="--seed 0" ;;
  1) NAME=tsn_untrained_bcs_seed1;  ARGS="--seed 1" ;;
  2) NAME=tsn_untrained_bcs_seed2;  ARGS="--seed 2" ;;
  3) NAME=tsn_untrained_base_seed0; ARGS="--seed 0 --no-lum-blur --no-lum-cs" ;;
  4) NAME=tsn_untrained_swap_seed0; ARGS="--seed 0 --transform-target parvo" ;;
  *) echo "bad array index"; exit 1 ;;
esac

echo "=== $NAME ($ARGS) ==="
$PY -m src.analysis.eeg.tsn_rdm_extractor \
    --model-name "$NAME" --out-root "$CACHE_P" \
    --out-root-spearman "$CACHE_S" $ARGS

$PY -m src.analysis.eeg.temporal_rsa_tsn \
    --model-rdm-dir "$CACHE_S" --model-name "$NAME" \
    --rdm-metric no_zscore --sources occipital,wholebrain \
    --out-dir "results/eeg_rsa/$NAME"
