#!/bin/bash
# Seeds 1-2 of the two front-end controls on the cortical instrument (NMI referee
# item 5: swap and parvo_highsf cortical fits are n = 1).
#
#   sbatch --array=0 slurm/encode_cc_controls.sh     # memory/time probe first
#   sbatch slurm/encode_cc_controls.sh               # then the rest
#
# Same pipeline as slurm/encode_cue_conflict.sh (results/encoding seed-0 cells),
# feature caches already written by job 26413641, no cue-conflict pass. The
# single-band ridge fits are batched (--ridge-batched) so the job fits under
# the 64G cap: the unbatched 2026-08 jobs peaked at 120-128 GB, almost all of it
# the n_alphas x n_val x n_targets validation predictions held at once.
#SBATCH --job-name=mpk_cc_ctrl
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-31%8
#SBATCH --output=slurm/logs/ccctrl_%A_%a.out

set -euo pipefail
mkdir -p slurm/logs
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

TASK=$SLURM_ARRAY_TASK_ID
SUBJECT=$(( TASK % 8 + 1 ))
GROUP=$(( TASK / 8 ))
case $GROUP in
  0) RUN=dkl_bcs_swap_seed1         ; FRONTEND="--transform-target parvo" ;;
  1) RUN=dkl_bcs_swap_seed2         ; FRONTEND="--transform-target parvo" ;;
  2) RUN=dkl_bcs_parvo_highsf_seed1 ; FRONTEND="--parvo-highsf" ;;
  3) RUN=dkl_bcs_parvo_highsf_seed2 ; FRONTEND="--parvo-highsf" ;;
  *) echo "bad group $GROUP" >&2; exit 1 ;;
esac
MODEL_NAME=${RUN}_trained
CKPT=/scratch/byrong/VST_data/mpk_runs/${RUN}/checkpoint_last.pt
PY=/scratch/byrong/anaconda3/envs/mpk/bin/python
REPO=$SLURM_SUBMIT_DIR
OUT=$REPO/results/encoding
CACHE_DIR=${MPK_FEATURE_CACHE:-/scratch/byrong/VST_data/mpk_feature_cache}
CACHE=$CACHE_DIR/${MODEL_NAME}_subj$(printf %02d "$SUBJECT").npz

echo "[ccctrl] task=$TASK model=$MODEL_NAME subj=$SUBJECT frontend='$FRONTEND'"
[[ -f "$CKPT" ]] || { echo "FATAL: checkpoint missing: $CKPT" >&2; exit 1; }
[[ -f "$CACHE" ]] || { echo "FATAL: feature cache missing: $CACHE" >&2; exit 1; }
cd "$REPO"
$PY -m src.analysis.encoding.run_encoding \
    --subject "$SUBJECT" --model-name "$MODEL_NAME" --out-dir "$OUT" \
    --feature-cache "$CACHE" --checkpoint "$CKPT" --ridge-batched \
    $FRONTEND
echo "[ccctrl] done $MODEL_NAME subj$SUBJECT"
