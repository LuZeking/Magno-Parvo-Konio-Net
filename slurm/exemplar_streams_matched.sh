#!/bin/bash
# PLAN_EXEMPLAR addendum (2026-09-22), array B': the exemplar-held-out stream probes on the six checkpoints
# of the accuracy-matched control (Fig. 2e): the diet's best saved checkpoints and the matched no-diet
# checkpoints. Same script and sizing as slurm/exemplar_streams.sh; Q0 must reproduce the stored JSON.
#
#   sbatch slurm/exemplar_streams_matched.sh
#SBATCH --job-name=mpk_exmatched
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-5%6
#SBATCH --output=/scratch/byrong/mpk/exemplar/logs/exmatched_%A_%a.out

set -euo pipefail
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1

PY=/home/byrong/anaconda3/envs/mpk/bin/python
RUNS=/scratch/byrong/VST_data/mpk_runs
MATCHED=/scratch/byrong/VST_data/mpk_runs_matched
FEAT=/scratch/byrong/mpk/exemplar/features/2026-09-22_streams_matched
OUT=/scratch/byrong/mpk/exemplar/results/2026-09-22_streams_matched
cd "$SLURM_SUBMIT_DIR"

CONFIG=(
  "dkl_bcs_seed0_ep125|$RUNS/dkl_bcs_seed0/checkpoint_125.pt|results/shape_texture_bestep|"
  "dkl_bcs_seed1_ep075|$RUNS/dkl_bcs_seed1/checkpoint_075.pt|results/shape_texture_bestep|"
  "dkl_bcs_seed2_ep050|$RUNS/dkl_bcs_seed2/checkpoint_050.pt|results/shape_texture_bestep|"
  "dkl_base_matched_seed0_ep010|$MATCHED/dkl_base_seed0/checkpoint_010.pt|results/shape_texture_matched|--no-lum-blur --no-lum-cs"
  "dkl_base_matched_seed1_ep010|$MATCHED/dkl_base_seed1/checkpoint_010.pt|results/shape_texture_matched|--no-lum-blur --no-lum-cs"
  "dkl_base_matched_seed2_ep011|$MATCHED/dkl_base_seed2/checkpoint_011.pt|results/shape_texture_matched|--no-lum-blur --no-lum-cs"
)
IFS='|' read -r NAME CKPT REF EXTRA <<< "${CONFIG[$SLURM_ARRAY_TASK_ID]}"
[[ -f "$CKPT" ]] || { echo "FATAL: checkpoint missing: $CKPT" >&2; exit 1; }
echo "[exmatched] task=$SLURM_ARRAY_TASK_ID $NAME ckpt=$CKPT ref=$REF extra=$EXTRA"
PYTHONPATH="$SLURM_SUBMIT_DIR" $PY -m src.analysis.behavior.stream_features \
    --model-name "$NAME" --checkpoint "$CKPT" --reference-dir "$REF" \
    --feat-dir "$FEAT" --out-dir "$OUT" $EXTRA
echo "[exmatched] done $NAME"
