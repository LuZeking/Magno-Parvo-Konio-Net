#!/bin/bash
# Unit-level layer-4 features for the 1-to-1 test (docs/PLAN_FINZI_1TO1_2026-09-24.md §3-4).
# One task per subject: images read once, all 17 networks run (15 TSN + 2 TDANN).
#
#   sbatch slurm/finzi_1to1_extract.sh             # all 8 subjects
#   SMOKE=1 sbatch --array=0 slurm/finzi_1to1_extract.sh
#
# Output dirs must already exist and be pinned (slurm/finzi_1to1_pin_dirs.sh).
# 24G = the A5000 fair share; features held in RAM are ~6 GB float16 per subject.
#SBATCH --job-name=f1to1_extract
#SBATCH --partition=agcichy
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --array=0-7
#SBATCH --exclude=c025,g017
#SBATCH --output=slurm/logs/f1to1_extract_%A_%a.out
# g017 has no /scratch mount (the env and every input live there).

set -euo pipefail
unset SLURM_CPU_BIND SLURM_CPU_BIND_LIST SLURM_CPU_BIND_TYPE SLURM_CPU_BIND_VERBOSE
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl_$SLURM_JOB_ID"

SUBJECT=$(( SLURM_ARRAY_TASK_ID + 1 ))
PY=/scratch/byrong/anaconda3/envs/mpk/bin/python
EXTRA=""
if [ "${SMOKE:-0}" = "1" ]; then
  export FINZI_OUT_ROOT=/scratch/byrong/TSN/finzi_1to1/smoke
  EXTRA="--limit 20"
fi
cd "$SLURM_SUBMIT_DIR"
echo "[f1to1_extract] subject=$SUBJECT host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-none} start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
$PY -m src.analysis.finzi_1to1.extract --subject "$SUBJECT" $EXTRA
echo "[f1to1_extract] done $(date -Is)"
