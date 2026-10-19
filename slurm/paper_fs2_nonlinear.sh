#!/bin/bash
# Fig. 4b on the TSN of record (2026-10-19), step 1 of 2: pooled per-stream
# penultimate features for the nonlinear readout test
# (tmp_analysis/nonlinear_readout_features.py), as slurm/nonlinear_features.sh,
# for each run in RUN_LIST (checkpoint_final.pt; tag <run>_final), from the
# pinned code. MiniEcoset train/val go through the trainer's StreamInputBuilder
# (the DVD diet at dvd_end_months); the cue-conflict set through
# dkl_frontend.checkpoint_frontend of the run's config. Step 2 (CPU) is
# slurm/paper_fs2_nonlinear_sweep.sh with --dependency=aftercorr:<this array>.
#
#   mkdir -p /scratch/byrong/mpk/nonlinear_readout/logs/2026-10-19_fs2
#   env -u TMPDIR sbatch --array=0-$((N-1)) [--export=ALL,RUN_LIST="..."] slurm/paper_fs2_nonlinear.sh
#
# Writes /scratch/byrong/TSN/nonlinear_readout/features/2026-09-10/<run>_final_features.npz (the BA's directory).
# SMOKE=1: --train-limit 512, under /scratch/byrong/TSN/nonlinear_readout/features/2026-09-10/smoke_fs2.
# GPU: one A5000; 22G stays under its node's fair share (~23G per GPU on a 95G node).
#SBATCH --job-name=fs2_nlfeat
#SBATCH --partition=agcichy
#SBATCH --qos=standard
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=22G
#SBATCH --time=06:00:00
#SBATCH --array=0-3
#SBATCH --output=/scratch/byrong/mpk/nonlinear_readout/logs/2026-10-19_fs2/nlfeat_%A_%a.out

set -euo pipefail
trap '[[ ${TMPDIR:-unset} == unset ]] || rm -rf $TMPDIR' EXIT
CODE=${CODE:-/home/byrong/projects/TSN/fusion_stage1_code/c543636}
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$CODE:$CODE/tmp_analysis"
PY=/home/byrong/anaconda3/envs/mpk/bin/python

read -r -a RUNS <<< "${RUN_LIST:-fs2_dvd300_avg_w050_ls_seed0 fs2_dvd300_avg_w050_ls_seed1 fs2_dvd300_avg_w050_ls_seed2 fs2_dvdfix300_avg_w050_ls_seed0}"
RUN=${RUNS[$SLURM_ARRAY_TASK_ID]}
TAG=${RUN}_final
OUT=/scratch/byrong/TSN/nonlinear_readout/features/2026-09-10
EXTRA=()
if [[ ${SMOKE:-0} == 1 ]]; then OUT=$OUT/smoke_fs2; EXTRA=(--train-limit 512); fi
DEST=$OUT/${TAG}_features.npz
if [[ -e $DEST && ${OVERWRITE:-0} != 1 ]]; then
  echo "FATAL: $DEST exists; refusing to overwrite" >&2; exit 1
fi
mkdir -p "$OUT"
echo "[fs2_nlfeat] task=$SLURM_ARRAY_TASK_ID run=$RUN -> $DEST code=$(cat "$CODE/COMMIT") $(hostname) $(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
cd "$CODE"
$PY tmp_analysis/nonlinear_readout_features.py \
    --run-dir "$RUN" --checkpoint-name checkpoint_final.pt --tag "$TAG" --out-dir "$OUT" "${EXTRA[@]}"
echo "[fs2_nlfeat] done $RUN $(date -Is)"
