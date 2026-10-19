#!/bin/bash
# Fig. 5 on the TSN of record (2026-10-19): error consistency (Cohen's kappa)
# with the ten Geirhos observers across the readout weight w
# (tmp_analysis/sweep_error_consistency.py, w-step 0.05), as
# slurm/sweep_error_consistency.sh, for each run in RUN_LIST
# (checkpoint_final.pt; named <run>_final), from the pinned code and with the
# checkpoint's own front end (DVD runs: --dvd-age-months <age>).
#
#   mkdir -p /scratch/byrong/mpk/sweep_error_consistency/logs/2026-10-19_fs2
#   env -u TMPDIR sbatch --array=0-$((N-1)) [--export=ALL,RUN_LIST="..."] slurm/paper_fs2_sweep_error_consistency.sh
#
# Writes $RES_REPO/results/sweep_error_consistency/<run>_final_kappa_sweep.json.
# SMOKE=1: w-step 0.5, output under /scratch/byrong/mpk/sweep_error_consistency/smoke/2026-10-19_fs2.
#SBATCH --job-name=fs2_ksweep
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-3
#SBATCH --output=/scratch/byrong/mpk/sweep_error_consistency/logs/2026-10-19_fs2/ksweep_%A_%a.out

set -euo pipefail
trap '[[ ${TMPDIR:-unset} == unset ]] || rm -rf $TMPDIR' EXIT
CODE=${CODE:-/home/byrong/projects/TSN/fusion_stage1_code/c543636}
RES_REPO=${RES_REPO:-/home/byrong/projects/TSN/Magno-Parvo-Konio-Net/.claude/worktrees/fusion-stage1}
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$CODE:$CODE/tmp_analysis"
PY=/home/byrong/anaconda3/envs/mpk/bin/python

read -r -a RUNS <<< "${RUN_LIST:-fs2_dvd300_avg_w050_ls_seed0 fs2_dvd300_avg_w050_ls_seed1 fs2_dvd300_avg_w050_ls_seed2 fs2_dvdfix300_avg_w050_ls_seed0}"
RUN=${RUNS[$SLURM_ARRAY_TASK_ID]}
NAME=${RUN}_final
FE=()
if [[ $RUN =~ ^fs2_dvd(fix)?([0-9]+)_ ]]; then FE=(--dvd-age-months "${BASH_REMATCH[2]}"); fi

OUT=$RES_REPO/results/sweep_error_consistency
EXTRA=(--w-step 0.05)
if [[ ${SMOKE:-0} == 1 ]]; then
  OUT=/scratch/byrong/mpk/sweep_error_consistency/smoke/2026-10-19_fs2; EXTRA=(--w-step 0.5)
fi
DEST=$OUT/${NAME}_kappa_sweep.json
if [[ -e $DEST && ${OVERWRITE:-0} != 1 ]]; then
  echo "FATAL: $DEST exists; refusing to overwrite" >&2; exit 1
fi
mkdir -p "$OUT"
echo "[fs2_ksweep] task=$SLURM_ARRAY_TASK_ID run=$RUN ${FE[*]} -> $DEST code=$(cat "$CODE/COMMIT") $(hostname) $(date -Is)"
cd "$CODE"
$PY tmp_analysis/sweep_error_consistency.py \
    --model-name "$NAME" --checkpoint "$RUN" --out-dir "$OUT" "${FE[@]}" "${EXTRA[@]}"
echo "[fs2_ksweep] done $RUN $(date -Is)"
