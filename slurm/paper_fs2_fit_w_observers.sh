#!/bin/bash
# Fig. S9 on the TSN of record (2026-10-19): the readout weight w fitted to
# each of the ten Geirhos observers (tmp_analysis/fit_w_observers.py), as
# slurm/fit_w_observers.sh (w-step 0.001, 1,000 bootstrap draws), for each run
# in RUN_LIST (checkpoint_final.pt; named <run>_final), from the pinned code
# and with the checkpoint's own front end (DVD runs: --dvd-age-months <age>).
#
#   mkdir -p /scratch/byrong/mpk/fit_w_observers/logs/2026-10-19_fs2
#   env -u TMPDIR sbatch --array=0-$((N-1)) [--export=ALL,RUN_LIST="..."] slurm/paper_fs2_fit_w_observers.sh
#
# Writes $RES_REPO/results/fit_w_observers/<run>_final_wfit.json (+ _wfit_curves.npz).
# SMOKE=1: w-step 0.1 and 20 draws/splits/permutations, output under
# /scratch/byrong/mpk/fit_w_observers/smoke/2026-10-19_fs2.
#SBATCH --job-name=fs2_wfit
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --array=0-3
#SBATCH --output=/scratch/byrong/mpk/fit_w_observers/logs/2026-10-19_fs2/wfit_%A_%a.out

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
if [[ $RUN =~ ^fs2_dvd(fix)?([0-9]+)(swap)?_ ]]; then FE=(--dvd-age-months "${BASH_REMATCH[2]}"); if [[ ${BASH_REMATCH[3]} == swap ]]; then FE+=(--transform-target parvo); fi; fi

OUT=$RES_REPO/results/fit_w_observers
EXTRA=(--w-step 0.001 --n-boot 1000)
if [[ ${SMOKE:-0} == 1 ]]; then
  OUT=/scratch/byrong/mpk/fit_w_observers/smoke/2026-10-19_fs2
  EXTRA=(--w-step 0.1 --n-boot 20 --n-splits 20 --n-perm 20)
fi
for DEST in "$OUT/${NAME}_wfit.json" "$OUT/${NAME}_wfit_curves.npz"; do
  if [[ -e $DEST && ${OVERWRITE:-0} != 1 ]]; then
    echo "FATAL: $DEST exists; refusing to overwrite" >&2; exit 1
  fi
done
mkdir -p "$OUT"
echo "[fs2_wfit] task=$SLURM_ARRAY_TASK_ID run=$RUN ${FE[*]} -> $OUT code=$(cat "$CODE/COMMIT") $(hostname) $(date -Is)"
cd "$CODE"
$PY tmp_analysis/fit_w_observers.py \
    --model-name "$NAME" --checkpoint "$RUN" --out-dir "$OUT" "${FE[@]}" "${EXTRA[@]}"
echo "[fs2_wfit] done $RUN $(date -Is)"
