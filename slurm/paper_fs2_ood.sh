#!/bin/bash
# Fig. 5 on the TSN of record (2026-10-19): OOD robustness on corrupted
# MiniEcoset (tmp_analysis/ood_robustness.py, all 5,600 test images), as
# slurm/ood_robustness.sh, for each run in RUN_LIST (checkpoint_final.pt),
# from the pinned code and with the checkpoint's own front end (DVD runs:
# --dvd-age-months <age>; the corruption is applied to the RGB image before it).
#
#   mkdir -p /scratch/byrong/mpk/ood/logs/2026-10-19_fs2
#   env -u TMPDIR sbatch --array=0-$((N-1)) [--export=ALL,RUN_LIST="..."] slurm/paper_fs2_ood.sh
#
# Writes $RES_REPO/results/ood/<run>.json (the BA's final-checkpoint directory;
# the figures read results/ood_bestep).
# SMOKE=1: 32 images, output under /scratch/byrong/mpk/ood/smoke/2026-10-19_fs2.
#SBATCH --job-name=fs2_ood
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=/scratch/byrong/mpk/ood/logs/2026-10-19_fs2/ood_%A_%a.out

set -euo pipefail
trap '[[ ${TMPDIR:-unset} == unset ]] || rm -rf $TMPDIR' EXIT
CODE=${CODE:-/home/byrong/projects/TSN/fusion_stage1_code/c543636}
RES_REPO=${RES_REPO:-/home/byrong/projects/TSN/Magno-Parvo-Konio-Net/.claude/worktrees/fusion-stage1}
export OPENBLAS_NUM_THREADS=16 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$CODE:$CODE/tmp_analysis"
PY=/home/byrong/anaconda3/envs/mpk/bin/python

read -r -a RUNS <<< "${RUN_LIST:-fs2_dvd300_avg_w050_ls_seed0 fs2_dvd300_avg_w050_ls_seed1 fs2_dvd300_avg_w050_ls_seed2 fs2_dvdfix300_avg_w050_ls_seed0}"
RUN=${RUNS[$SLURM_ARRAY_TASK_ID]}
FE=()
if [[ $RUN =~ ^fs2_dvd(fix)?([0-9]+)_ ]]; then FE=(--dvd-age-months "${BASH_REMATCH[2]}"); fi

OUT=$RES_REPO/results/ood
EXTRA=()
if [[ ${SMOKE:-0} == 1 ]]; then OUT=/scratch/byrong/mpk/ood/smoke/2026-10-19_fs2; EXTRA=(--n-images 32); fi
DEST=$OUT/$RUN.json
if [[ -e $DEST && ${OVERWRITE:-0} != 1 ]]; then
  echo "FATAL: $DEST exists; refusing to overwrite" >&2; exit 1
fi
mkdir -p "$OUT"
echo "[fs2_ood] task=$SLURM_ARRAY_TASK_ID run=$RUN ${FE[*]} -> $DEST code=$(cat "$CODE/COMMIT") $(hostname) $(date -Is)"
cd "$CODE"
$PY tmp_analysis/ood_robustness.py \
    --model-name "$RUN" --checkpoint "$RUN" --out-dir "$OUT" "${FE[@]}" "${EXTRA[@]}"
echo "[fs2_ood] done $RUN $(date -Is)"
