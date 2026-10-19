#!/bin/bash
# Table 1 / Fig. 2a behaviour on the TSN of record (2026-10-19): cue-conflict
# shape bias with the per-stream decomposition (src.analysis.behavior.cue_conflict),
# as slurm/cue_conflict.sh, for each run in RUN_LIST, from the pinned code and
# with the checkpoint's own front end (DVD runs: --dvd-age-months <age>;
# dkl_frontend.check_frontend refuses anything else).
# `untrained_dvd<age>` = untrained network (seed 0) on that DVD front end.
#
#   mkdir -p /scratch/byrong/mpk/cue_conflict/logs/2026-10-19_fs2
#   env -u TMPDIR sbatch --array=0-$((N-1)) [--export=ALL,RUN_LIST="..."] slurm/paper_fs2_cue_conflict.sh
#
# Writes $RES_REPO/results/cue_conflict/<run>.json and <run>_logits.npz.
# SMOKE=1: 20 bootstrap draws, output under /scratch/byrong/mpk/cue_conflict/smoke/2026-10-19_fs2.
#SBATCH --job-name=fs2_cueconf
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-4
#SBATCH --output=/scratch/byrong/mpk/cue_conflict/logs/2026-10-19_fs2/cueconf_%A_%a.out

set -euo pipefail
trap '[[ ${TMPDIR:-unset} == unset ]] || rm -rf $TMPDIR' EXIT
CODE=${CODE:-/home/byrong/projects/TSN/fusion_stage1_code/c543636}
RES_REPO=${RES_REPO:-/home/byrong/projects/TSN/Magno-Parvo-Konio-Net/.claude/worktrees/fusion-stage1}
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$CODE:$CODE/tmp_analysis"
PY=/home/byrong/anaconda3/envs/mpk/bin/python

read -r -a RUNS <<< "${RUN_LIST:-fs2_dvd300_avg_w050_ls_seed0 fs2_dvd300_avg_w050_ls_seed1 fs2_dvd300_avg_w050_ls_seed2 fs2_dvdfix300_avg_w050_ls_seed0 untrained_dvd300}"
RUN=${RUNS[$SLURM_ARRAY_TASK_ID]}
CK=(); FE=()
if [[ $RUN =~ ^untrained_dvd([0-9]+)$ ]]; then
  FE=(--dvd-age-months "${BASH_REMATCH[1]}")
else
  CK=(--checkpoint "$RUN")   # an fs2 run name -> its checkpoint_final.pt
  if [[ $RUN =~ ^fs2_dvd(fix)?([0-9]+)_ ]]; then FE=(--dvd-age-months "${BASH_REMATCH[2]}"); fi
fi

OUT=$RES_REPO/results/cue_conflict
EXTRA=()
if [[ ${SMOKE:-0} == 1 ]]; then
  OUT=/scratch/byrong/mpk/cue_conflict/smoke/2026-10-19_fs2; EXTRA=(--n-boot 20)
fi
for DEST in "$OUT/$RUN.json" "$OUT/${RUN}_logits.npz"; do
  if [[ -e $DEST && ${OVERWRITE:-0} != 1 ]]; then
    echo "FATAL: $DEST exists; refusing to overwrite" >&2; exit 1
  fi
done
mkdir -p "$OUT"
echo "[fs2_cueconf] task=$SLURM_ARRAY_TASK_ID run=$RUN ${CK[*]} ${FE[*]} -> $OUT code=$(cat "$CODE/COMMIT") $(hostname) $(date -Is)"
cd "$CODE"
$PY -m src.analysis.behavior.cue_conflict \
    --model-name "$RUN" --out-dir "$OUT" "${CK[@]}" "${FE[@]}" "${EXTRA[@]}"
echo "[fs2_cueconf] done $RUN $(date -Is)"
