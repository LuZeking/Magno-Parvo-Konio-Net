#!/bin/bash
# Fig. 4a on the TSN of record (2026-10-19): cue-conflict shape bias vs the
# readout weight w (tmp_analysis/fusion_sweep.py), as slurm/fusion_sweep.sh,
# for each run in RUN_LIST (checkpoint_final.pt, the EMA network; named
# <run>_final as the BA's final policy), from the pinned code and with the
# checkpoint's own front end (DVD runs: --dvd-age-months <age>).
#
#   mkdir -p /scratch/byrong/mpk/fusion_sweep/logs/2026-10-19_fs2
#   env -u TMPDIR sbatch --array=0-$((N-1)) [--export=ALL,RUN_LIST="..."] slurm/paper_fs2_fusion_sweep.sh
#
# Writes $RES_REPO/results/fusion_sweep/<run>_final_wsweep.json.
# SMOKE=1: w-step 0.5, 20 bootstrap draws, output under /scratch/byrong/mpk/fusion_sweep/smoke/2026-10-19_fs2.
#SBATCH --job-name=fs2_wsweep
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#SBATCH --output=/scratch/byrong/mpk/fusion_sweep/logs/2026-10-19_fs2/wsweep_%A_%a.out

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

OUT=$RES_REPO/results/fusion_sweep
EXTRA=(--w-step 0.05)
if [[ ${SMOKE:-0} == 1 ]]; then
  OUT=/scratch/byrong/mpk/fusion_sweep/smoke/2026-10-19_fs2; EXTRA=(--w-step 0.5 --n-boot 20)
fi
DEST=$OUT/${NAME}_wsweep.json
if [[ -e $DEST && ${OVERWRITE:-0} != 1 ]]; then
  echo "FATAL: $DEST exists; refusing to overwrite" >&2; exit 1
fi
mkdir -p "$OUT"
echo "[fs2_wsweep] task=$SLURM_ARRAY_TASK_ID run=$RUN ${FE[*]} -> $DEST code=$(cat "$CODE/COMMIT") $(hostname) $(date -Is)"
cd "$CODE"
$PY tmp_analysis/fusion_sweep.py \
    --model-name "$NAME" --checkpoint "$RUN" --out-dir "$OUT" "${FE[@]}" "${EXTRA[@]}"
echo "[fs2_wsweep] done $RUN $(date -Is)"
