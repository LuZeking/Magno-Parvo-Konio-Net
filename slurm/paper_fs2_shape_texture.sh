#!/bin/bash
# Table 1 / Fig. 2a on the TSN of record (2026-10-19): the stream-level
# shape/texture probe (src.analysis.behavior.stream_shape_texture, --checkpoint
# path, the paper's front end) for each run in RUN_LIST, one array element per
# run. As slurm/shape_texture.sh, but run from the pinned code and with the
# front end the checkpoint was trained with: the DVD-diet runs get
# --dvd-age-months <age>, and dkl_frontend.check_frontend refuses anything else.
# `untrained_dvd<age>` = untrained network (seed 0) on that DVD front end.
#
#   mkdir -p /scratch/byrong/mpk/shape_texture/logs/2026-10-19_fs2
#   env -u TMPDIR sbatch --array=0-$((N-1)) [--export=ALL,RUN_LIST="..."] slurm/paper_fs2_shape_texture.sh
#
# Writes $RES_REPO/results/shape_texture/<run>.json (next to the BA's
# <model>_trained.json) and the PCA/readout features to
# /scratch/byrong/mpk/shape_texture/features/2026-10-19_fs2/<run>.npz.
# SMOKE=1 writes both under /scratch/byrong/mpk/shape_texture/smoke/2026-10-19_fs2.
# Memory: earlier probe jobs peaked at 17.6 GB MaxRSS (slurm/shape_texture_grouped.sh).
#SBATCH --job-name=fs2_shptex
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --array=0-4
#SBATCH --output=/scratch/byrong/mpk/shape_texture/logs/2026-10-19_fs2/shptex_%A_%a.out

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
  if [[ $RUN =~ ^fs2_dvd(fix)?([0-9]+)(swap)?_ ]]; then FE=(--dvd-age-months "${BASH_REMATCH[2]}"); if [[ ${BASH_REMATCH[3]} == swap ]]; then FE+=(--transform-target parvo); fi; fi
fi

OUT=$RES_REPO/results/shape_texture
FEAT=/scratch/byrong/mpk/shape_texture/features/2026-10-19_fs2
if [[ ${SMOKE:-0} == 1 ]]; then
  OUT=/scratch/byrong/mpk/shape_texture/smoke/2026-10-19_fs2; FEAT=$OUT
fi
DEST=$OUT/$RUN.json
if [[ -e $DEST && ${OVERWRITE:-0} != 1 ]]; then
  echo "FATAL: $DEST exists; refusing to overwrite" >&2; exit 1
fi
mkdir -p "$OUT" "$FEAT"
echo "[fs2_shptex] task=$SLURM_ARRAY_TASK_ID run=$RUN ${CK[*]} ${FE[*]} -> $DEST code=$(cat "$CODE/COMMIT") $(hostname) $(date -Is)"
cd "$CODE"
$PY -m src.analysis.behavior.stream_shape_texture \
    --model-name "$RUN" --out "$OUT" --save-features "$FEAT/$RUN.npz" \
    "${CK[@]}" "${FE[@]}"
echo "[fs2_shptex] done $RUN $(date -Is)"
