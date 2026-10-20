#!/bin/bash
# Table 1 (exemplars held out) on the TSN of record (2026-10-19): stored
# stage-pooled stream features + the exemplar-held-out probes
# (src.analysis.behavior.stream_features), as slurm/exemplar_streams.sh, for
# each run in RUN_LIST. Q0 must reproduce the probe JSON that
# slurm/paper_fs2_shape_texture.sh wrote for the same run, so submit this with
# --dependency=aftercorr:<that array> and the same RUN_LIST order.
# `untrained_dvd<age>` = untrained network (seed 0) on that DVD front end.
#
#   mkdir -p /scratch/byrong/mpk/exemplar/logs/2026-10-19_fs2
#   env -u TMPDIR sbatch --array=0-$((N-1)) --dependency=aftercorr:<shptex> slurm/paper_fs2_exemplar_streams.sh
#
# Writes into the BA's directories, read by tmp_analysis/figs_v2/exemplar_aggregate.py:
#   /scratch/byrong/mpk/exemplar/features/2026-09-21_streams/<run>_streams.npz
#   /scratch/byrong/mpk/exemplar/results/2026-09-21_streams/<run>_stream_exemplar.json
# SMOKE=1 writes both under /scratch/byrong/mpk/exemplar/smoke/2026-10-19_fs2.
#SBATCH --job-name=fs2_exstreams
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-4
#SBATCH --output=/scratch/byrong/mpk/exemplar/logs/2026-10-19_fs2/exstreams_%A_%a.out

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

FEAT=/scratch/byrong/mpk/exemplar/features/2026-09-21_streams
OUT=/scratch/byrong/mpk/exemplar/results/2026-09-21_streams
REF=$RES_REPO/results/shape_texture
if [[ ${SMOKE:-0} == 1 ]]; then
  FEAT=/scratch/byrong/mpk/exemplar/smoke/2026-10-19_fs2; OUT=$FEAT
fi
# stream_features silently reuses an existing feature file, so an existing one
# is refused as well as an existing result.
for DEST in "$OUT/${RUN}_stream_exemplar.json" "$FEAT/${RUN}_streams.npz"; do
  if [[ -e $DEST && ${OVERWRITE:-0} != 1 ]]; then
    echo "FATAL: $DEST exists; refusing to overwrite" >&2; exit 1
  fi
done
if [[ ${SMOKE:-0} != 1 && ! -f $REF/$RUN.json ]]; then
  echo "FATAL: $REF/$RUN.json missing -- the Q0 reproduction check needs the probe JSON" >&2; exit 1
fi
mkdir -p "$FEAT" "$OUT"
echo "[fs2_exstreams] task=$SLURM_ARRAY_TASK_ID run=$RUN ${CK[*]} ${FE[*]} -> $OUT code=$(cat "$CODE/COMMIT") $(hostname) $(date -Is)"
cd "$CODE"
$PY -m src.analysis.behavior.stream_features \
    --model-name "$RUN" --feat-dir "$FEAT" --out-dir "$OUT" --reference-dir "$REF" \
    "${CK[@]}" "${FE[@]}"
echo "[fs2_exstreams] done $RUN $(date -Is)"
