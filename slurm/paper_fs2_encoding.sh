#!/bin/bash
# NSD encoding of the untrained anchor on the TSN of record's front end
# (2026-10-19): an untrained network (seed from the name) through the DVD diet
# at the age in its name (untrained_dvd300_seed0 -> --dvd-age-months 300), as
# slurm/encode_fs2.sh: same OUT/CACHE directories, --cue-conflict
# --ridge-batched, from the pinned code. Array element = run x subject
# (TASK / 8 indexes RUN_LIST, TASK % 8 + 1 is the subject).
#
#   mkdir -p /scratch/byrong/mpk/encoding/logs/2026-10-19_fs2
#   env -u TMPDIR sbatch --array=0-$((8*N-1)) [--export=ALL,RUN_LIST="..."] slurm/paper_fs2_encoding.sh
#
# Writes /scratch/byrong/mpk/encoding/results/2026-10-16_fs2/<run>_subjNN.npz
# (cache: /scratch/byrong/mpk/encoding/cache/2026-10-16_fs2/<run>_subjNN.npz).
# --mem/--time as encode_fs2.sh (batched fits 21-34.7 GB MaxRSS + 25%; 3-8 h per subject).
#SBATCH --job-name=fs2_enc_untr
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=44G
#SBATCH --time=14:00:00
#SBATCH --array=0-7
#SBATCH --output=/scratch/byrong/mpk/encoding/logs/2026-10-19_fs2/enc_%A_%a.out

set -euo pipefail
trap '[[ ${TMPDIR:-unset} == unset ]] || rm -rf $TMPDIR' EXIT
CODE=${CODE:-/home/byrong/projects/TSN/fusion_stage1_code/c543636}
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$CODE:$CODE/tmp_analysis"
PY=/home/byrong/anaconda3/envs/mpk/bin/python

read -r -a RUNS <<< "${RUN_LIST:-untrained_dvd300_seed0}"
TASK=$SLURM_ARRAY_TASK_ID
RUN=${RUNS[$(( TASK / 8 ))]}
SUBJECT=$(( TASK % 8 + 1 ))
if [[ $RUN =~ ^untrained_dvd([0-9]+)_seed([0-9]+)$ ]]; then
  ARGS=(--seed "${BASH_REMATCH[2]}" --dvd-age-months "${BASH_REMATCH[1]}")
else
  echo "FATAL: $RUN is not untrained_dvd<age>_seed<n> (trained runs: slurm/encode_fs2.sh)" >&2; exit 1
fi
OUT=/scratch/byrong/mpk/encoding/results/2026-10-16_fs2
CACHE=/scratch/byrong/mpk/encoding/cache/2026-10-16_fs2/${RUN}_subj$(printf %02d "$SUBJECT").npz
DEST=$OUT/${RUN}_subj$(printf %02d "$SUBJECT").npz
if [[ ${SMOKE:-0} == 1 ]]; then
  OUT=/scratch/byrong/mpk/encoding/smoke/2026-10-19_fs2
  CACHE=$OUT/${RUN}_subj$(printf %02d "$SUBJECT")_cache.npz; DEST=$OUT/${RUN}_subj$(printf %02d "$SUBJECT").npz
fi
for F in "$DEST" "$CACHE"; do
  if [[ -e $F && ${OVERWRITE:-0} != 1 ]]; then
    echo "FATAL: $F exists; refusing to overwrite" >&2; exit 1
  fi
done
mkdir -p "$OUT" "$(dirname "$CACHE")"
echo "[fs2_enc_untr] task=$TASK run=$RUN subj=$SUBJECT ${ARGS[*]} -> $DEST code=$(cat "$CODE/COMMIT") $(hostname) $(date -Is)"
cd "$CODE"
$PY -m src.analysis.encoding.run_encoding \
    --subject "$SUBJECT" --model-name "$RUN" --out-dir "$OUT" \
    --feature-cache "$CACHE" --cue-conflict --ridge-batched "${ARGS[@]}"
echo "[fs2_enc_untr] done $RUN subj$SUBJECT $(date -Is)"
