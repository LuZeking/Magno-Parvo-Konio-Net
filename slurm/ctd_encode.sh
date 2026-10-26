#!/bin/bash
# Cortex-to-decision test (docs/PLAN_CORTEX_DECISION_2026-10-26.md §6): refit the
# paper's NSD encoders (Methods 4.5) for the model, its control, the channel
# swap, the untrained anchor and the input-level (Gabor) floor, this time
# storing the per-image predictions of the cue-conflict and template designs
# and the fitted estimators. Same code path, seed and feature caches as
# slurm/encode_fs2.sh / slurm/paper_fs2_encoding.sh; every trained/untrained
# refit must reproduce its stored 2026-10-16_fs2 fit (--reproduce-check,
# tol 1e-3) or the task exits 3 after saving.
#
#   mkdir -p /scratch/byrong/mpk/encoding/logs/2026-10-26_ctd \
#            /scratch/byrong/mpk/encoding/results/2026-10-26_ctd \
#            /scratch/byrong/mpk/encoding/cache/2026-10-26_ctd
#   # pin the output directories to a healthy OST first (probe as in the
#   # Lustre memory; e.g. lfs setstripe -i 30 <dir>)
#   E=--export=ALL,CODE=/home/byrong/projects/TSN/fusion_stage1_code/<sha>
#   ENC=$(env -u TMPDIR sbatch --parsable $E slurm/ctd_encode.sh)
#   env -u TMPDIR sbatch $E --dependency=afterok:$ENC slurm/ctd_analysis.sh
#
# 11 runs x 8 subjects = 88 tasks (TASK / 8 = run group, TASK % 8 + 1 = subject).
# CPU only (himalaya on torch CPU). --mem as encode_fs2.sh (batched fits
# 21-34.7 GB MaxRSS + 25%); the ridge alone took 3-8 h per subject there, and
# the feature caches are reused, so the trained tasks pay no feature extraction.
#SBATCH --job-name=mpk_ctd_enc
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=44G
#SBATCH --time=14:00:00
#SBATCH --array=0-87%16
#SBATCH --output=/scratch/byrong/mpk/encoding/logs/2026-10-26_ctd/enc_%A_%a.out
set -euo pipefail
trap '[[ ${TMPDIR:-unset} == unset ]] || rm -rf $TMPDIR' EXIT
: "${CODE:?pass --export=ALL,CODE=<pinned code dir>}"
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$CODE"
TASK=$SLURM_ARRAY_TASK_ID
SUBJECT=$(( TASK % 8 + 1 ))
GROUP=$(( TASK / 8 ))
SS=$(printf %02d "$SUBJECT")
CKPT_ROOT=/scratch/byrong/TSN/fusion_stage1/checkpoints/2026-10-12_stage2
REF_DIR=/scratch/byrong/mpk/encoding/results/2026-10-16_fs2
FS2_CACHE=/scratch/byrong/mpk/encoding/cache/2026-10-16_fs2
OUT=/scratch/byrong/mpk/encoding/results/2026-10-26_ctd
CKPT=""
case $GROUP in
  0|1|2)   RUN=fs2_dvd300_avg_w050_ls_seed$GROUP
           CKPT=$CKPT_ROOT/$RUN/checkpoint_final.pt
           ARGS=(--checkpoint "$CKPT" --dvd-age-months 300) ;;
  3|4|5)   RUN=fs2_dvdfix300_avg_w050_ls_seed$(( GROUP - 3 ))
           CKPT=$CKPT_ROOT/$RUN/checkpoint_final.pt
           ARGS=(--checkpoint "$CKPT" --dvd-age-months 300) ;;
  6|7|8)   RUN=fs2_dvd300swap_avg_w050_ls_seed$(( GROUP - 6 ))
           CKPT=$CKPT_ROOT/$RUN/checkpoint_final.pt
           ARGS=(--checkpoint "$CKPT" --dvd-age-months 300 --transform-target parvo) ;;
  9)       RUN=untrained_dvd300_seed0
           ARGS=(--seed 0 --dvd-age-months 300) ;;
  10)      RUN=gabor_dvd300
           ARGS=(--features gabor --dvd-age-months 300) ;;
  *) echo "bad group $GROUP" >&2; exit 1 ;;
esac
if [[ $GROUP -le 9 ]]; then
  CACHE=$FS2_CACHE/${RUN}_subj$SS.npz          # provenance-verified reuse: the ridge only
  REF=$REF_DIR/${RUN}_subj$SS.npz
  [[ -f "$REF" ]] || { echo "FATAL: stored fit missing: $REF" >&2; exit 1; }
  [[ -f "$CACHE" ]] || { echo "FATAL: feature cache missing: $CACHE" >&2; exit 1; }
  ARGS+=(--reproduce-check "$REF" --reproduce-tol 1e-3)
else
  CACHE=/scratch/byrong/mpk/encoding/cache/2026-10-26_ctd/${RUN}_subj$SS.npz
fi
[[ -z $CKPT || -f $CKPT ]] || { echo "FATAL: checkpoint missing: $CKPT" >&2; exit 1; }
DEST=$OUT/${RUN}_subj$SS.npz
if [[ -e $DEST && ${OVERWRITE:-0} != 1 ]]; then
  echo "FATAL: $DEST exists; refusing to overwrite" >&2; exit 1
fi
mkdir -p "$OUT" "$(dirname "$CACHE")"
PY=/home/byrong/anaconda3/envs/mpk/bin/python
echo "[ctd_enc] task=$TASK run=$RUN subj=$SUBJECT ${ARGS[*]} -> $DEST code=$(cat "$CODE/COMMIT") $(hostname) $(date -Is)"
cd "$CODE"
$PY -m src.analysis.encoding.run_encoding \
    --subject "$SUBJECT" --model-name "$RUN" --out-dir "$OUT" \
    --feature-cache "$CACHE" --cue-conflict --ridge-batched \
    --save-predictions --template-design miniecoset16 --save-encoder \
    "${ARGS[@]}"
echo "[ctd_enc] done $RUN subj$SUBJECT $(date -Is)"
