#!/bin/bash
# ED Table 3 row 1 / S11 on the TSN of record (2026-10-19): the generality test
# on concat:<run> (both streams' penultimate units side by side, head
# [W/2, W/2]) for each run in RUN_LIST, from the pinned code. One element per
# run runs every per-network step the BA's concat rows went through
# (slurm/generality_cc.sh, generality_rules.sh, subspace_cc.sh) and adds the
# MiniEcoset features the BA's concat never had, so the accuracy gate exists:
#   extract cue_conflict -> unit_factor (median, positive, decile) -> subspace_factor
#   extract miniecoset   -> gain_accuracy for each of the four factorisations
# The front end comes from the run's config (generality/models.py ->
# dkl_frontend.checkpoint_frontend): the DVD diet for the fs2_dvd* runs; the
# MiniEcoset path goes through the trainer's StreamInputBuilder.
#
#   mkdir -p /scratch/byrong/mpk/generality/logs/2026-10-19_fs2
#   env -u TMPDIR sbatch --array=0-$((N-1)) [--export=ALL,RUN_LIST="..."] slurm/paper_fs2_generality.sh
#
# Writes into the BA's run directories (RUNID 2026-09-14_gen), tag concat_<run>:
#   /scratch/byrong/mpk/generality/features/2026-09-14_gen/concat_<run>_{cue_conflict,miniecoset}.npz
#   /scratch/byrong/mpk/generality/behaviour/2026-09-14_gen/concat_<run>_{unit_factor*,subspace,gain_accuracy*}.{json,npz}
# SMOKE=1: the two extractions on 64 images each, under /scratch/byrong/mpk/generality/smoke/2026-10-19_fs2.
# Time/memory: generality_cc.sh (minutes, 16G) + generality_mini.sh (10 h, 32G);
# two ResNet-18 trunks over the 263,200 + 5,600 MiniEcoset images dominate.
#SBATCH --job-name=fs2_gen
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=/scratch/byrong/mpk/generality/logs/2026-10-19_fs2/gen_%A_%a.out

set -euo pipefail
trap '[[ ${TMPDIR:-unset} == unset ]] || rm -rf $TMPDIR' EXIT
CODE=${CODE:-/home/byrong/projects/TSN/fusion_stage1_code/c543636}
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$CODE:$CODE/tmp_analysis"
PY=/home/byrong/anaconda3/envs/mpk/bin/python

read -r -a RUNS <<< "${RUN_LIST:-fs2_dvd300_avg_w050_ls_seed0 fs2_dvd300_avg_w050_ls_seed1 fs2_dvd300_avg_w050_ls_seed2 fs2_dvdfix300_avg_w050_ls_seed0}"
RUN=${RUNS[$SLURM_ARRAY_TASK_ID]}
MODEL=concat:$RUN
TAG=concat_$RUN
RUNID=${GEN_RUNID:-2026-09-14_gen}
FEAT=/scratch/byrong/mpk/generality/features/$RUNID
BEH=/scratch/byrong/mpk/generality/behaviour/$RUNID
if [[ ${SMOKE:-0} == 1 ]]; then FEAT=/scratch/byrong/mpk/generality/smoke/2026-10-19_fs2; BEH=$FEAT; fi
for DEST in "$FEAT/${TAG}_cue_conflict.npz" "$FEAT/${TAG}_miniecoset.npz" "$BEH/${TAG}_unit_factor.json"; do
  if [[ -e $DEST && ${OVERWRITE:-0} != 1 ]]; then
    echo "FATAL: $DEST exists; refusing to overwrite" >&2; exit 1
  fi
done
mkdir -p "$FEAT" "$BEH"
echo "[fs2_gen] task=$SLURM_ARRAY_TASK_ID model=$MODEL tag=$TAG -> $FEAT, $BEH code=$(cat "$CODE/COMMIT") $(hostname) $(date -Is)"
cd "$CODE"

if [[ ${SMOKE:-0} == 1 ]]; then
  $PY -m src.analysis.generality.extract --model "$MODEL" --set cue_conflict --out-dir "$FEAT" --limit 64
  $PY -m src.analysis.generality.extract --model "$MODEL" --set miniecoset --out-dir "$FEAT" --limit 64
  echo "[fs2_gen] smoke done $RUN $(date -Is)"; exit 0
fi

CC=$FEAT/${TAG}_cue_conflict.npz
$PY -m src.analysis.generality.extract --model "$MODEL" --set cue_conflict --out-dir "$FEAT"
$PY -m src.analysis.generality.unit_factor --features "$CC" --out-dir "$BEH" --n-splits 10
for RULE in positive decile; do
  $PY -m src.analysis.generality.unit_factor --features "$CC" --out-dir "$BEH" \
      --n-splits 10 --rule "$RULE" --out-suffix "_$RULE"
done
$PY -m src.analysis.generality.subspace_factor --features "$CC" --out-dir "$BEH" --n-splits 10
echo "[fs2_gen] cue-conflict steps done $(date -Is)"

VAL=$FEAT/${TAG}_miniecoset.npz
$PY -m src.analysis.generality.extract --model "$MODEL" --set miniecoset --out-dir "$FEAT"
$PY -m src.analysis.generality.gain_accuracy --factor "$BEH/${TAG}_unit_factor.npz" --val "$VAL" --out-dir "$BEH"
for RULE in positive decile; do
  $PY -m src.analysis.generality.gain_accuracy --factor "$BEH/${TAG}_unit_factor_$RULE.npz" \
      --val "$VAL" --out-dir "$BEH" --out-suffix "_$RULE"
done
$PY -m src.analysis.generality.gain_accuracy --factor "$BEH/${TAG}_subspace.npz" \
    --val "$VAL" --out-dir "$BEH" --out-suffix "_subspace"
echo "[fs2_gen] done $RUN $(date -Is)"
