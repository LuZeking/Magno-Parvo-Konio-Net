#!/bin/bash
# Fig. 5 on the TSN of record (2026-10-19): per-image, per-stream cue-conflict
# logits (tmp_analysis/perimage_logits.py), as slurm/perimage_logits.sh, for
# each run in RUN_LIST (checkpoint_final.pt; named <run>_final), from the
# pinned code and with the checkpoint's own front end (DVD runs:
# --dvd-age-months <age>). The per-image human correspondence is computed from
# these afterwards (tmp_analysis/perimage_correspondence.py).
#
#   mkdir -p /scratch/byrong/mpk/perimage_behavior/logs/2026-10-19_fs2
#   env -u TMPDIR sbatch --array=0-$((N-1)) [--export=ALL,RUN_LIST="..."] slurm/paper_fs2_perimage_logits.sh
#
# Writes /scratch/byrong/mpk/perimage_behavior/results/2026-09-07_geirhos_perimage/logits/<run>_final_perimage_logits.npz
# (the BA's directory). SMOKE=1 writes under /scratch/byrong/mpk/perimage_behavior/smoke/2026-10-19_fs2.
#SBATCH --job-name=fs2_pilogits
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-3
#SBATCH --output=/scratch/byrong/mpk/perimage_behavior/logs/2026-10-19_fs2/pilogits_%A_%a.out

set -euo pipefail
trap '[[ ${TMPDIR:-unset} == unset ]] || rm -rf $TMPDIR' EXIT
CODE=${CODE:-/home/byrong/projects/TSN/fusion_stage1_code/c543636}
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$CODE:$CODE/tmp_analysis"
PY=/home/byrong/anaconda3/envs/mpk/bin/python

read -r -a RUNS <<< "${RUN_LIST:-fs2_dvd300_avg_w050_ls_seed0 fs2_dvd300_avg_w050_ls_seed1 fs2_dvd300_avg_w050_ls_seed2 fs2_dvdfix300_avg_w050_ls_seed0}"
RUN=${RUNS[$SLURM_ARRAY_TASK_ID]}
NAME=${RUN}_final
FE=()
if [[ $RUN =~ ^fs2_dvd(fix)?([0-9]+)_ ]]; then FE=(--dvd-age-months "${BASH_REMATCH[2]}"); fi

OUT=/scratch/byrong/mpk/perimage_behavior/results/2026-09-07_geirhos_perimage/logits
if [[ ${SMOKE:-0} == 1 ]]; then OUT=/scratch/byrong/mpk/perimage_behavior/smoke/2026-10-19_fs2; fi
DEST=$OUT/${NAME}_perimage_logits.npz
if [[ -e $DEST && ${OVERWRITE:-0} != 1 ]]; then
  echo "FATAL: $DEST exists; refusing to overwrite" >&2; exit 1
fi
mkdir -p "$OUT"
echo "[fs2_pilogits] task=$SLURM_ARRAY_TASK_ID run=$RUN ${FE[*]} -> $DEST code=$(cat "$CODE/COMMIT") $(hostname) $(date -Is)"
cd "$CODE"
$PY tmp_analysis/perimage_logits.py \
    --model-name "$NAME" --checkpoint "$RUN" --out-dir "$OUT" "${FE[@]}"
echo "[fs2_pilogits] done $RUN $(date -Is)"
