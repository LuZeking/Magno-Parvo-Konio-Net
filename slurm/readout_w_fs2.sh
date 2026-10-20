#!/bin/bash
# Cortex-optimal readout weight (PAPER_v2 Extended Data Fig. 5 / Table 2) for the
# Stage-2 fusion TSNs (2026-10-16): same models, front ends and guard as
# slurm/encode_fs2.sh; the fit is src/analysis/encoding/readout_w_cortex.py, as
# in slurm/readout_w_cortex.sh. For the fs2 average head the classifier mixes
# the same pooled 512-d trunk outputs, z(w) = w h_M + (1 - w) h_P.
#
#   env -u TMPDIR sbatch $E --dependency=afterok:$PRE slurm/readout_w_fs2.sh
# ($E and $PRE as in slurm/encode_fs2.sh; Python = the /home env.)
#
# 7 models x 8 subjects = 56 tasks; CPU. --mem from readout_w 26476097
# (MaxRSS 17-31 GB) plus 25%; those took 2-3.5 h per subject.
#SBATCH --job-name=mpk_rw_fs2
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH --array=0-55%16
#SBATCH --output=/scratch/byrong/mpk/readout_w/logs/2026-10-16_fs2/rw_%A_%a.out

set -euo pipefail
trap '[[ ${TMPDIR:-unset} == unset ]] || rm -rf $TMPDIR' EXIT
: "${CODE:?pass --export=ALL,CODE=<pinned code dir>}"
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$CODE"

TASK=$SLURM_ARRAY_TASK_ID
SUBJECT=$(( TASK % 8 + 1 ))
GROUP=$(( TASK / 8 ))
case $GROUP in
  0|1|2) RUN=fs2_dvd300_avg_w050_ls_seed$GROUP ; FRONTEND="--dvd-age-months 300" ;;
  3)     RUN=fs2_dvdfix300_avg_w050_ls_seed0   ; FRONTEND="--dvd-age-months 300" ;;
  4|5|6) RUN=fs2_bcs_avg_w050_ls_seed$(( GROUP - 4 )) ; FRONTEND="" ;;
  # 2026-10-20: the control's seeds 1-2 and the channel swap (--array=56-95)
  7|8)   RUN=fs2_dvdfix300_avg_w050_ls_seed$(( GROUP - 6 )) ; FRONTEND="--dvd-age-months 300" ;;
  9|10|11) RUN=fs2_dvd300swap_avg_w050_ls_seed$(( GROUP - 9 )) ; FRONTEND="--dvd-age-months 300 --transform-target parvo" ;;
  *) echo "bad group $GROUP" >&2; exit 1 ;;
esac

CKPT=/scratch/byrong/TSN/fusion_stage1/checkpoints/2026-10-12_stage2/$RUN/checkpoint_final.pt
PY=/home/byrong/anaconda3/envs/mpk/bin/python
OUT=/scratch/byrong/mpk/readout_w/results/2026-10-16_fs2_w21
FEAT=/scratch/byrong/mpk/readout_w/features

echo "[rw_fs2] task=$TASK run=$RUN subj=$SUBJECT frontend='${FRONTEND:-blur+CS48}' $(date -Is) $(hostname) code=$(cat "$CODE/COMMIT")"
[[ -f "$CKPT" ]] || { echo "FATAL: checkpoint missing: $CKPT" >&2; exit 1; }
cd "$CODE"
$PY -m src.analysis.encoding.readout_w_cortex \
    --subject "$SUBJECT" --checkpoint "$CKPT" --model-name "$RUN" \
    --out-dir "$OUT" --feature-dir "$FEAT" $FRONTEND
echo "[rw_fs2] done $RUN subj$SUBJECT $(date -Is)"
