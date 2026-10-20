#!/bin/bash
# NSD encoding of the Stage-2 fusion TSNs (2026-10-16): PAPER_v2 Results 2.3 /
# Fig. 3 on the candidate main TSN fs2_dvd300_avg_w050_ls (3 seeds), its
# no-trajectory control fs2_dvdfix300_avg_w050_ls (seed 0) and the current main
# TSN fs2_bcs_avg_w050_ls (3 seeds). Same instrument as the paper's controls
# (slurm/encode_cc_controls.sh: --ridge-batched --cue-conflict).
# Front end = the run's own config: the DVD arms get --dvd-age-months 300
# (their dvd_end_months), the bcs arm the fixed blur + CS48;
# run_encoding.check_frontend_matches_checkpoint refuses any mismatch.
# checkpoint_final.pt holds the EMA network, the one every logged metric scores.
#
#   E=--export=ALL,CODE=/home/byrong/projects/TSN/fusion_stage1_code/<sha>
#   PRE=$(env -u TMPDIR sbatch --parsable $E slurm/encode_fs2_preflight.sh)
#   env -u TMPDIR sbatch $E --dependency=afterok:$PRE slurm/encode_fs2.sh
# Python = the /home env (see slurm/encode_fs2_preflight.sh).
#
# 7 models x 8 subjects = 56 tasks; CPU (himalaya on torch CPU). --mem from the
# batched fits with in-job features (konioseeds 26496697, MaxRSS 21-34.7 GB)
# plus 25%; those took 3-8 h per subject.
#SBATCH --job-name=mpk_enc_fs2
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=44G
#SBATCH --time=14:00:00
#SBATCH --array=0-55%24
#SBATCH --output=/scratch/byrong/mpk/encoding/logs/2026-10-16_fs2/enc_%A_%a.out

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
OUT=/scratch/byrong/mpk/encoding/results/2026-10-16_fs2
CACHE=/scratch/byrong/mpk/encoding/cache/2026-10-16_fs2/${RUN}_subj$(printf %02d "$SUBJECT").npz

echo "[enc_fs2] task=$TASK run=$RUN subj=$SUBJECT frontend='${FRONTEND:-blur+CS48}' $(date -Is) $(hostname) code=$(cat "$CODE/COMMIT")"
[[ -f "$CKPT" ]] || { echo "FATAL: checkpoint missing: $CKPT" >&2; exit 1; }
mkdir -p "$OUT"
cd "$CODE"
$PY -m src.analysis.encoding.run_encoding \
    --subject "$SUBJECT" --model-name "$RUN" --out-dir "$OUT" \
    --feature-cache "$CACHE" --checkpoint "$CKPT" \
    --cue-conflict --ridge-batched $FRONTEND
echo "[enc_fs2] done $RUN subj$SUBJECT $(date -Is)"
