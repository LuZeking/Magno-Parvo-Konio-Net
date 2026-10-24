#!/bin/bash
# The estimator's own bias (PAPER_v3 Supplementary Note S3, Extended Data Fig. 3) on the
# model of record and its control: vp_recovery_realnc.py's zero-truth simulation with the
# fitted feature geometry of fs2_dvd300_avg_w050_ls_seed0 and fs2_dvdfix300_avg_w050_ls_seed0,
# read from the encoding feature cache of slurm/encode_fs2.sh. Seed 0 of each arm, as for the
# BA network (slurm/vp_recovery_realnc.sh). The measured per-ROI noise-ceiling profile of
# both arms equals the script's default (0.408 ... 0.322; NC > 0.1), so --roi-nc is not passed.
#
#   env -u TMPDIR -u SLURM_CPU_BIND -u SLURM_CPU_BIND_LIST -u SLURM_CPU_BIND_TYPE \
#       -u SLURM_CPU_BIND_VERBOSE sbatch slurm/vp_recovery_fs2.sh
#
# 2 arms x 8 subjects = 16 tasks, ONE array. CPU (himalaya numpy backend): main, no GPU.
# --mem from the BA run (26440398: MaxRSS 7.6-8.4 GB, 1.4-1.7 h; same cache size) + ~40%.
#SBATCH --job-name=mpk_vprec_fs2
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=16
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --array=0-15
#SBATCH --output=/scratch/byrong/mpk/encoding/logs/2026-10-24_vprec_fs2/vprec_%A_%a.out

set -euo pipefail
trap '[[ ${TMPDIR:-unset} == unset ]] || rm -rf $TMPDIR' EXIT
export OPENBLAS_NUM_THREADS=16 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
REPO=$SLURM_SUBMIT_DIR
PY=/home/byrong/anaconda3/envs/mpk/bin/python
OUT=/scratch/byrong/mpk/encoding/results/2026-10-24_vprec_fs2
mkdir -p "$OUT"
cd "$REPO"

MODELS=(fs2_dvd300_avg_w050_ls_seed0 fs2_dvdfix300_avg_w050_ls_seed0)
MODEL=${MODELS[$(( SLURM_ARRAY_TASK_ID / 8 ))]}
SUBJ=$(( SLURM_ARRAY_TASK_ID % 8 + 1 ))
echo "[$SLURM_ARRAY_TASK_ID] model=$MODEL subject=$SUBJ $(date -Is) $(hostname) commit=$(git rev-parse --short HEAD)"

PYTHONPATH="$REPO" $PY tmp_analysis/vp_recovery_realnc.py \
    --subject "$SUBJ" \
    --model "$MODEL" \
    --cache-dir /scratch/byrong/mpk/encoding/cache/2026-10-16_fs2 \
    --n-comp 1000 \
    --n-vox 200 \
    --backend numpy \
    --out-dir "$OUT"
echo "[$SLURM_ARRAY_TASK_ID] done $(date -Is)"
