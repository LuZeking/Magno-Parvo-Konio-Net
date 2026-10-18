#!/bin/bash
# Preflight for slurm/encode_fs2.sh and slurm/readout_w_fs2.sh (2026-10-16):
# tmp_analysis/fs2_encoding_preflight.py checks that the encoding front end and
# model loader reproduce what the trainer scored for every fs2 run. Exits 1 on
# any failure, so the two arrays are submitted with --dependency=afterok on it.
#
#   env -u TMPDIR sbatch --export=ALL,CODE=/home/byrong/projects/TSN/fusion_stage1_code/<sha> \
#       slurm/encode_fs2_preflight.sh
#
# Python is the /home env (as slurm/shape_texture_grouped.sh): the first
# submission (26654758, /scratch env) sat silent until its 1.5 h limit, the
# known hang of a torch import page-faulting its CUDA libraries off Lustre.
#SBATCH --job-name=mpk_fs2_pre
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/byrong/mpk/encoding/logs/2026-10-16_fs2/preflight_%j.out

set -euo pipefail
trap '[[ ${TMPDIR:-unset} == unset ]] || rm -rf $TMPDIR' EXIT
: "${CODE:?pass --export=ALL,CODE=<pinned code dir>}"
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$CODE"
cd "$CODE"
echo "[fs2_pre] $(date -Is) $(hostname) $(cat COMMIT)"
/home/byrong/anaconda3/envs/mpk/bin/python tmp_analysis/fs2_encoding_preflight.py
