#!/bin/bash
# Front-end check of every paper pipeline on the Stage-2 TSNs (2026-10-19):
# tmp_analysis/fs2_frontend_check.py runs each entry point up to its first
# front-end call and compares the input against the trainer's
# StreamInputBuilder (fs2 runs) or the committed HEAD front end (BA run).
# Exits 1 on any failure. CPU only.
#
#   mkdir -p /scratch/byrong/mpk/encoding/logs/2026-10-19_frontend
#   env -u TMPDIR sbatch slurm/fs2_frontend_check.sh      # from the worktree
#
# Python is the /home env: the /scratch env hangs on torch import.
#SBATCH --job-name=mpk_fs2_fe
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/byrong/mpk/encoding/logs/2026-10-19_frontend/check_%j.out

set -euo pipefail
trap '[[ ${TMPDIR:-unset} == unset ]] || rm -rf $TMPDIR' EXIT
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$SLURM_SUBMIT_DIR"
cd "$SLURM_SUBMIT_DIR"
echo "[fs2_fe] $(date -Is) $(hostname) $(git rev-parse HEAD) TMPDIR=${TMPDIR:-unset}"
/home/byrong/anaconda3/envs/mpk/bin/python tmp_analysis/fs2_frontend_check.py
