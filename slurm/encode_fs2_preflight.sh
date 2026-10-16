#!/bin/bash
# Preflight for slurm/encode_fs2.sh and slurm/readout_w_fs2.sh (2026-10-16):
# tmp_analysis/fs2_encoding_preflight.py checks that the encoding front end and
# model loader reproduce what the trainer scored for every fs2 run. Exits 1 on
# any failure, so the two arrays are submitted with --dependency=afterok on it.
#
#   env -u TMPDIR sbatch slurm/encode_fs2_preflight.sh
#SBATCH --job-name=mpk_fs2_pre
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/byrong/mpk/encoding/logs/2026-10-16_fs2/preflight_%j.out

set -euo pipefail
trap '[[ ${TMPDIR:-unset} == unset ]] || rm -rf $TMPDIR' EXIT
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=$SLURM_SUBMIT_DIR
echo "[fs2_pre] $(date -Is) $(hostname) $(git rev-parse --short HEAD)"
/scratch/byrong/anaconda3/envs/mpk/bin/python tmp_analysis/fs2_encoding_preflight.py
