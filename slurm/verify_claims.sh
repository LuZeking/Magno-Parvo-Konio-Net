#!/bin/bash
# Re-derive every pinned draft number. Single task, CPU only.
#SBATCH --job-name=mpk_verify
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/verify_%j.out
set -euo pipefail
mkdir -p slurm/logs
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
cd "$SLURM_SUBMIT_DIR"
PYTHONPATH="$SLURM_SUBMIT_DIR" /home/byrong/anaconda3/envs/mpk/bin/python \
    tmp_analysis/verify_new_claims.py
