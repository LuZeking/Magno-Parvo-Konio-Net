#!/bin/bash
# Cheap validation of every encoding I/O path before launching real fits.
#
#SBATCH --job-name=mpk_preflight
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/preflight_%j.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

/scratch/byrong/anaconda3/envs/mpk/bin/python -m src.analysis.encoding.preflight \
    --subject "${1:-1}"
