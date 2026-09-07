#!/bin/bash
# Per-image human correspondence across the whole readout family.
# Single task (N=1, no array). CPU only: `main`, no --gres.
#SBATCH --job-name=mpk_picorr
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/picorr_%j.out

set -euo pipefail
mkdir -p slurm/logs
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1
REPO=$SLURM_SUBMIT_DIR
PY=/scratch/byrong/anaconda3/envs/mpk/bin/python
RUN=/scratch/byrong/mpk/perimage_behavior/results/2026-09-07_geirhos_perimage
cd "$REPO"
PYTHONPATH="$REPO:$REPO/tmp_analysis" $PY tmp_analysis/perimage_correspondence.py \
    --run-dir "$RUN" --n-splits 200 --n-boot 2000 --w-step 0.005 --expect-n 15
