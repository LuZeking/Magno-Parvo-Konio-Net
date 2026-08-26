#!/bin/bash
# One-time MiniEcoset RGB -> DKL uint8 conversion (CPU).
#
#SBATCH --job-name=mpk_dkl_conv
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=slurm/logs/dkl_conv_%j.out

set -euo pipefail
mkdir -p slurm/logs
cd "$SLURM_SUBMIT_DIR"

/scratch/byrong/anaconda3/envs/mpk/bin/python scripts/convert_miniecoset_dkl.py \
    --out /scratch/byrong/VST_data/VST_datasets/miniecoset_256px_dkl_uint8.h5
