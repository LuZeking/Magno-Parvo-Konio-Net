#!/bin/bash
# One-time: global DKL per-channel min/max of the training dataset
# (must match src/data/convert_to_dkl.py pass 1). CPU only.
#
#SBATCH --job-name=mpk_dkl_minmax
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/dkl_minmax_%j.out

set -euo pipefail
mkdir -p slurm/logs
cd "$SLURM_SUBMIT_DIR"
/scratch/byrong/anaconda3/envs/mpk/bin/python -c "
from src.analysis.encoding.dkl_frontend import compute_dkl_channel_minmax
mn, mx = compute_dkl_channel_minmax()
print('min:', mn); print('max:', mx)
"
