#!/bin/bash
# Human side of the per-image cue-conflict comparison + its split-half ceiling.
#
# Single task (N=1, so no array). Heterogeneous with slurm/perimage_logits.sh --
# different script, different resources, no shared work -- so the two are
# submitted separately rather than forced into one array.
# CPU only: `main`, no --gres.
#SBATCH --job-name=mpk_humanpi
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/humanpi_%j.out

set -euo pipefail
mkdir -p slurm/logs
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
REPO=$SLURM_SUBMIT_DIR
PY=/scratch/byrong/anaconda3/envs/mpk/bin/python
OUT=/scratch/byrong/mpk/perimage_behavior/results/2026-09-07_geirhos_perimage
cd "$REPO"
PYTHONPATH="$REPO" $PY tmp_analysis/human_perimage.py --out-dir "$OUT"
