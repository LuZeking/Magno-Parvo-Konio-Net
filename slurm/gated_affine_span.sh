#!/bin/bash
# Accuracy-gated span of the frozen affine readout sweep (tmp_analysis/gated_affine_span.py)
# for the BA diet best checkpoints (reproduces PAPER_final 2.1's 0.307 [0.261, 0.337])
# and the dvd300 / control final checkpoints. CPU only; reads the cached features.
#
#   env -u TMPDIR sbatch slurm/gated_affine_span.sh
#SBATCH --job-name=gated_span
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/byrong/_staging/TSN/gated_span_%j.out

set -euo pipefail
trap '[[ ${TMPDIR:-unset} == unset ]] || rm -rf $TMPDIR' EXIT
REPO=/home/byrong/projects/TSN/Magno-Parvo-Konio-Net/.claude/worktrees/fusion-stage1
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
cd "$REPO"
PYTHONPATH="$REPO" /home/byrong/anaconda3/envs/mpk/bin/python tmp_analysis/gated_affine_span.py \
  --out results/figures_v2/gated_affine_span.json \
  --tags dkl_bcs_seed0_best dkl_bcs_seed1_best dkl_bcs_seed2_best \
         fs2_dvd300_avg_w050_ls_seed0_final fs2_dvd300_avg_w050_ls_seed1_final fs2_dvd300_avg_w050_ls_seed2_final \
         fs2_dvdfix300_avg_w050_ls_seed0_final
