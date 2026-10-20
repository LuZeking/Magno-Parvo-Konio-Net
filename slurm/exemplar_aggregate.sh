#!/bin/bash
# Re-aggregate the per-network exemplar outputs on /scratch into results/figures_v2/exemplar_{zoo,streams}.json
# (tmp_analysis/figs_v2/exemplar_aggregate.py). CPU only; run as a job because the session node's Lustre reads hang.
#
#   env -u TMPDIR -u SLURM_CPU_BIND -u SLURM_CPU_BIND_LIST -u SLURM_CPU_BIND_TYPE -u SLURM_CPU_BIND_VERBOSE sbatch slurm/exemplar_aggregate.sh
#SBATCH --job-name=ex_agg
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/byrong/TSN/fusion_stage1/logs/ex_agg_%j.out

set -euo pipefail
trap '[[ ${TMPDIR:-unset} == unset ]] || rm -rf $TMPDIR' EXIT
REPO=/home/byrong/projects/TSN/Magno-Parvo-Konio-Net/.claude/worktrees/fusion-stage1
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
cd "$REPO"
PYTHONPATH="$REPO" /home/byrong/anaconda3/envs/mpk/bin/python tmp_analysis/figs_v2/exemplar_aggregate.py
