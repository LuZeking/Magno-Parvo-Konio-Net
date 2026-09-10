#!/bin/bash
#SBATCH --job-name=nl_sweep
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --array=0-5
#SBATCH --output=/home/byrong/projects/TSN/Magno-Parvo-Konio-Net/.claude/worktrees/mpk-dev/slurm/logs/nlsweep_%A_%a.out
# §104 T1 step 2: refit heads at each w on the cached features.
# CPU by design -- a 512-unit MLP over cached features needs no GPU, so this
# goes to main with no GRES per the standing rule.
set -euo pipefail
cd /home/byrong/projects/TSN/Magno-Parvo-Konio-Net/.claude/worktrees/mpk-dev
export PYTHONPATH=.:tmp_analysis
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/mpl_$SLURM_JOB_ID"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
PY=/scratch/byrong/anaconda3/envs/mpk/bin/python
FEAT=/scratch/byrong/TSN/nonlinear_readout/features/2026-09-10
OUT=/scratch/byrong/TSN/nonlinear_readout/results/2026-09-10

TAGS=(dkl_bcs_seed0_best dkl_bcs_seed1_best dkl_bcs_seed2_best \
      dkl_bcs_seed0_final dkl_bcs_seed1_final dkl_bcs_seed2_final)
T=${TAGS[$SLURM_ARRAY_TASK_ID]}
$PY tmp_analysis/nonlinear_readout_sweep.py \
    --features "$FEAT/${T}_features.npz" --out-dir "$OUT"
