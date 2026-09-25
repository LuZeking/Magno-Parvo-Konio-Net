#!/bin/bash
# Spatial-loss pilot probe (docs/PLAN_TSN_SPATIAL_PILOT_2026-09-25.md §3, §6): per-unit
# ACH / LSF / MR on the shared-515 probe stimuli, trunk G-probe and layer4.1 G-SL for
# fused_a0 and fused_a2p5. CPU only. Chained after slurm/spatial_pilot_migrate.sh.
#SBATCH --job-name=sl_probe
#SBATCH --partition=main
#SBATCH --qos=inefficient
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --exclude=c025,c094,c072,c098,c150,c088
#SBATCH --output=slurm/logs/sl_probe_%j.out

set -euo pipefail
unset SLURM_CPU_BIND SLURM_CPU_BIND_LIST SLURM_CPU_BIND_TYPE SLURM_CPU_BIND_VERBOSE
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl_$SLURM_JOB_ID"
PY=/home/byrong/anaconda3/envs/mpk/bin/python
cd "$SLURM_SUBMIT_DIR"
echo "[sl_probe] host=$(hostname) start=$(date -Is)"
$PY -m src.analysis.finzi_1to1.spatial_pilot probe
echo "[sl_probe] done $(date -Is)"
