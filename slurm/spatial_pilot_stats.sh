#!/bin/bash
# Spatial-loss pilot readout (docs/PLAN_TSN_SPATIAL_PILOT_2026-09-25.md §4-§8): gates, per-subject
# DV/LV verdicts, A1 - A0, T1/T2 topography, descriptive -> results/finzi_1to1/SPATIAL_PILOT_*.
# CPU only; run after the probe (slurm/spatial_pilot_probe.sh) and map (slurm/finzi_1to1_map.sh).
#SBATCH --job-name=sl_stats
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --exclude=c025,c094,c072,c098,c150,c088
#SBATCH --output=slurm/logs/sl_stats_%j.out

set -euo pipefail
unset SLURM_CPU_BIND SLURM_CPU_BIND_LIST SLURM_CPU_BIND_TYPE SLURM_CPU_BIND_VERBOSE
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl_$SLURM_JOB_ID"
PY=/home/byrong/anaconda3/envs/mpk/bin/python
cd "$SLURM_SUBMIT_DIR"
echo "[sl_stats] host=$(hostname) start=$(date -Is)"
$PY -m src.analysis.finzi_1to1.spatial_pilot stats --out results/finzi_1to1
echo "[sl_stats] done $(date -Is)"
