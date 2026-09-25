#!/bin/bash
# TDANN magno-likeness probe (docs/PLAN_TDANN_MAGNO_PROBE_2026-09-25.md): build the
# shared-515 probe stimuli and write per-unit response SDs for every probe network.
# CPU only: 9 networks x 5 conditions x 515 images.
#
#   sbatch slurm/finzi_1to1_probe.sh
#SBATCH --job-name=f1to1_probe
#SBATCH --partition=main
#SBATCH --qos=inefficient
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --exclude=c025
#SBATCH --output=slurm/logs/f1to1_probe_%j.out

set -euo pipefail
unset SLURM_CPU_BIND SLURM_CPU_BIND_LIST SLURM_CPU_BIND_TYPE SLURM_CPU_BIND_VERBOSE
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl_$SLURM_JOB_ID"
PY=/home/byrong/anaconda3/envs/mpk/bin/python
cd "$SLURM_SUBMIT_DIR"
echo "[f1to1_probe] host=$(hostname) start=$(date -Is)"
$PY -m src.analysis.finzi_1to1.magno_probe extract
echo "[f1to1_probe] done $(date -Is)"
