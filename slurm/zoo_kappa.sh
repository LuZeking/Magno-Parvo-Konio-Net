#!/bin/bash
# PLAN_KAPPA (docs/PLAN_KAPPA_2026-09-17.md, pre-registered 2026-09-17): the rate and the error
# consistency with the ten observers at every gain w, for the 42 stored zoo networks.
#
#   sbatch slurm/zoo_kappa.sh
#
# Nothing is extracted or refitted: this reads the cue-conflict features, the all-image
# subspace bases and the accuracy gate that slurm/zoo_cc.sh already wrote, and walks the
# 21-point w grid. One CPU job, one python process over all 42 networks (the import chain
# pulls in torch, so a process per network would spend minutes importing and seconds working).
#SBATCH --job-name=mpk_zoo_kappa
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/zoo_kappa_%j.out

set -euo pipefail
mkdir -p slurm/logs
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1

PY=/scratch/byrong/anaconda3/envs/mpk/bin/python
RUNID=${ZOO_RUNID:-2026-09-15_zoo}
FEAT=/scratch/byrong/mpk/generality/features/$RUNID
BEH=/scratch/byrong/mpk/generality/behaviour/$RUNID
cd "$SLURM_SUBMIT_DIR"

PYTHONPATH="$SLURM_SUBMIT_DIR" $PY -m src.analysis.generality.kappa_sweep \
    --features-dir "$FEAT" --out-dir "$BEH"
echo "[zoo_kappa] done"
