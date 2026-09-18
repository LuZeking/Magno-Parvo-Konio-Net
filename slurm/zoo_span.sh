#!/bin/bash
# PLAN_SPAN (docs/PLAN_SPAN_2026-09-18.md, pre-registered 2026-09-18): the per-image evidence the
# subspace gain acts on, the two-moment span model, and a random-direction null, for the 42
# stored zoo networks.
#
#   sbatch slurm/zoo_span.sh                         # all 42
#   ONLY=tv_resnet18,hub_resnet50_sin sbatch slurm/zoo_span.sh   # a test subset
#
# Nothing is extracted or refitted: this reads the cue-conflict and ImageNet-validation features,
# the all-image subspace bases, the accuracy gate and the kappa sweep that slurm/zoo_cc.sh and
# slurm/zoo_kappa.sh already wrote. One CPU job, one python process over all networks.
#SBATCH --job-name=mpk_zoo_span
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/zoo_span_%j.out

set -euo pipefail
mkdir -p slurm/logs
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1

PY=/scratch/byrong/anaconda3/envs/mpk/bin/python
RUNID=${ZOO_RUNID:-2026-09-15_zoo}
FEAT=/scratch/byrong/mpk/generality/features/$RUNID
BEH=/scratch/byrong/mpk/generality/behaviour/$RUNID
cd "$SLURM_SUBMIT_DIR"

PYTHONPATH="$SLURM_SUBMIT_DIR" $PY -m src.analysis.generality.span_model \
    --features-dir "$FEAT" --out-dir "$BEH" ${ONLY:+--only "$ONLY"}
echo "[zoo_span] done"
