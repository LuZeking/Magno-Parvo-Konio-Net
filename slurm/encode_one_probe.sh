#!/bin/bash
# Single-subject encoding run with peak-memory reporting, used to size the
# array's --mem before launching 8 of them.
#
# Usage: sbatch slurm/encode_one_probe.sh <subject> <model-name> [args...]
#
#SBATCH --job-name=mpk_enc_probe
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/enc_probe_%j.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export OPENBLAS_NUM_THREADS=16 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16

SUBJECT=${1:?subject required}
MODEL_NAME=${2:?model name required}
shift 2

PY=/scratch/byrong/anaconda3/envs/mpk/bin/python
OUT=$SLURM_SUBMIT_DIR/results/encoding
CACHE=$OUT/feature_cache/${MODEL_NAME}_subj$(printf %02d "$SUBJECT").npz

/usr/bin/time -v $PY -m src.analysis.encoding.run_encoding \
    --subject "$SUBJECT" --model-name "$MODEL_NAME" \
    --out-dir "$OUT" --feature-cache "$CACHE" "$@" 2>&1 |
    grep -vE "random sampling with cv"

echo "=== slurm accounting ==="
sacct -j "$SLURM_JOB_ID" --format=JobID,State,Elapsed,MaxRSS,MaxVMSize
