#!/bin/bash
# Single-subject encoding run with peak-memory + staged timing, used to size
# the array's --mem and --time before launching 8 of them.
#
# Usage: sbatch slurm/encode_one_probe.sh <subject> <model-name> [args...]
#
#SBATCH --job-name=mpk_enc_probe
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/enc_probe_%j.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export OPENBLAS_NUM_THREADS=16 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
export PYTHONUNBUFFERED=1

SUBJECT=${1:?subject required}
MODEL_NAME=${2:?model name required}
shift 2

PY=/scratch/byrong/anaconda3/envs/mpk/bin/python
OUT=$SLURM_SUBMIT_DIR/results/encoding
# NOT $OUT/feature_cache: $OUT is under the repo on /home, which is quota'd,
# and a ~1.2 GB-per-subject cache written there is what filled the quota and
# killed subject 1 of 26383806 with `OSError: [Errno 122] Disk quota
# exceeded` after 34 minutes. encode_subjects.sh was fixed for this; this
# script still had the old path. Scratch has hundreds of TB and the cache is
# regenerable, so it belongs there.
CACHE_DIR=${MPK_FEATURE_CACHE:-/scratch/byrong/VST_data/mpk_feature_cache}
CACHE=$CACHE_DIR/${MODEL_NAME}_subj$(printf %02d "$SUBJECT").npz
RAW=slurm/logs/enc_probe_${SLURM_JOB_ID}_raw.log

# Write the raw stream straight to a file. A previous version piped through
# grep, which buffered; when the job hit its time limit every line was lost
# and the log contained only the Slurm cancellation notice.
/usr/bin/time -v $PY -m src.analysis.encoding.run_encoding \
    --subject "$SUBJECT" --model-name "$MODEL_NAME" \
    --out-dir "$OUT" --feature-cache "$CACHE" "$@" > "$RAW" 2>&1
STATUS=$?

echo "=== progress / timing (raw log: $RAW) ==="
grep -aE "^\[subj|^\[stage|voxels |dropped |Elapsed \(wall|Maximum resident" "$RAW" | tail -40
echo "=== exit status: $STATUS ==="
sacct -j "$SLURM_JOB_ID" --format=JobID,State,Elapsed,MaxRSS,MaxVMSize
