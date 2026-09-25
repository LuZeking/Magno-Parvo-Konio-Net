#!/bin/bash
# Move the spatial-loss pilot runs from the g017 /home staging area to /scratch, then
# remove the staged runs and the staged dataset (CLAUDE.md data rule 2). Chained
# afterok on slurm/train_spatial_pilot.sh, so a failed run keeps its staged copy.
# Copy first, verify per-file sizes and file counts, then delete as a separate step.
#SBATCH --job-name=sl_migrate
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --exclude=c025,c094,c072,c098,c150,c088
#SBATCH --output=slurm/logs/sl_migrate_%j.out

set -euo pipefail
SRC=/home/byrong/_staging/TSN/spatial_pilot/runs
DST=/scratch/byrong/TSN/spatial_pilot/runs
DATA=/home/byrong/_staging/TSN/spatial_pilot_data
mkdir -p "$DST"
# Interrupted torch.save leaves *.tmp files; they are not part of the run.
find "$SRC" -name "*.tmp" -print -delete
rsync -aH --info=stats2 "$SRC/" "$DST/"
manifest() { (cd "$1" && find . -type f -printf "%p %s\n" | sort); }
if ! diff <(manifest "$SRC") <(manifest "$DST") > /dev/null; then
    echo "VERIFY FAILED: manifests differ; staged copy kept"
    diff <(manifest "$SRC") <(manifest "$DST") | head -20
    exit 1
fi
echo "verified $(manifest "$DST" | wc -l) files"
rm -rf "$SRC"
rm -rf "$DATA"
echo "removed staged runs and dataset"
