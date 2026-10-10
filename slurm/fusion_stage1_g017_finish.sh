#!/bin/bash
# After the g017 Stage-1 array and its cleanup job (afterany): stop the rsync
# gateway, then move the g017 logs (staged on /home, which g017 can write) to the
# Stage-1 log dir on /scratch: copy, verify file count and checksums, and only
# then remove the /home copies. GW_JOB comes through the environment.
#SBATCH --job-name=fs1_g017_fin
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:30:00
#SBATCH --exclude=c025,c094,c072,c098,c150,c088
#SBATCH --output=/scratch/byrong/TSN/fusion_stage1/logs/g017_finish_%j.out

set -uo pipefail
trap '[[ ${TMPDIR:-unset} == unset ]] || rm -rf $TMPDIR' EXIT
echo "[finish] cancelling gateway ${GW_JOB:?}"
scancel "$GW_JOB"
rm -f /home/byrong/_staging/TSN/fs1_gateway.txt

SRC=/home/byrong/_staging/TSN/fusion_stage1/logs
DST=/scratch/byrong/TSN/fusion_stage1/logs
rsync -aH --info=stats2 "$SRC/" "$DST/" || { echo "[finish] copy failed; /home logs kept"; exit 1; }
check=$(rsync -a --dry-run --checksum --itemize-changes "$SRC/" "$DST/") \
    || { echo "[finish] verify could not run; /home logs kept"; exit 1; }
diffs=$(printf '%s\n' "$check" | grep -v '^\.' | grep -v '^$')
if [ -n "$diffs" ]; then
    echo "[finish] /scratch copy differs; /home logs kept:"; echo "$diffs" | head
    exit 1
fi
n=$(ls -A "$SRC" | wc -l)
echo "[finish] $n log files verified on $DST; removing /home copies"
rm -rf "$SRC"
