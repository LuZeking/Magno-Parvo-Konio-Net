#!/bin/bash
# Removes the g017 Stage-1 tasks' node-local data copy (59 GB h5 + cue-conflict
# cache) from /localscratch2 once the g017 array has ended (afterany). Run dirs
# are removed by each task after its /scratch copy verifies; any left over belong
# to a failed task and are kept and listed, never deleted here.
#SBATCH --job-name=fs1_g017_clean
#SBATCH --partition=agcichy
#SBATCH --qos=standard
#SBATCH --nodelist=g017
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=/home/byrong/_staging/TSN/fusion_stage1/logs/g017_clean_%j.out

set -uo pipefail
trap '[[ ${TMPDIR:-unset} == unset ]] || rm -rf $TMPDIR' EXIT
S=/localscratch2/groupdata/byrong/tsn_fs1
du -sh "$S" 2>/dev/null
rm -f "$S/miniecoset_256px_dkl_uint8.h5" "$S/cue_conflict_dkl_u8_v1.npz" "$S/.data.lock"
left=$(ls -A "$S/runs" 2>/dev/null)
if [ -z "$left" ]; then
    rm -rf "$S"
    echo "[g017-clean] removed $S $(date -Is)"
else
    echo "[g017-clean] data removed; KEPT unverified run dirs in $S/runs: $left"
fi
