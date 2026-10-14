#!/bin/bash
# Fusion Stage 1 pilots, tasks 6-11, moved to g017 (B200) on 2026-10-11: from
# 00:05 another user's 3-day array held 14 of the 16 agcichy A5000s, which left
# two GPUs for the six queued tasks of 26608893 (last result ~10-12 evening).
# Same pinned code, conditions, recipe, seed and array indices as
# slurm/train_fusion_stage1.sh; the pending A5000 elements 26608893_[6-11] are
# held and then cancelled once these run.
#
# g017 has no /scratch mount, so (as slurm/train_spatial_pilot.sh) data and
# checkpoints move through the rsync gateway slurm/fusion_stage1_gateway.sh:
#   1. pull the 59 GB h5 and the cue-conflict cache once per node into
#      /localscratch2 (flock-serialised; rsync skips a complete file; size check),
#   2. pull the run dir from /scratch if it exists (resume; --update, so a local
#      copy left by a failed verification is never replaced by an older one),
#   3. train from RAM (--cache-ram forces num_workers 0: same images, labels and
#      recipe as the A5000 tasks' streamed reads; only the GPU differs),
#   4. push the run dir to /scratch every 15 min and at the end; a --checksum dry
#      run that must succeed and list nothing, else a fallback push and the local
#      run dir is kept for a re-push. It is removed only after verification.
# Run dirs and data live under /localscratch2, not $TMPDIR: g017's per-job TMPDIR
# is on /localscratch (93 % full on 10-09), and a run dir must outlive a failed
# task. slurm/fusion_stage1_g017_clean.sh removes the data copy afterwards.
#
# Resources, from measurement: the g017 spatial pilot (26532924: same h5, same
# --cache-ram path, two-stream ResNet-18) peaked at MaxRSS 53.4 GiB on the task
# whose figure was not capped by page cache; +25 % -> 67 GiB, so 72G. It averaged
# 1.7 busy cores (TotalCPU 21 h over 12.4 h), so 8 CPUs. It ran 150 epochs in
# 12.4 h, so 50 epochs should take ~4-5 h; 16 h leaves room.
#
#   env -u TMPDIR sbatch --export=ALL,CODE=/home/byrong/projects/TSN/fusion_stage1_code/<sha> \
#       slurm/train_fusion_stage1_g017.sh
#
# Another list of conditions at seed 0 (as COND_LIST in slurm/train_fusion_stage1.sh):
# COND_LIST is indexed by the array index; OUT_DIR is the run root relative to the
# gateway's fs1 module (= /scratch/byrong/TSN/fusion_stage1), fallback OUT_DIR_fallback.
# 2026-10-13: Stage-2 task 4 (fs2_hsf_avg_w085), queued on agcichy behind ~6 days of
# another user's arrays, moved here with --array=4 and OUT_DIR=fs1/checkpoints/2026-10-12_stage2.
# 2026-10-14: SEED_LIST (e.g. COND_LIST="c c" SEED_LIST="1 2") for further seeds of one condition.
#SBATCH --job-name=fs1_g017
#SBATCH --partition=agcichy
#SBATCH --qos=standard
#SBATCH --nodelist=g017
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=72G
#SBATCH --time=16:00:00
#SBATCH --array=6-11
#SBATCH --output=/home/byrong/_staging/TSN/fusion_stage1/logs/train_g017_%A_%a.out

set -uo pipefail
trap '[[ ${TMPDIR:-unset} == unset ]] || rm -rf $TMPDIR' EXIT
: "${CODE:?set CODE to the archived code directory}"
cd "$CODE"
unset SLURM_CPU_BIND SLURM_CPU_BIND_LIST SLURM_CPU_BIND_TYPE SLURM_CPU_BIND_VERBOSE
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$CODE"

PY=/home/byrong/anaconda3/envs/mpk/bin/python
GW_FILE=/home/byrong/_staging/TSN/fs1_gateway.txt
PWF=/home/byrong/.config/tsn_pilot_rsync/password
H5=miniecoset_256px_dkl_uint8.h5
CCN=cue_conflict_dkl_u8_v1.npz
SHARED=/localscratch2/groupdata/byrong/tsn_fs1
LOCAL=$SHARED/runs
OUT=${OUT_DIR:-fs1/checkpoints/2026-10-10_pilot}
OUT_FALLBACK=${OUT}_fallback

CONDS=(fs1_bcs_avg
       fs1_bcs_add_w050 fs1_bcs_add_w070 fs1_bcs_add_w085
       fs1_hsf_add_w050 fs1_hsf_add_w070 fs1_hsf_add_w085
       fs1_hsf_hinge_k3_l2 fs1_hsf_hinge_k3_l4
       fs1_bcs_add_w070_mstem64
       fs1_cs_add_w070 fs1_b2cs_add_w070)
if [ -n "${COND_LIST:-}" ]; then
    read -r -a LIST <<< "$COND_LIST"
    COND=${LIST[$SLURM_ARRAY_TASK_ID]}
else
    COND=${CONDS[$SLURM_ARRAY_TASK_ID]}
fi
# SEED_LIST (optional) is indexed like COND_LIST; without it every task is seed 0.
SEED=0
if [ -n "${SEED_LIST:-}" ]; then
    read -r -a SEEDS <<< "$SEED_LIST"
    SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
fi
RUN=${COND}_seed${SEED}

echo "task $SLURM_ARRAY_TASK_ID -> $COND seed $SEED on $(hostname) $(date -Is) code $(cat "$CODE/COMMIT")"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

for _ in $(seq 60); do [ -s "$GW_FILE" ] && break; sleep 30; done
[ -s "$GW_FILE" ] || { echo "no gateway file $GW_FILE after 30 min"; exit 1; }
read -r GW_HOST GW_PORT < "$GW_FILE"
R="rsync://pilot@$GW_HOST:$GW_PORT"
RS=(rsync --password-file="$PWF")
rsize() { "${RS[@]}" --list-only "$R/$1" | awk '{gsub(",", "", $2); print $2}'; }
h5_size=$(rsize "data/$H5")
cc_size=$(rsize "fs1/cache/$CCN")
if [ -z "$h5_size" ] || [ -z "$cc_size" ]; then
    echo "gateway $GW_HOST:$GW_PORT unreachable"; exit 1
fi
echo "gateway $GW_HOST:$GW_PORT, dataset $h5_size bytes"

mkdir -p "$LOCAL/$RUN" || { echo "cannot create $LOCAL/$RUN"; exit 1; }
df -h "$SHARED" | tail -1
(
    flock 9
    t0=$(date +%s)
    "${RS[@]}" -a "$R/data/$H5" "$SHARED/$H5" || exit 1
    "${RS[@]}" -a "$R/fs1/cache/$CCN" "$SHARED/$CCN" || exit 1
    echo "data sync $(( $(date +%s) - t0 )) s"
) 9> "$SHARED/.data.lock" || { echo "data pull failed"; exit 1; }
if [ "$(stat -c %s "$SHARED/$H5")" != "$h5_size" ] || [ "$(stat -c %s "$SHARED/$CCN")" != "$cc_size" ]; then
    echo "data size mismatch"; exit 1
fi

"${RS[@]}" -a --update --exclude='*.tmp' "$R/$OUT/$RUN/" "$LOCAL/$RUN/" 2>/dev/null \
    && echo "pulled $(ls "$LOCAL/$RUN" | wc -l) files from /scratch (resume)" \
    || echo "no run dir on /scratch yet"

push() { "${RS[@]}" -a --exclude='*.tmp' "$LOCAL/$RUN/" "$R/$1/$RUN/"; }
( while sleep 900; do push "$OUT" || echo "[push] failed $(date -Is)"; done ) &
PUSHER=$!

$PY -m src.training.standalone_trainer \
    --dataset "$SHARED/$H5" --condition "$COND" --seed "$SEED" --out-dir "$LOCAL" \
    --epochs 50 --batch-size 256 --num-workers 0 --cache-ram \
    --ckpt-every 10 --resume --cue-conflict-npz "$SHARED/$CCN"
rc=$?
# Stop the pusher and any rsync it has in flight. Killing only the subshell left
# an orphaned rsync that landed stale files after the final push (26608893_2).
kids=$(pgrep -P "$PUSHER")
kill "$PUSHER" $kids 2>/dev/null
wait "$PUSHER" 2>/dev/null
for p in $kids; do
    for _ in $(seq 60); do kill -0 "$p" 2>/dev/null || break; sleep 1; done
done
echo "trainer exit $rc $(date -Is); final push"

push "$OUT" || { echo "final push failed"; rc=1; }
check=$("${RS[@]}" -a --dry-run --checksum --itemize-changes --exclude='*.tmp' \
        "$LOCAL/$RUN/" "$R/$OUT/$RUN/") || { echo "verify could not run"; rc=1; }
diffs=$(printf '%s\n' "$check" | grep -v '^\.' | grep -v '^$')
if [ -n "$diffs" ]; then
    echo "remote copy differs:"; echo "$diffs" | head
    push "$OUT_FALLBACK" && echo "copied to /scratch/byrong/TSN/$OUT_FALLBACK/$RUN" \
        || echo "fallback copy failed"
    rc=1
fi
if [ $rc -eq 0 ]; then
    echo "verified /scratch/byrong/TSN/$OUT/$RUN ($(ls "$LOCAL/$RUN" | wc -l) files); removing local copy"
    rm -rf "${LOCAL:?}/$RUN"
else
    echo "keeping local run dir g017:$LOCAL/$RUN for a re-push"
fi
echo "task $SLURM_ARRAY_TASK_ID finished with status $rc $(date -Is)"
exit $rc
