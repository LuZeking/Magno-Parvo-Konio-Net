#!/bin/bash
# TDANN spatial-loss pilot training (PLAN_TSN_SPATIAL_PILOT_2026-09-25 §1, §10).
# Task 0 = A0 dkl_bcs_fused_a0 (alpha 0), task 1 = A1 dkl_bcs_fused_a2p5 (alpha 2.5).
#
# g017 (B200): the agcichy A5000s are held by another user's ~190-task array. g017 has
# no /scratch, and /home cannot hold the 59 GB dataset (hidden ~400G quota: the first
# launch, 26532868, died with EDQUOT). So data and checkpoints move over the network
# through the rsync gateway (slurm/spatial_pilot_gateway.sh, a `main` job that mounts
# /scratch):
#   1. pull the dataset to g017's local NVMe (one copy per node, serialised by a lock;
#      rsync skips a complete file and verifies each transfer by whole-file checksum),
#   2. pull the run dir from /scratch if it exists (resume),
#   3. train from RAM (--cache-ram), writing to local NVMe,
#   4. push the run dir to /scratch/byrong/TSN/spatial_pilot/runs every 15 min and at the
#      end, then fail the task unless a --checksum dry run finds no difference.
# --mem: smoke test 26532505 peaked at MaxRSS = its 72G request (the cgroup also counts
# the page cache from reading the dataset; 48.2 GiB train + 1.0 GiB val cache, ~6 GiB
# process). 80G adds headroom; g017 is exempt from the 64G cap, fair share ~258G/GPU.
# Smoke speed: ~0.43 s/step -> ~440 s/epoch -> ~18.5 h for 150 epochs.
#SBATCH --job-name=sl_train
#SBATCH --partition=agcichy
#SBATCH --qos=standard
#SBATCH --nodelist=g017
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-1
#SBATCH --output=slurm/logs/sl_train_%A_%a.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

PY=/home/byrong/anaconda3/envs/mpk/bin/python
GW_FILE=/home/byrong/_staging/TSN/sl_gateway.txt
PWF=/home/byrong/.config/tsn_pilot_rsync/password
H5=miniecoset_256px_dkl_uint8.h5
LOCAL=/localscratch/byrong/tsn_spatial_pilot
OUT=$LOCAL/runs
CONDS=(dkl_bcs_fused_a0 dkl_bcs_fused_a2p5)
COND=${CONDS[$SLURM_ARRAY_TASK_ID]}
RUN=${COND}_seed0
echo "task $SLURM_ARRAY_TASK_ID -> $COND on $(hostname) $(date -Is)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

for _ in $(seq 60); do [ -s "$GW_FILE" ] && break; sleep 30; done
read -r GW_HOST GW_PORT < "$GW_FILE"
R="rsync://pilot@$GW_HOST:$GW_PORT"
RS=(rsync --password-file="$PWF")
remote_size=$("${RS[@]}" --list-only "$R/data/$H5" | awk '{gsub(",", "", $2); print $2}')
if [ -z "$remote_size" ]; then echo "gateway $GW_HOST:$GW_PORT unreachable"; exit 1; fi
echo "gateway $GW_HOST:$GW_PORT, dataset $remote_size bytes"

mkdir -p "$LOCAL" "$OUT"
df -h "$LOCAL" | tail -1
(
    flock 9
    t0=$(date +%s)
    "${RS[@]}" -a "$R/data/$H5" "$LOCAL/$H5" || exit 1
    echo "dataset sync $(( $(date +%s) - t0 )) s"
) 9> "$LOCAL/.data.lock" || { echo "dataset pull failed"; exit 1; }
if [ "$(stat -c %s "$LOCAL/$H5")" != "$remote_size" ]; then
    echo "dataset size mismatch"; exit 1
fi

"${RS[@]}" -a "$R/pilot/runs/$RUN/" "$OUT/$RUN/" 2>/dev/null \
    && echo "pulled existing run dir (resume)" || echo "no run dir on /scratch yet"

push() { "${RS[@]}" -a --exclude='*.tmp' "$OUT/$RUN/" "$R/pilot/runs/$RUN/"; }
( while sleep 900; do push || echo "[push] failed $(date -Is)"; done ) &
PUSHER=$!

$PY -m src.training.standalone_trainer --dataset "$LOCAL/$H5" --condition "$COND" \
    --seed 0 --out-dir "$OUT" --epochs 150 --batch-size 256 --num-workers 0 \
    --cache-ram --ckpt-every 25 --resume
rc=$?
kill "$PUSHER" 2>/dev/null

push || { echo "final push failed"; rc=1; }
diffs=$("${RS[@]}" -a --dry-run --checksum --itemize-changes --exclude='*.tmp' \
        "$OUT/$RUN/" "$R/pilot/runs/$RUN/" | grep -v '^\.' | grep -v '^$')
if [ -n "$diffs" ]; then echo "remote copy differs:"; echo "$diffs" | head; rc=1; fi
$PY -c "
import torch
ck = torch.load('$OUT/$RUN/checkpoint_final.pt', map_location='cpu', weights_only=False)
print('final checkpoint ok:', len(ck['state_dict']), 'tensors')" || rc=1
[ $rc -eq 0 ] && echo "run on /scratch verified; removing local run dir" && rm -rf "$OUT/$RUN"
exit $rc
