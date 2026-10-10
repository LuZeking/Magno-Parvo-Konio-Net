#!/bin/bash
# Fusion Stage 1 pilots (TSN_FUSION_PROPOSAL_2026-10-09.md §4): 12 arms x
# seed 0 x 50 epochs, one array. Conditions and the shared recipe (AdamW
# 1e-3 / wd 0.05, cosine, EMA 0.9995, aux per-stream CE 0.3) are defined in
# src/training/standalone_trainer.py (STAGE1_RECIPE, fs1_* conditions).
#
#   task  arm  condition
#    0    A0   fs1_bcs_avg                tied average head, new recipe
#    1-3  A1   fs1_bcs_add_w{050,070,085} untied additive, dieted input
#    4-6  A1'  fs1_hsf_add_w{050,070,085} untied additive, parvo_highsf
#    7-8  A2   fs1_hsf_hinge_k3_l{2,4}    coarse-to-fine hinge, parvo_highsf
#    9    A3   fs1_bcs_add_w070_mstem64   magno trunk at 64 px
#   10-11 A4   fs1_{cs,b2cs}_add_w070     diet dose: CS only, sigma 2 + CS
#
# Code: runs from a `git archive` of the submitted commit under
# /scratch/byrong/TSN/fusion_stage1/code/<sha> (CODE), so edits to the
# worktree while tasks wait in the queue cannot change what trains.
#
# Checkpoints go to node-local $TMPDIR, not straight to /scratch: in the smoke
# test (26606696) a 360 MB checkpoint write to /scratch stalled for 85 min on
# one OST (16) a minute after another file on the same OST wrote fine. A
# background rsync pushes the run dir to $OUT every 15 min (a hung push stalls
# only the pusher, never the GPU); the task pulls $OUT/<run> first, so a
# requeued or resubmitted task resumes from the last pushed epoch. At the end:
# final push, a --checksum dry run that must find no difference, else a copy to
# $OUT_FALLBACK; the EXIT trap then removes $TMPDIR.
#
# Resources, from measurement: the 150-epoch runs of slurm/train_array.sh
# (26373625, 26421237) took 19.5 h - 2.1 days on one A5000 streaming the 59 GB
# h5 off Lustre with 8 workers, i.e. 6.5-17 h per 50 epochs; MaxRSS sat at the
# 22G request (page cache in the cgroup) and every task completed, so 22G is
# kept (~fair share of a 95 GB / 4-GPU node). 36 h covers the slowest case
# twice. QOS `standard`.
#
#   env -u TMPDIR sbatch --export=ALL,CODE=/scratch/byrong/TSN/fusion_stage1/code/<sha> \
#       slurm/train_fusion_stage1.sh
#SBATCH --job-name=fs1_train
#SBATCH --partition=agcichy
#SBATCH --qos=standard
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=22G
#SBATCH --time=1-12:00:00
#SBATCH --array=0-11%12
#SBATCH --output=/scratch/byrong/TSN/fusion_stage1/logs/train_%A_%a.out

set -uo pipefail
trap '[[ ${TMPDIR:-unset} == unset ]] || rm -rf $TMPDIR' EXIT
: "${CODE:?set CODE to the archived code directory}"
: "${TMPDIR:?no node-local TMPDIR}"
cd "$CODE"
unset SLURM_CPU_BIND SLURM_CPU_BIND_LIST SLURM_CPU_BIND_TYPE SLURM_CPU_BIND_VERBOSE
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$CODE"

PY=/home/byrong/anaconda3/envs/mpk/bin/python
DKL=/scratch/byrong/VST_data/VST_datasets/miniecoset_256px_dkl_uint8.h5
CC=/scratch/byrong/TSN/fusion_stage1/cache/cue_conflict_dkl_u8_v1.npz
OUT=/scratch/byrong/TSN/fusion_stage1/checkpoints/2026-10-10_pilot
OUT_FALLBACK=/scratch/byrong/TSN/fusion_stage1/checkpoints/2026-10-10_pilot_fallback
LOCAL=$TMPDIR/runs

CONDS=(fs1_bcs_avg
       fs1_bcs_add_w050 fs1_bcs_add_w070 fs1_bcs_add_w085
       fs1_hsf_add_w050 fs1_hsf_add_w070 fs1_hsf_add_w085
       fs1_hsf_hinge_k3_l2 fs1_hsf_hinge_k3_l4
       fs1_bcs_add_w070_mstem64
       fs1_cs_add_w070 fs1_b2cs_add_w070)
COND=${CONDS[$SLURM_ARRAY_TASK_ID]}
RUN=${COND}_seed0

echo "task $SLURM_ARRAY_TASK_ID -> $COND seed 0 on $(hostname) $(date -Is) code $(cat "$CODE/COMMIT")"
nvidia-smi --query-gpu=name --format=csv,noheader
df -h "$TMPDIR" | tail -1
[ -s "$CC" ] || { echo "missing cue-conflict cache $CC"; exit 1; }

mkdir -p "$LOCAL/$RUN" "$OUT/$RUN"
rsync -a --exclude='*.tmp' "$OUT/$RUN/" "$LOCAL/$RUN/" \
    && echo "pulled $(ls "$LOCAL/$RUN" | wc -l) files from $OUT/$RUN"

push() { rsync -a --exclude='*.tmp' "$LOCAL/$RUN/" "$1/$RUN/"; }
( while sleep 900; do push "$OUT" || echo "[push] failed $(date -Is)"; done ) &
PUSHER=$!

$PY -m src.training.standalone_trainer \
    --dataset "$DKL" --condition "$COND" --seed 0 --out-dir "$LOCAL" \
    --epochs 50 --batch-size 256 --num-workers 8 \
    --ckpt-every 10 --resume --cue-conflict-npz "$CC"
rc=$?
kill "$PUSHER" 2>/dev/null
echo "trainer exit $rc $(date -Is); final push"

push "$OUT" || { echo "final push failed"; rc=1; }
diffs=$(rsync -a --dry-run --checksum --itemize-changes --exclude='*.tmp' \
        "$LOCAL/$RUN/" "$OUT/$RUN/" | grep -v '^\.' | grep -v '^$')
if [ -n "$diffs" ]; then
    echo "remote copy differs:"; echo "$diffs" | head
    mkdir -p "$OUT_FALLBACK/$RUN"
    push "$OUT_FALLBACK" && echo "copied to $OUT_FALLBACK/$RUN" || echo "fallback copy failed"
    rc=1
else
    echo "verified $OUT/$RUN ($(ls "$OUT/$RUN" | wc -l) files)"
fi
echo "task $SLURM_ARRAY_TASK_ID finished with status $rc $(date -Is)"
exit $rc
