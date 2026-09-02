#!/bin/bash
# Epoch-jitter control for the PR "set point": is the trained magno/parvo PR
# ratio a property of the LEARNED SOLUTION, or of WHICH EPOCH the checkpoint
# happened to be saved at?
#
#   sbatch slurm/cache_epoch_jitter.sh
#
# WHY THIS IS NOW THE CRITICAL CONTROL. Job 26413641 seed-replicated the PR
# ratio and found the diet-on-luminance arm ranges 1.603 / 2.507 / 1.823 over
# three seeds -- a spread of 0.904, i.e. 24x the 0.037 gap that §49.4 calls
# "converge to within 2%". The checkpoint histories then explained it:
#
#   dkl_bcs (diet/lum) is the ONLY condition that has not converged.
#     val_top1 over the last 20 epochs: sd 0.042-0.071
#     every other condition:            sd 0.005-0.010   (~8x tighter)
#     final epoch vs that run's best:   0.527/0.588, 0.347/0.593, 0.460/0.596
#     best epoch lands at 120, 59, 141 -- scattered, not at the end.
#
#   And within diet/lum the PR ratio ranks PERFECTLY INVERSELY with final
#   accuracy (spearman -1.00, n=3): the worse the checkpoint, the higher the
#   ratio. So `checkpoint_last.pt` samples one point of an oscillation, and
#   §49.2's "set point at 1.62" may be reading that sample rather than a
#   learned target.
#
# The three diet/lum seeds nevertheless reach nearly IDENTICAL best accuracy
# (0.5877, 0.5929, 0.5957 -- sd 0.004), which is the tell: what the condition
# can achieve is stable; only where the last checkpoint lands is not. There is
# no checkpoint_best.pt on disk, but there are periodic snapshots at epochs
# 25/50/75/100/125/150, so the epoch axis is measurable at cache-only cost.
#
# WHAT IT DECIDES.
#   If PR at ep100/125/150 within ONE seed moves as much as it does ACROSS
#   seeds, the level is a checkpoint-timing artifact. §49.2/§49.4 must be
#   withdrawn rather than merely re-worded, and every dkl_bcs_seed0 result --
#   including all of leg (c), which rests on that single checkpoint -- needs an
#   epoch-stability disclosure.
#   If PR is flat across epochs but varies across seeds, the variability is a
#   genuine property of the learned solution and is reportable as such.
#
# dkl_base_seed0 is included as the CONTROL: it is converged (sd 0.0065), so
# its PR must be flat across epochs. If it is not, the instrument itself is
# epoch-sensitive and nothing here is about the diet.
#
# dkl_bcs_seed0 @ checkpoint_150 is included as a REPRODUCTION CHECK: it should
# return 1.603, the value already cached from checkpoint_last.pt. If it does
# not, the two checkpoint files are not the same weights and that must be
# resolved before anything else here is believed.
#
# RESOURCES. Measured from job 26413641 (56/56 COMPLETED): elapsed 8.5 min,
# MaxRSS ~12 GB. 48G leaves 4x headroom and schedules on every `main` node.
# No CUDA on the --cache-only path, so per project policy: `main`, no --gres.
#
#SBATCH --job-name=mpk_epjit
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-39%8
#SBATCH --output=slurm/logs/epjit_%A_%a.out

set -euo pipefail
mkdir -p slurm/logs
export OPENBLAS_NUM_THREADS=16 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16

# 5 (run, epoch) cells x 8 subjects = 40 array elements.
RUNS=(
  dkl_bcs_seed0
  dkl_bcs_seed0
  dkl_bcs_seed0
  dkl_base_seed0
  dkl_base_seed0
)
EPOCHS=(100 125 150 100 125)
# Front-end flags verified against the prov_* keys of each run's seed0 cache
# (§22 lost a subject to a model encoded through the wrong front end).
FLAGS=(
  ""
  ""
  ""
  "--no-lum-blur --no-lum-cs"
  "--no-lum-blur --no-lum-cs"
)

MI=$(( SLURM_ARRAY_TASK_ID / 8 ))
SUBJECT=$(( SLURM_ARRAY_TASK_ID % 8 + 1 ))
RUN=${RUNS[$MI]}
EP=${EPOCHS[$MI]}
EXTRA=${FLAGS[$MI]}
MODEL_NAME=${RUN}_ep${EP}_trained

PY=/scratch/byrong/anaconda3/envs/mpk/bin/python
REPO=$SLURM_SUBMIT_DIR
OUT=$REPO/results/encoding
CACHE_DIR=${MPK_FEATURE_CACHE:-/scratch/byrong/VST_data/mpk_feature_cache}
CACHE=$CACHE_DIR/${MODEL_NAME}_subj$(printf %02d "$SUBJECT").npz
CKPT=/scratch/byrong/VST_data/mpk_runs/${RUN}/checkpoint_$(printf %03d "$EP").pt

echo "[epjit] task=$SLURM_ARRAY_TASK_ID model=$MODEL_NAME subj=$SUBJECT"
echo "[epjit] flags='${EXTRA}'"
echo "[epjit] ckpt=$CKPT"

# Fail loudly rather than silently extracting through a default front end.
if [[ ! -f "$CKPT" ]]; then
    echo "[epjit] FATAL: checkpoint missing: $CKPT" >&2
    exit 1
fi

cd "$REPO"

# --seed left at default so the NSD train/test split is identical across the
# whole series; the model seed lives in the checkpoint.
$PY -m src.analysis.encoding.run_encoding \
    --subject "$SUBJECT" \
    --model-name "$MODEL_NAME" \
    --out-dir "$OUT" \
    --feature-cache "$CACHE" \
    --checkpoint "$CKPT" \
    --cache-only \
    ${EXTRA}

echo "[epjit] done $MODEL_NAME subj$SUBJECT"
