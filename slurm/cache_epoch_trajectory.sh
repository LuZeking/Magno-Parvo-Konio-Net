#!/bin/bash
# Fill in the early half of the PR-vs-epoch trajectory for the two matched
# arms, so the §61 result becomes a curve instead of three points.
#
#   sbatch slurm/cache_epoch_trajectory.sh
#
# WHY. Job 26416534 measured epochs 100/125/150 and found a dissociation that
# was not predicted by anything in the plan:
#
#   arm        val_top1 spread (ep100-149)   PR spread (ep100-150)
#   dkl_bcs            0.2302                       0.1003
#   dkl_base           0.0441                       0.5580
#
# The two are INVERTED. The diet arm has unstable accuracy and a stable
# representation; the no-diet arm has converged accuracy and a representation
# still migrating -- 1.5046 -> 1.0185 -> 0.9466, i.e. from clearly
# magno-dominant down to parity, over epochs where its accuracy moves 0.04.
#
# THE HYPOTHESIS THIS TESTS. Both arms start magno-dominant at their front
# ends (no diet 1.634, diet 1.402 untrained). If the no-diet curve descends
# monotonically from there while the diet curve stays flat, then the diet does
# NOT install magno dominance -- it PREVENTS ITS EROSION during training. That
# is a mechanism claim, and it would replace the set-point account that the
# seed replication (§58) killed. It currently rests on three points from one
# seed per arm, which is not enough to assert a trajectory shape.
#
# Checkpoints at epochs 25/50/75 already exist for both runs, so the first
# half of both curves costs only cache-only extraction. Combined with the
# existing 100/125/150 caches this yields 6-point trajectories.
#
# WHAT WOULD FALSIFY IT. If dkl_base is FLAT from 25 to 100 and only drops
# after 100, the "erosion during training" reading is wrong and the drop is
# something specific to late training (LR schedule, say) that applies to both
# arms and merely shows up in one. If dkl_bcs is NOT flat over 25-75, then its
# apparent epoch-stability at 100-150 is a late-training coincidence and the
# dissociation is not real.
#
# SCOPE DISCLOSURE that travels with any result: one seed per arm. Seed
# generality of the trajectory SHAPE is untested and must not be claimed --
# §58 showed the trained LEVEL varies substantially across seeds on the diet
# arm (range 0.904), so the shape may too.
#
# RESOURCES. Measured from jobs 26413641 (56/56) and 26416534 (40/40): elapsed
# ~8.5 min, MaxRSS ~12 GB. No CUDA on the --cache-only path, so per project
# policy: `main`, no --gres.
#
#SBATCH --job-name=mpk_eptraj
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-47%8
#SBATCH --output=slurm/logs/eptraj_%A_%a.out

set -euo pipefail
mkdir -p slurm/logs
export OPENBLAS_NUM_THREADS=16 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16

# 6 (run, epoch) cells x 8 subjects = 48 array elements.
RUNS=(
  dkl_bcs_seed0
  dkl_bcs_seed0
  dkl_bcs_seed0
  dkl_base_seed0
  dkl_base_seed0
  dkl_base_seed0
)
EPOCHS=(25 50 75 25 50 75)
# Front-end flags verified against the prov_* keys already stamped into each
# run's existing caches (§22 lost a subject to a wrong front end; the ep100/125
# caches from job 26416534 were confirmed to match the originals exactly).
FLAGS=(
  ""
  ""
  ""
  "--no-lum-blur --no-lum-cs"
  "--no-lum-blur --no-lum-cs"
  "--no-lum-blur --no-lum-cs"
)

MI=$(( SLURM_ARRAY_TASK_ID / 8 ))
SUBJECT=$(( SLURM_ARRAY_TASK_ID % 8 + 1 ))
RUN=${RUNS[$MI]}
EP=${EPOCHS[$MI]}
EXTRA=${FLAGS[$MI]}
MODEL_NAME=${RUN}_ep$(printf %03d "$EP")_trained

PY=/scratch/byrong/anaconda3/envs/mpk/bin/python
REPO=$SLURM_SUBMIT_DIR
OUT=$REPO/results/encoding
CACHE_DIR=${MPK_FEATURE_CACHE:-/scratch/byrong/VST_data/mpk_feature_cache}
CACHE=$CACHE_DIR/${MODEL_NAME}_subj$(printf %02d "$SUBJECT").npz
CKPT=/scratch/byrong/VST_data/mpk_runs/${RUN}/checkpoint_$(printf %03d "$EP").pt

echo "[eptraj] task=$SLURM_ARRAY_TASK_ID model=$MODEL_NAME subj=$SUBJECT"
echo "[eptraj] flags='${EXTRA}'"
echo "[eptraj] ckpt=$CKPT"

if [[ ! -f "$CKPT" ]]; then
    echo "[eptraj] FATAL: checkpoint missing: $CKPT" >&2
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

echo "[eptraj] done $MODEL_NAME subj$SUBJECT"
