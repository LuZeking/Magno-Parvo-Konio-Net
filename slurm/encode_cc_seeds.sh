#!/bin/bash
# Seed replication of the CORTICAL claim (PAPER_DRAFT §2.3).
#
#   sbatch slurm/encode_cc_seeds.sh
#
# WHY. §2.3 is the paper's human-brain evidence: a per-stream cue-conflict
# dissociation of +1.534 with the diet against +0.110 without (p=.0078, 8/8
# subjects), plus the ventral gradient (hV4-V1v -0.862 vs -0.117). Both rest on
# results/encoding_cc, which contains exactly ONE checkpoint per arm:
# dkl_bcs_seed0_trained and dkl_base_seed0_trained. The subject axis is
# replicated 8-fold; the SEED axis is not replicated at all.
#
# The PR seed replication (job 26413641 -> tmp_analysis/pr_seeds.py) has just
# shown why that matters: on the PR instrument the diet arm moves 1.603 ->
# 2.507 -> 1.823 across nothing but a training seed. The LEVEL of a
# single-checkpoint number on that instrument was not reproducible. Until the
# same check is run here, we do not know whether +1.534 is a condition effect
# or one checkpoint.
#
# Feature caches for all four models already exist (job 26413641 wrote seeds
# 1-2 for every condition), so this array skips extraction and runs the ridge
# fit plus the cue-conflict forward pass only -- the expensive half is already
# paid for.
#
# 4 models x 8 subjects = 32 tasks. Diet and no-diet only: those are the two
# arms in the §2.3 contrast. swap and parvo_highsf have never been run on the
# cortical instrument in any seed and are a separate question, not this one.
#
# CPU per project policy -- himalaya runs the primal solver on torch CPU and no
# CUDA is required. Memory/time mirror slurm/encode_cue_conflict.sh, whose
# measured peak was ~137G.
#SBATCH --job-name=mpk_cc_seeds
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --array=0-31%6
#SBATCH --output=slurm/logs/ccseeds_%A_%a.out

set -euo pipefail
mkdir -p slurm/logs
export OPENBLAS_NUM_THREADS=16 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16

TASK=$SLURM_ARRAY_TASK_ID
SUBJECT=$(( TASK % 8 + 1 ))
GROUP=$(( TASK / 8 ))

# Front-end flags are NOT guessed: they mirror slurm/encode_cue_conflict.sh,
# which is the script that produced the seed-0 cells these will be compared to.
case $GROUP in
  0) RUN=dkl_bcs_seed1  ; FRONTEND="" ;;
  1) RUN=dkl_bcs_seed2  ; FRONTEND="" ;;
  2) RUN=dkl_base_seed1 ; FRONTEND="--no-lum-blur --no-lum-cs" ;;
  3) RUN=dkl_base_seed2 ; FRONTEND="--no-lum-blur --no-lum-cs" ;;
  *) echo "bad group $GROUP" >&2; exit 1 ;;
esac

MODEL_NAME=${RUN}_trained
CKPT=/scratch/byrong/VST_data/mpk_runs/${RUN}/checkpoint_last.pt

PY=/scratch/byrong/anaconda3/envs/mpk/bin/python
REPO=$SLURM_SUBMIT_DIR
OUT=$REPO/results/encoding_cc
CACHE_DIR=${MPK_FEATURE_CACHE:-/scratch/byrong/VST_data/mpk_feature_cache}
CACHE=$CACHE_DIR/${MODEL_NAME}_subj$(printf %02d "$SUBJECT").npz

echo "[ccseeds] task=$TASK model=$MODEL_NAME subj=$SUBJECT frontend='${FRONTEND:-blur+CS}'"

if [[ ! -f "$CKPT" ]]; then
    echo "[ccseeds] FATAL: checkpoint missing: $CKPT" >&2; exit 1
fi
if [[ ! -f "$CACHE" ]]; then
    echo "[ccseeds] FATAL: feature cache missing: $CACHE" >&2; exit 1
fi

cd "$REPO"
$PY -m src.analysis.encoding.run_encoding \
    --subject "$SUBJECT" \
    --model-name "$MODEL_NAME" \
    --out-dir "$OUT" \
    --feature-cache "$CACHE" \
    --checkpoint "$CKPT" \
    --cue-conflict \
    $FRONTEND

echo "[ccseeds] done $MODEL_NAME subj$SUBJECT"
