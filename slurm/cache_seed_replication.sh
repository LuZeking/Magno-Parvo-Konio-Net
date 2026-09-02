#!/bin/bash
# Seed replication of the PR/set-point measurements: feature caches only.
#
#   sbatch slurm/cache_seed_replication.sh
#
# WHY. Every headline number in PAPER_PLAN §47/§49/§50 rests on ONE checkpoint
# per condition. §50 made that load-bearing rather than merely untidy: §49.4
# instructs reporting the two diet-on-luminance arms as "converge to within
# 2%", a gap of 0.037 in the trained magno/parvo PR ratio, and §50 showed the
# level is channel-specific. A 0.037 gap is only interpretable if seed-to-seed
# spread is smaller than 0.037, which no run has ever tested. seeds 1 and 2
# exist on disk for all five training conditions and have never been
# feature-extracted.
#
# --cache-only writes the feature cache and stops before betas and the ridge
# fit (run_encoding.py:284). The PR analysis reads only the cache, so this
# answers the seed question at ~1/6 the cost of an encoding run. It also warms
# the cache that a later encoding array would hit, so staging it first is
# strictly cheaper than going straight to encoding.
#
# RESOURCES. Measured from the last cache-only array (26406434, 32/32
# COMPLETED): MaxRSS 11.7-12.6 GB, elapsed 1:54-2:07. That job requested 120G,
# which was ~10x its peak. 48G here still leaves 4x headroom over the measured
# peak and schedules on every `main` node instead of queueing behind the
# large-memory subset (the §encode_subjects.sh 190G-vs-192080MB lesson).
# No CUDA is used on this path, so per project policy: `main`, no --gres.
# Throttled to 8 because `main` is already carrying ~20 jobs from other
# projects in this account.
#
#SBATCH --job-name=mpk_seedrep
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --array=0-55%8
#SBATCH --output=slurm/logs/seedrep_%A_%a.out

set -euo pipefail
mkdir -p slurm/logs
export OPENBLAS_NUM_THREADS=16 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16

# 7 models x 8 subjects = 56 array elements.
#
# Front-end flags are NOT guessed -- each row was verified against the prov_*
# keys already stamped into the corresponding seed0 cache, because §22 lost a
# subject to a model encoded through the wrong front end:
#   dkl_bcs_seed0_trained            lum_blur=T lum_cs=T  (defaults)
#   dkl_bcs_swap_seed0_trained       + transform_target=parvo
#   dkl_bcs_parvo_highsf_seed0_...   + parvo_highsf=True, transform_target=magno
#   dkl_base_seed0_trained           lum_blur=F lum_cs=F
#
# dkl_base_seed2 is included so the no-diet arm reaches three seeds as well;
# without it the diet/no-diet contrast would be seed-replicated on one side
# only, which is exactly the asymmetry a reviewer would pick up.
RUNS=(
  dkl_bcs_seed1
  dkl_bcs_seed2
  dkl_bcs_swap_seed1
  dkl_bcs_swap_seed2
  dkl_bcs_parvo_highsf_seed1
  dkl_bcs_parvo_highsf_seed2
  dkl_base_seed2
)
FLAGS=(
  ""
  ""
  "--transform-target parvo"
  "--transform-target parvo"
  "--parvo-highsf"
  "--parvo-highsf"
  "--no-lum-blur --no-lum-cs"
)

MI=$(( SLURM_ARRAY_TASK_ID / 8 ))
SUBJECT=$(( SLURM_ARRAY_TASK_ID % 8 + 1 ))
RUN=${RUNS[$MI]}
EXTRA=${FLAGS[$MI]}
MODEL_NAME=${RUN}_trained

PY=/scratch/byrong/anaconda3/envs/mpk/bin/python
REPO=$SLURM_SUBMIT_DIR
OUT=$REPO/results/encoding
CACHE_DIR=${MPK_FEATURE_CACHE:-/scratch/byrong/VST_data/mpk_feature_cache}
CACHE=$CACHE_DIR/${MODEL_NAME}_subj$(printf %02d "$SUBJECT").npz
CKPT=/scratch/byrong/VST_data/mpk_runs/${RUN}/checkpoint_last.pt

echo "[seedrep] task=$SLURM_ARRAY_TASK_ID model=$MODEL_NAME subj=$SUBJECT"
echo "[seedrep] flags='${EXTRA}'"
echo "[seedrep] ckpt=$CKPT"

# Fail loudly rather than silently extracting through a default front end.
if [[ ! -f "$CKPT" ]]; then
    echo "[seedrep] FATAL: checkpoint missing: $CKPT" >&2
    exit 1
fi

cd "$REPO"

# --seed is deliberately left at its default so the NSD train/test split is
# identical across every model in the series. The model seed lives in the
# checkpoint; changing the analysis seed would break the paired comparisons.
$PY -m src.analysis.encoding.run_encoding \
    --subject "$SUBJECT" \
    --model-name "$MODEL_NAME" \
    --out-dir "$OUT" \
    --feature-cache "$CACHE" \
    --checkpoint "$CKPT" \
    --cache-only \
    ${EXTRA}

echo "[seedrep] done $MODEL_NAME subj$SUBJECT"
