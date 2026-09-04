#!/bin/bash
# Feature caches at the BEST SAVED checkpoint, for the §2.4 PR re-run
# (PAPER_PLAN §71.8). Cache-only: the PR analysis reads nothing else.
#
# WHY. §68 Test C showed the magno/parvo PR ratio tracks checkpoint degradation
# at r = -0.99 across the diet arm's own accuracy span. Every PR number in §2.4
# was computed from caches extracted at checkpoint_last.pt, which for the diet
# arm is up to 0.246 below that seed's own peak. §2.4 is therefore the section
# most exposed by §71, and the only way to settle it is to re-extract.
#
# WHAT IS MISSING, and only that. Best saved checkpoints (§71.1):
#   dkl_bcs_seed0            ep125  -> cache ALREADY EXISTS (epoch trajectory)
#   dkl_bcs_seed1            ep075  -> extract
#   dkl_bcs_seed2            ep050  -> extract
#   dkl_bcs_swap_seed0       ep100  -> extract
#   dkl_bcs_swap_seed1       ep100  -> extract
#   dkl_bcs_swap_seed2       ep150  == checkpoint_last, cache exists
#   dkl_base_seed{0,1,2}     ep150  == checkpoint_last, caches exist
#   dkl_bcs_parvo_highsf_*   ep150  == checkpoint_last, caches exist
# checkpoint_last.pt is rewritten every epoch, so at the end of a 150-epoch run
# it holds the same weights as checkpoint_150.pt. 4 models x 8 subjects = 32.
#
# Caches are named with the epoch, matching the existing
# dkl_bcs_seed0_ep125_trained convention, so nothing published is overwritten
# and each file says which weights produced it.
#
# RESOURCES from array 26413641 (measured): MaxRSS 11.7-12.6 GB, 1:54-2:07.
# No CUDA on this path, so per project policy: `main`, no --gres.
#SBATCH --job-name=mpk_cache_be
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --array=0-31%8
#SBATCH --output=slurm/logs/cache_be_%A_%a.out

set -euo pipefail
mkdir -p slurm/logs
export OPENBLAS_NUM_THREADS=16 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16

RUNS=(dkl_bcs_seed1      dkl_bcs_seed2      dkl_bcs_swap_seed0        dkl_bcs_swap_seed1)
EPS=(075                 050                100                       100)
FLAGS=(""                ""                 "--transform-target parvo" "--transform-target parvo")

MI=$(( SLURM_ARRAY_TASK_ID / 8 ))
SUBJECT=$(( SLURM_ARRAY_TASK_ID % 8 + 1 ))
RUN=${RUNS[$MI]}; EP=${EPS[$MI]}; EXTRA=${FLAGS[$MI]}
MODEL_NAME=${RUN}_ep${EP}_trained

PY=/scratch/byrong/anaconda3/envs/mpk/bin/python
REPO=$SLURM_SUBMIT_DIR
CACHE_DIR=${MPK_FEATURE_CACHE:-/scratch/byrong/VST_data/mpk_feature_cache}
CACHE=$CACHE_DIR/${MODEL_NAME}_subj$(printf %02d "$SUBJECT").npz
CKPT=/scratch/byrong/VST_data/mpk_runs/${RUN}/checkpoint_${EP}.pt

echo "[cache_be] task=$SLURM_ARRAY_TASK_ID model=$MODEL_NAME subj=$SUBJECT flags='${EXTRA}'"
echo "[cache_be] ckpt=$CKPT"

# Fail loudly rather than silently extracting through a default front end, and
# verify the checkpoint really is this run's best saved one (§22, §71.1).
[[ -f "$CKPT" ]] || { echo "[cache_be] FATAL: checkpoint missing: $CKPT" >&2; exit 1; }
PYTHONPATH=. $PY - "$CKPT" "/scratch/byrong/VST_data/mpk_runs/${RUN}" <<'PYEOF'
import json, os, sys
ckpt, d = sys.argv[1], sys.argv[2]
h = json.load(open(os.path.join(d, "history.json")))
avail = [n for n in (25, 50, 75, 100, 125, 150)
         if os.path.exists(os.path.join(d, f"checkpoint_{n:03d}.pt")) and n - 1 < len(h)]
best = max(avail, key=lambda n: h[n - 1]["val_top1"])
want = os.path.join(d, f"checkpoint_{best:03d}.pt")
if os.path.realpath(want) != os.path.realpath(ckpt):
    sys.exit(f"FATAL: {ckpt} is not the best saved checkpoint; that is {want}")
print(f"[cache_be] verified best saved checkpoint = epoch {best} "
      f"(val_top1 {h[best-1]['val_top1']:.4f})")
PYEOF

cd "$REPO"
$PY -m src.analysis.encoding.run_encoding \
    --subject "$SUBJECT" \
    --model-name "$MODEL_NAME" \
    --out-dir "$REPO/results/encoding" \
    --feature-cache "$CACHE" \
    --checkpoint "$CKPT" \
    --cache-only \
    ${EXTRA}

echo "[cache_be] done $MODEL_NAME subj$SUBJECT"
