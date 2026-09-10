#!/bin/bash
# Leg (b) for the PARAMETER-MATCHED single-stream arm (§97), the capacity
# control on §64's architectural control: ONE ResNet34 trunk (21,338,992
# params) against two ResNet18 trunks (~22.4M), same input, same diet, same
# scorer.
#
# Why this exists as a separate script from `cue_conflict_single.sh`: that one
# hardcodes `NAME=dkl_bcs_single_seed${SEED}` and writes to
# `results/cue_conflict_single{,_bestep}`. Reusing it with a different run name
# would put ResNet34 numbers in the ResNet18 output directory under names that
# differ only by a substring -- the §87/§97 collision class a third time.
# Separate script, separate output directories, no shared filename.
#
# The architecture is NOT passed here. `cue_conflict_single.py` reads
# `like_resnet` from the checkpoint's own config (§97 bound it to the
# CONDITION), and its strict load then fails loudly if the resolution is
# wrong -- a wrong architecture cannot yield 0 missing and 0 unexpected keys.
#
# Both checkpoint policies, 3 seeds each, exactly as the ResNet18 arm: §71.4
# forbids pairing an accuracy figure with a behavioural figure from a
# different weight file, and running both keeps the comparison
# policy-consistent instead of choosing the policy after seeing the result.
#
# SUBMIT WITH A DEPENDENCY on the training array, so partial checkpoints can
# never be scored:
#     sbatch --dependency=afterok:26444916 slurm/cue_conflict_single34.sh
#
# Pure CPU (1200 images through one trunk plus an item bootstrap), so per the
# project Slurm rules this goes to `main` with no --gres. 32G is the ResNet18
# arm's proven request; ResNet34 adds ~10M params (~40 MB fp32), far inside it.
#SBATCH --job-name=mpk_cc_single34
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-5
#SBATCH --output=slurm/logs/cc_single34_%A_%a.out

set -euo pipefail
mkdir -p slurm/logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTHONUNBUFFERED=1

PY=/scratch/byrong/anaconda3/envs/mpk/bin/python
REPO=$SLURM_SUBMIT_DIR
RUNS=/scratch/byrong/VST_data/mpk_runs
cd "$REPO"

POLICY=$(( SLURM_ARRAY_TASK_ID / 3 ))   # 0 = best saved, 1 = final
SEED=$(( SLURM_ARRAY_TASK_ID % 3 ))
NAME=dkl_bcs_single34_seed${SEED}
RUNDIR="$RUNS/${NAME}"

[[ -f "$RUNDIR/history.json" ]] || { echo "FATAL: no history.json in $RUNDIR"; exit 1; }

# §64.6 asserts 150 epochs. A short history means the training array did not
# finish this seed, and every downstream number would silently describe a
# different training length.
NEP=$($PY -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$RUNDIR/history.json")
[[ "$NEP" -eq 150 ]] || { echo "FATAL: $NAME has $NEP epochs, expected 150"; exit 1; }

if [[ $POLICY -eq 0 ]]; then
  # checkpoint_NNN.pt holds the weights after history index NNN-1
  # (standalone_trainer.py writes epoch+1). Pick the highest-val_top1 epoch
  # among the checkpoints that actually exist on disk -- §71 recorded a cost
  # figure taken from per-epoch maxima at epochs whose weights were never
  # saved, and only a checkpoint that exists can be re-evaluated.
  BEST_EP=$(PYTHONPATH=. $PY - "$RUNDIR" <<'PYEOF'
import json, os, sys
d = sys.argv[1]
h = json.load(open(os.path.join(d, "history.json")))
avail = [n for n in (25, 50, 75, 100, 125, 150)
         if os.path.exists(os.path.join(d, f"checkpoint_{n:03d}.pt")) and n - 1 < len(h)]
if not avail:
    sys.exit("FATAL: no numbered checkpoints on disk")
best = max(avail, key=lambda n: h[n - 1]["val_top1"])
print(best)
PYEOF
)
  CKPT="$RUNDIR/checkpoint_$(printf '%03d' "$BEST_EP").pt"
  TAG="${NAME}_ep$(printf '%03d' "$BEST_EP")"
  OUT="$REPO/results/cue_conflict_single34_bestep"
else
  CKPT="$RUNDIR/checkpoint_final.pt"
  TAG="${NAME}_final"
  OUT="$REPO/results/cue_conflict_single34"
fi

[[ -f "$CKPT" ]] || { echo "FATAL: resolved checkpoint missing: $CKPT"; exit 1; }
echo "[$SLURM_ARRAY_TASK_ID] $TAG policy=$POLICY ckpt=$CKPT on $(hostname)"

PYTHONPATH=. $PY -m tmp_analysis.cue_conflict_single \
    --model-name "$TAG" \
    --checkpoint "$CKPT" \
    --out-dir "$OUT"
