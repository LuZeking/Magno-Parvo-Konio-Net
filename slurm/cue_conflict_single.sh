#!/bin/bash
# Leg (b) for the SINGLE-STREAM arm (PAPER_PLAN §64), the architectural
# control: one trunk carrying both DKL channels, diet on luminance, against
# the two-stream `dkl_bcs`.
#
# Both checkpoint policies are run, 3 seeds each, because §71.4 forbids
# pairing an accuracy figure with a behavioural figure from a different weight
# file, while §64.4's declared reference (0.6749) was measured at the final
# checkpoint. Running both keeps the comparison policy-consistent instead of
# choosing the policy after seeing the result:
#
#   best saved ckpt -> compare against dkl_bcs full = 0.6547 (results/cue_conflict_bestep)
#   final ckpt      -> compare against dkl_bcs full = 0.6749 (results/cue_conflict, §64.4)
#
# The best epoch is resolved from history.json by the job itself (the §71.6
# precedent), so the §71.1 off-by-one cannot be reintroduced by hand.
#
# Pure CPU (1280 images through ONE resnet18 trunk plus an item bootstrap), so
# per the project Slurm rules this goes to `main` with no --gres.
#SBATCH --job-name=mpk_cc_single
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-5
#SBATCH --output=slurm/logs/cc_single_%A_%a.out

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
NAME=dkl_bcs_single_seed${SEED}
RUNDIR="$RUNS/${NAME}"

[[ -f "$RUNDIR/history.json" ]] || { echo "FATAL: no history.json in $RUNDIR"; exit 1; }

if [[ $POLICY -eq 0 ]]; then
  # checkpoint_NNN.pt holds the weights after history index NNN-1
  # (standalone_trainer.py writes epoch+1). Pick the highest-val_top1 epoch
  # among the checkpoints that actually exist on disk.
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
  OUT="$REPO/results/cue_conflict_single_bestep"
else
  CKPT="$RUNDIR/checkpoint_final.pt"
  TAG="${NAME}_final"
  OUT="$REPO/results/cue_conflict_single"
fi

[[ -f "$CKPT" ]] || { echo "FATAL: resolved checkpoint missing: $CKPT"; exit 1; }
echo "[$SLURM_ARRAY_TASK_ID] $TAG policy=$POLICY ckpt=$CKPT on $(hostname)"

PYTHONPATH=. $PY -m tmp_analysis.cue_conflict_single \
    --model-name "$TAG" \
    --checkpoint "$CKPT" \
    --out-dir "$OUT"
