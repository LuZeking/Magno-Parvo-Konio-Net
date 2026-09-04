#!/bin/bash
# OOD robustness on the BEST SAVED checkpoint, not the final one (PAPER_PLAN
# §71). The published §2.5 numbers used checkpoint_last.pt, which for the diet
# arm means weights up to 0.246 below that seed's own peak — including the
# collapsed dkl_bcs_seed1. The policy is applied to all four cells, not just
# the diet arm, so the comparison stays checkpoint-policy-consistent.
#
# CPU partition, no --gres: resnet18 inference over 230k image-passes does not
# need CUDA, and per the project Slurm rules that goes to `main`.
#SBATCH --job-name=mpk_ood_be
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-11%6
#SBATCH --output=slurm/logs/ood_be_%A_%a.out

set -euo pipefail
mkdir -p slurm/logs
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16

PY=/scratch/byrong/anaconda3/envs/mpk/bin/python
REPO=$SLURM_SUBMIT_DIR
RUNS=/scratch/byrong/VST_data/mpk_runs
cd "$REPO"

CELL=$(( SLURM_ARRAY_TASK_ID / 3 ))
SEED=$(( SLURM_ARRAY_TASK_ID % 3 ))

case $CELL in
  0) NAME=dkl_bcs_seed${SEED};              EXTRA=() ;;
  1) NAME=dkl_base_seed${SEED};             EXTRA=(--no-lum-blur --no-lum-cs) ;;
  2) NAME=dkl_bcs_swap_seed${SEED};         EXTRA=(--transform-target parvo) ;;
  3) NAME=dkl_bcs_parvo_highsf_seed${SEED}; EXTRA=(--parvo-highsf) ;;
esac

RUNDIR="$RUNS/${NAME}"
[[ -f "$RUNDIR/history.json" ]] || { echo "FATAL: no history.json in $RUNDIR"; exit 1; }

# checkpoint_NNN.pt holds the weights after history index NNN-1 (the trainer
# writes epoch+1, standalone_trainer.py:414). Pick the highest-val_top1 epoch
# among the checkpoints that actually exist on disk.
BEST_EP=$(PYTHONPATH=. $PY - "$RUNDIR" <<'PYEOF'
import json, os, sys
d = sys.argv[1]
h = json.load(open(os.path.join(d, "history.json")))
avail = [n for n in (25, 50, 75, 100, 125, 150)
         if os.path.exists(os.path.join(d, f"checkpoint_{n:03d}.pt")) and n - 1 < len(h)]
if not avail:
    sys.exit("FATAL: no numbered checkpoints on disk")
print(max(avail, key=lambda n: h[n - 1]["val_top1"]))
PYEOF
)

CKPT="$RUNDIR/checkpoint_$(printf '%03d' "$BEST_EP").pt"
[[ -f "$CKPT" ]] || { echo "FATAL: resolved checkpoint missing: $CKPT"; exit 1; }
echo "[$NAME] best saved checkpoint = epoch $BEST_EP -> $CKPT"

PYTHONPATH=. $PY tmp_analysis/ood_robustness.py \
    --model-name "${NAME}_ep$(printf '%03d' "$BEST_EP")" \
    --checkpoint "$CKPT" \
    --out-dir "$REPO/results/ood_bestep" \
    "${EXTRA[@]}"
