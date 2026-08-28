#!/bin/bash
# Verify --resume before trusting it with 15 three-day runs.
#
# Runs 2 epochs, then re-invokes with the same out-dir and 4 epochs. A
# working resume prints "resumed from epoch 1 -> starting at 2" and reports
# epochs 2 and 3 only. A broken one silently retrains from 0, which would be
# invisible in the final checkpoint but would waste days and reset Adam.
#
#SBATCH --job-name=mpk_resume
#SBATCH --partition=agcichy
#SBATCH --qos=standard
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=22G
#SBATCH --time=00:40:00
#SBATCH --output=slurm/logs/smoke_resume_%j.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1

PY=/scratch/byrong/anaconda3/envs/mpk/bin/python
DKL=/scratch/byrong/VST_data/VST_datasets/miniecoset_256px_dkl_uint8.h5
OUT=$SLURM_SUBMIT_DIR/tmp_analysis/resume_test
rm -rf "$OUT"

# Small steps per epoch so the test is quick; resume logic is per-epoch.
COMMON="--dataset $DKL --condition dkl_bcs --seed 0 --out-dir $OUT \
        --batch-size 32 --num-workers 8 --max-steps 20 --val-batches 2 \
        --ckpt-every 100"

echo "=== first run: 2 epochs ==="
$PY -m src.training.standalone_trainer $COMMON --epochs 2 --resume 2>&1 |
    grep -aE "^\[|resumed|Traceback|Error"

echo "=== second run: same dir, 4 epochs, must resume at epoch 2 ==="
$PY -m src.training.standalone_trainer $COMMON --epochs 4 --resume 2>&1 |
    grep -aE "^\[|resumed|Traceback|Error"

echo "=== history.json epochs (expect 0,1,2,3 with no repeats) ==="
$PY -c "
import json
h = json.load(open('$OUT/dkl_bcs_seed0/history.json'))
print([r['epoch'] for r in h])
print('losses:', [round(r['train_loss'], 3) for r in h])
"
echo "=== optimizer state present in checkpoint_last.pt? ==="
$PY -c "
import torch
ck = torch.load('$OUT/dkl_bcs_seed0/checkpoint_last.pt', map_location='cpu',
                weights_only=False)
print('keys:', sorted(ck.keys()))
print('epoch:', ck['epoch'], '| adam param groups:',
      len(ck['optimizer']['param_groups']),
      '| adam state entries:', len(ck['optimizer']['state']))
"
