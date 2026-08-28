#!/bin/bash
# End-to-end throughput with bf16 autocast + channels_last, at the worker
# count the profiler showed the h5 loader needs (32). Confirms the isolated
# 670 img/s model figure survives contact with the real dataloader, and that
# every condition still trains through the new autocast path.
#
#SBATCH --job-name=mpk_amp
#SBATCH --partition=agcichy
#SBATCH --qos=standard
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=30
#SBATCH --mem=85G
#SBATCH --time=01:30:00
#SBATCH --output=slurm/logs/train_amp_%j.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1

PY=/scratch/byrong/anaconda3/envs/mpk/bin/python
DKL=/scratch/byrong/VST_data/VST_datasets/miniecoset_256px_dkl_uint8.h5
OUT=$SLURM_SUBMIT_DIR/tmp_analysis/train_amp

nvidia-smi --query-gpu=name --format=csv,noheader
echo "--- node sharing ---"
squeue -w "$SLURMD_NODENAME" -o "%.12i %.10u %.14j" | head

# 200 steps is enough for the epoch-time extrapolation to be stable.
echo "=== throughput: dkl_bcs, amp on, 32 workers ==="
$PY -m src.training.standalone_trainer \
    --dataset "$DKL" --condition dkl_bcs --seed 0 --out-dir "$OUT" \
    --epochs 1 --max-steps 200 --batch-size 256 --num-workers 0 --cache-ram \
    --val-batches 2 --ckpt-every 10000 2>&1 | grep -aE "^\[|cach|Traceback|Error"

echo "=== same, amp off (fp32 reference, still cached) ==="
$PY -m src.training.standalone_trainer \
    --dataset "$DKL" --condition dkl_bcs --seed 0 --out-dir "$OUT"_fp32 \
    --epochs 1 --max-steps 200 --batch-size 256 --num-workers 0 --cache-ram \
    --val-batches 2 --ckpt-every 10000 --no-amp 2>&1 |
    grep -aE "^\[|cach|Traceback|Error"

# Every condition must survive the autocast path, including the ones with
# non-default channel counts and stream mixing.
for COND in rgb_tsn dkl_base dkl_bcs_mheavy dkl_bcs_inter dkl_bcs_swap \
            rgb_tsn_blur dkl_bcs_parvo_highsf; do
  echo "=== $COND ==="
  $PY -m src.training.standalone_trainer \
      --dataset "$DKL" --condition "$COND" --seed 0 --out-dir "$OUT" \
      --epochs 1 --max-steps 3 --batch-size 16 --num-workers 4 \
      --val-batches 1 --ckpt-every 10000 2>&1 |
      grep -aE "^\[|cach|Traceback|Error" | tail -3
done
echo "=== amp probe finished ==="
