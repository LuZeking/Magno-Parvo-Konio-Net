#!/bin/bash
# Core seed x condition training array: 5 conditions x 3 seeds = 15 runs.
#
# Conditions, and why each is in the core set:
#   dkl_bcs              the model the paper is about (M/P split + blur + CSF)
#   dkl_base             DKL channels, no blur/CSF -- separates "chromatic vs
#                        achromatic channel" from "low-pass front end", the
#                        same contrast as the --no-lum-blur encoding control
#   rgb_tsn              two streams, no M/P input split -- isolates the
#                        architecture from the input transformation
#   dkl_bcs_swap         blur+CSF applied to the rg channel instead of lum;
#                        rules out "whichever stream got blurred wins"
#   dkl_bcs_parvo_highsf Track B corrected parvo (high-acuity lum + L-M)
# Deferred until these land: dkl_b, dkl_cs, dkl_bcs_mheavy, dkl_bcs_inter,
# rgb_tsn_blur.
#
# Resources, from measurement rather than guesswork
# (tmp_analysis/profile_{train,opt,aug}.py):
#   The a5000 nodes are 32 CPU / 95 GB / 4 GPUs, so a job that takes one GPU
#   gets ~8 CPUs and ~23 GB. Per-node dataloader throughput off Lustre tops
#   out near 591 img/s no matter how it is divided, while one GPU can consume
#   ~360 img/s of it once bf16+channels_last is on. Packing 4 jobs per node
#   therefore beats running 1 cached job per node (~591 vs ~311 img/s of node
#   throughput), which is why --cache-ram is NOT used here.
#   Expect roughly 130-200 img/s per job -> ~55-85 h for 150 epochs. --time
#   is set well above that; the partition allows 14 days.
#
# QOS: `standard` (cap gres/gpu=16); `prio` caps at 8 and is kept free for
# latency-critical work, per project policy.
#SBATCH --job-name=mpk_train
#SBATCH --partition=agcichy
#SBATCH --qos=standard
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=22G
#SBATCH --time=4-00:00:00
#SBATCH --array=0-14%8
#SBATCH --output=slurm/logs/train_%A_%a.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1

PY=/scratch/byrong/anaconda3/envs/mpk/bin/python
DKL=/scratch/byrong/VST_data/VST_datasets/miniecoset_256px_dkl_uint8.h5
OUT=/scratch/byrong/VST_data/mpk_runs

CONDS=(dkl_bcs dkl_base rgb_tsn dkl_bcs_swap dkl_bcs_parvo_highsf)
SEEDS=(0 1 2)
C=$(( SLURM_ARRAY_TASK_ID / ${#SEEDS[@]} ))
S=$(( SLURM_ARRAY_TASK_ID % ${#SEEDS[@]} ))
COND=${CONDS[$C]}
SEED=${SEEDS[$S]}

echo "task $SLURM_ARRAY_TASK_ID -> condition=$COND seed=$SEED on $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader

# --resume makes an interrupted 3-day run continue from checkpoint_last.pt
# (weights AND optimizer state) instead of restarting.
$PY -m src.training.standalone_trainer \
    --dataset "$DKL" --condition "$COND" --seed "$SEED" --out-dir "$OUT" \
    --epochs 150 --batch-size 256 --num-workers 8 \
    --ckpt-every 25 --resume

echo "task $SLURM_ARRAY_TASK_ID finished with status $?"
