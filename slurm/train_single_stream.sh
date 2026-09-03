#!/bin/bash
# The single-stream control: what does splitting the input across two streams
# actually buy?  3 seeds of `dkl_bcs_single`.
#
#   sbatch slurm/train_single_stream.sh
#
# WHY THIS ARM. The paper's leg (a) says biologically constrained magno/parvo
# input channels induce specialised streams. Existing controls vary the input
# (dkl_base: no diet; dkl_bcs_swap: diet on the other channel;
# dkl_bcs_parvo_highsf: corrected parvo) or the colour space (rgb_tsn: two
# streams, no M/P split). NOTHING asks whether the two-stream architecture is
# needed at all.
#
# `dkl_bcs_single` feeds ONE trunk both DKL channels stacked, with the diet on
# luminance exactly as in dkl_bcs. Identical input information, identical
# diet, identical classifier and hyperparameters -- only the architectural
# split differs. It uses the TwoStreamNetwork baseline=True path that already
# existed in src/models/pytorch_models.py (stream2_block1 = None, x_2=None
# forward), wired to the trainer in this commit.
#
# WHAT IT CAN AND CANNOT TEST. It has no streams, so it cannot be measured on
# the magno/parvo PR instrument or on stream-wise variance partitioning -- do
# not attempt either. What it can be measured on:
#   * top-1 accuracy (does the split cost or buy task performance?)
#   * Geirhos cue-conflict shape bias through the classifier -- the important
#     one, because §54 found the two-stream INTEGRATED readout falls to 0.675
#     while its magno stream alone reaches 0.878. If one stream with the diet
#     matches 0.878, the split is HURTING leg (b) and that must be reported.
#   * whole-model encoding: cortical variance explained without stream-wise
#     partitioning.
#
# THE CAPACITY CONFOUND, stated up front. One ResNet18 trunk against two, so
# this arm has roughly half the trunk parameters (exact counts printed by
# tmp_analysis/smoke_single_stream.py). The confound is ASYMMETRIC and that is
# why one arm is worth running before any width-matched variant: if the
# single stream WINS or TIES, less capacity makes the conclusion stronger and
# the confound is irrelevant. Only if it LOSES is a param-matched control
# (resnet34 trunk, ~21.3M vs ~22.4M for two resnet18 trunks) required before
# concluding anything, and that run should be submitted then, not now.
#
# 3 SEEDS, NOT 1. §58 measured a 0.904 seed range on this condition's
# two-stream sibling and §60.6 states the rule: a diet-vs-control contrast
# replicated on one side only is the asymmetry a reviewer picks up first.
# dkl_bcs has 3 seeds, so this does too.
#
# RESOURCES. Copied from slurm/train_array.sh, which measured 130-200 img/s
# per job packed 4-to-a-node off Lustre (--cache-ram deliberately NOT used;
# see that script's note on node throughput) -> ~55-85 h for 150 epochs.
# --time is set well above that. CUDA is genuinely used here, so unlike the
# cache-only arrays this one takes a GPU. QOS `standard` (cap gres/gpu=16);
# `prio` caps at 8 and is kept for latency-critical work, per project policy.
# Checked before submitting: no QOSMaxGRESPerUser in the pending queue.
#
#SBATCH --job-name=mpk_single
#SBATCH --partition=agcichy
#SBATCH --qos=standard
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=22G
#SBATCH --time=4-00:00:00
#SBATCH --array=0-2
#SBATCH --output=slurm/logs/single_%A_%a.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1

PY=/scratch/byrong/anaconda3/envs/mpk/bin/python
DKL=/scratch/byrong/VST_data/VST_datasets/miniecoset_256px_dkl_uint8.h5
OUT=/scratch/byrong/VST_data/mpk_runs

SEED=$SLURM_ARRAY_TASK_ID

echo "task $SLURM_ARRAY_TASK_ID -> condition=dkl_bcs_single seed=$SEED on $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader

# --resume makes an interrupted multi-day run continue from checkpoint_last.pt
# (weights AND optimizer state) instead of restarting.
# --ckpt-every 25 matches every other run, so the epoch grid that §61/§62 used
# for the epoch-stability audit exists for this arm too. §62.5 showed the
# no-diet arm's representation was still migrating at epoch 150, so this arm
# must be auditable on the same grid rather than only at its endpoint.
$PY -m src.training.standalone_trainer \
    --dataset "$DKL" --condition dkl_bcs_single --seed "$SEED" --out-dir "$OUT" \
    --epochs 150 --batch-size 256 --num-workers 8 \
    --ckpt-every 25 --resume

echo "task $SLURM_ARRAY_TASK_ID finished with status $?"
