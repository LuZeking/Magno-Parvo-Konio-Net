#!/bin/bash
# The PARAMETER-MATCHED single-stream control: 3 seeds of `dkl_bcs_single34`.
#
#   sbatch slurm/train_single_stream34.sh
#
# WHY NOW. `slurm/train_single_stream.sh` says the param-matched arm is needed
# "only if this arm LOSES ... and that run should be submitted then, not now."
# It lost. §72.1 recorded outcome B -- the split buys shape bias -- and §64.5's
# B branch pre-registered a `resnet34` arm as required in exactly that case.
# §72.6 then recorded it as "3 seeds x ~26 h and has not been submitted; that
# is a decision, not an oversight." This is that decision reversed: the arm is
# triggered by the letter of a pre-registration, and an NHB/NMI referee reading
# §64.5 will find the branch and ask for it.
#
# WHAT IT DECIDES. The two-stream model's integrated shape bias is 0.6749; the
# half-capacity single stream beat it. Two readings survive that: the split
# buys shape bias (leg (b)'s architectural claim), or twice the trunk
# parameters do. This arm holds parameters fixed and varies only the split.
#   * if resnet34-single still beats 0.6749 -> the effect is CAPACITY, and
#     leg (b) must stop attributing it to the two-stream architecture;
#   * if it does not -> the split is doing the work, and the 0.0875 gap stands
#     with the confound retired rather than disclosed.
# Either way §64.5's B is resolved instead of carried as a caveat.
#
# PARAMETERS ARE ACTUALLY MATCHED -- measured, not assumed (§97):
#   two ResNet18 trunks (dkl_bcs)      22,397,936
#   one ResNet34 trunk  (this arm)     21,338,992   = 95.3%, 4.7% mismatch
#   one ResNet18 trunk  (the arm that  11,230,832   = 50.1%
#                        triggered it)
# §72.6's "~21.3M vs ~22.4M" is confirmed to three significant figures.
#
# ARCHITECTURE IS BOUND TO THE CONDITION, NOT PASSED AS A FLAG. The run
# directory is `{condition}_seed{n}` with no architecture in it, so
# `--condition dkl_bcs_single --like-resnet resnet34` would write a ResNet34
# run into `dkl_bcs_single_seed0`, overwrite the existing ResNet18 checkpoints
# and let --resume load them. `dkl_bcs_single34` carries `like_resnet` in
# CONDITIONS instead, so the name and the architecture cannot diverge. §87 lost
# a whole arm of a 16-task array to precisely this collision; the fix is to
# make the two inseparable rather than to remember the flag.
#
# ANALYSIS PROTOCOL is fixed by §64.6 and is not renegotiable after seeing the
# numbers: assert 150 epochs and report sd(val_top1) over the last 20 plus
# best-epoch as well as last-epoch accuracy; three seeds both sides; report
# complete separation and the raw difference with its range, NOT a p-value
# (3v3 floors at p=0.10); report the cue-conflict decision rate alongside the
# shape bias, because a bias over a small decided subset is not comparable to
# one over a large subset; no stream statistics -- this arm has no streams.
#
# RESOURCES. `mpk_single` (26421237, ResNet18) ran 150 epochs in 1d02-03h with
# MaxRSS 23,067,728K = 22.0 GiB against a 22G request -- i.e. it sat exactly at
# its limit with no headroom. ResNet34 is deeper, so --mem is sized from that
# measurement plus headroom: 32G (measured + ~45%), well inside the project's
# 64G hard cap. Runtime should be ~1.8x the ResNet18 arm's FLOPs, so ~2 days;
# --time=4-00:00:00 leaves margin for a slow node. CUDA is genuinely used, so
# this takes a GPU. QOS `standard` (cap gres/gpu=16), `prio` kept for
# latency-critical work per project policy; checked before submitting: no
# QOSMaxGRESPerUser in the pending queue, 12 GPU jobs running.
#
#SBATCH --job-name=mpk_single34
#SBATCH --partition=agcichy
#SBATCH --qos=standard
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4-00:00:00
#SBATCH --array=0-2
#SBATCH --output=slurm/logs/single34_%A_%a.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1

PY=/scratch/byrong/anaconda3/envs/mpk/bin/python
DKL=/scratch/byrong/VST_data/VST_datasets/miniecoset_256px_dkl_uint8.h5
OUT=/scratch/byrong/VST_data/mpk_runs

SEED=$SLURM_ARRAY_TASK_ID

echo "task $SLURM_ARRAY_TASK_ID -> condition=dkl_bcs_single34 seed=$SEED on $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader

# No --like-resnet here on purpose: the condition supplies resnet34, and the
# trainer prints the resolved architecture and parameter count at startup and
# records both in config.json (`like_resnet_resolved`, `n_trunk_params`), so
# the log and the artefact both state what was actually built.
#
# --resume continues from checkpoint_last.pt (weights AND optimizer state)
# after a node failure or walltime kill. --ckpt-every 25 matches every other
# run so the §61/§62 epoch-stability grid exists for this arm too.
$PY -m src.training.standalone_trainer \
    --dataset "$DKL" --condition dkl_bcs_single34 --seed "$SEED" --out-dir "$OUT" \
    --epochs 150 --batch-size 256 --num-workers 8 \
    --ckpt-every 25 --resume

echo "task $SLURM_ARRAY_TASK_ID finished with status $?"
