#!/bin/bash
# Leg (b): behavioural cue-conflict shape bias, with the per-stream logit
# decomposition (src/analysis/behavior/cue_conflict.py).
#
# This leg has never been run on any model.  "Their integration yields
# human-like shape-biased behaviour" is in the working claim and, until this
# array, was measured nowhere -- legs (a) and (c) both have numbers, (b) had
# none.  The module was written before the checkpoints existed and then never
# invoked.
#
# Model set mirrors the §17 2x2 so leg (b) can be read against leg (a)/(c) on
# the same conditions: untrained x2 (both front ends), dkl_bcs x3, dkl_base x3,
# dkl_bcs_swap x2 (converged seeds only), thesis x2.
# rgb_tsn is excluded: it is a 3-channel RGB two-stream model and this entry
# point builds the 1-channel DKL front end only.
#
# Pure CPU (1280 images, two ResNet18 streams, bootstrap over items).
#SBATCH --job-name=mpk_cueconf
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-11
#SBATCH --output=slurm/logs/cueconf_%A_%a.out

set -euo pipefail
mkdir -p slurm/logs
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1
REPO=$SLURM_SUBMIT_DIR
PY=/scratch/byrong/anaconda3/envs/mpk/bin/python
RUNS=/scratch/byrong/VST_data/mpk_runs
TCK=/scratch/byrong/tsn_checkpoints
cd "$REPO"

# name | checkpoint (- = untrained) | extra args
CONFIG=(
  "untrained_blur_cs|-|"
  "untrained_noblur|-|--no-lum-blur --no-lum-cs"
  "dkl_bcs_seed0|$RUNS/dkl_bcs_seed0/checkpoint_last.pt|"
  "dkl_bcs_seed1|$RUNS/dkl_bcs_seed1/checkpoint_last.pt|"
  "dkl_bcs_seed2|$RUNS/dkl_bcs_seed2/checkpoint_last.pt|"
  "dkl_base_seed0|$RUNS/dkl_base_seed0/checkpoint_last.pt|--no-lum-blur --no-lum-cs"
  "dkl_base_seed1|$RUNS/dkl_base_seed1/checkpoint_last.pt|--no-lum-blur --no-lum-cs"
  "dkl_base_seed2|$RUNS/dkl_base_seed2/checkpoint_last.pt|--no-lum-blur --no-lum-cs"
  "dkl_bcs_swap_seed0|$RUNS/dkl_bcs_swap_seed0/checkpoint_last.pt|--transform-target parvo"
  "dkl_bcs_swap_seed1|$RUNS/dkl_bcs_swap_seed1/checkpoint_last.pt|--transform-target parvo"
  "thesis_dvd_trans|$TCK/dkl_blur_cs_dvd_trans/epoch=147-step=152292.ckpt|--stream1-weight 0.5"
  "thesis_s1_90|$TCK/dkl_blur_cs_s1_90/epoch=147-step=152292.ckpt|--stream1-weight 0.9"
)

IFS='|' read -r NAME CK EXTRA <<< "${CONFIG[$SLURM_ARRAY_TASK_ID]}"
CK_ARG=""; [[ "$CK" != "-" ]] && CK_ARG="--checkpoint $CK"
echo "[$SLURM_ARRAY_TASK_ID] $NAME ckpt=$CK extra=$EXTRA on $(hostname)"

$PY -m src.analysis.behavior.cue_conflict \
    --model-name "$NAME" --out-dir "$REPO/results/cue_conflict" \
    $CK_ARG $EXTRA
