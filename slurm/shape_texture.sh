#!/bin/bash
# Stream-level shape/texture representational probe, one array element per
# model. Pure CPU work (1280 images through two ResNet18 streams plus small
# logistic probes), so per project policy this runs on `main` with no --gres.
#
# The model list pairs each trained model with the untrained control that
# shares its front end, because that is the only comparison in which "what
# training added" is separable from "what the input transform did".
#SBATCH --job-name=mpk_shptex
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-18%4
#SBATCH --output=slurm/logs/shptex_%A_%a.out

set -euo pipefail
mkdir -p slurm/logs
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

REPO=$SLURM_SUBMIT_DIR
PY=/scratch/byrong/anaconda3/envs/mpk/bin/python
RUNS=/scratch/byrong/VST_data/mpk_runs
THESIS=/scratch/byrong/tsn_checkpoints
cd "$REPO"

# name | checkpoint (- for untrained) | extra front-end args
# The two thesis checkpoints (Lightning, epoch 147, downloaded 2026-08-29 --
# see /scratch/byrong/tsn_checkpoints/PROVENANCE.md) are the models the BA
# thesis actually reports, so they go first: they answer the trained-dkl_bcs
# question without waiting on our retrain, and they are the reference our
# retrain has to reproduce.
#
# `legacy_asymmetric_mixing` differs between the thesis runs and ours, but
# `_mix_streams` is only reached under `stream_concat`, which is false for
# every condition here -- so the flag is a no-op and the models are directly
# comparable. It would matter only for `dkl_bcs_inter`.
CONFIG=(
  "untrained_blur_cs|-|"
  "untrained_noblur|-|--no-lum-blur --no-lum-cs"
  "dkl_base_seed0_trained|$RUNS/dkl_base_seed0/checkpoint_last.pt|--no-lum-blur --no-lum-cs"
  "dkl_base_seed1_trained|$RUNS/dkl_base_seed1/checkpoint_last.pt|--no-lum-blur --no-lum-cs"
  "dkl_base_seed2_trained|$RUNS/dkl_base_seed2/checkpoint_last.pt|--no-lum-blur --no-lum-cs"
  "dkl_bcs_swap_seed0_trained|$RUNS/dkl_bcs_swap_seed0/checkpoint_last.pt|--transform-target parvo"
  "dkl_bcs_swap_seed1_trained|$RUNS/dkl_bcs_swap_seed1/checkpoint_last.pt|--transform-target parvo"
  "thesis_dvd_trans|$THESIS/dkl_blur_cs_dvd_trans/epoch=147-step=152292.ckpt|"
  "thesis_s1_90|$THESIS/dkl_blur_cs_s1_90/epoch=147-step=152292.ckpt|--stream1-weight 0.9"
  "dkl_bcs_seed0_trained|$RUNS/dkl_bcs_seed0/checkpoint_last.pt|"
  "dkl_bcs_seed1_trained|$RUNS/dkl_bcs_seed1/checkpoint_last.pt|"
  "dkl_bcs_seed2_trained|$RUNS/dkl_bcs_seed2/checkpoint_last.pt|"
  "dkl_bcs_parvo_highsf_seed1_trained|$RUNS/dkl_bcs_parvo_highsf_seed1/checkpoint_last.pt|--parvo-highsf"
  "dkl_bcs_parvo_highsf_seed2_trained|$RUNS/dkl_bcs_parvo_highsf_seed2/checkpoint_last.pt|--parvo-highsf"
  "dkl_bcs_swap_seed2_trained|$RUNS/dkl_bcs_swap_seed2/checkpoint_last.pt|--transform-target parvo"
  "dkl_bcs_parvo_highsf_seed0_trained|$RUNS/dkl_bcs_parvo_highsf_seed0/checkpoint_last.pt|--parvo-highsf"
  "rgb_tsn_seed0_trained|$RUNS/rgb_tsn_seed0/checkpoint_last.pt|--rgb-mode"
  "rgb_tsn_seed1_trained|$RUNS/rgb_tsn_seed1/checkpoint_last.pt|--rgb-mode"
  "rgb_tsn_seed2_trained|$RUNS/rgb_tsn_seed2/checkpoint_last.pt|--rgb-mode"
)

IFS='|' read -r NAME CKPT EXTRA <<< "${CONFIG[$SLURM_ARRAY_TASK_ID]}"
CKPT_ARG=""
[[ "$CKPT" != "-" ]] && CKPT_ARG="--checkpoint $CKPT"

echo "[$SLURM_ARRAY_TASK_ID] $NAME  ckpt=$CKPT  extra=$EXTRA"
$PY -m src.analysis.behavior.stream_shape_texture \
    --model-name "$NAME" \
    --out "$REPO/results/shape_texture" \
    $CKPT_ARG $EXTRA
