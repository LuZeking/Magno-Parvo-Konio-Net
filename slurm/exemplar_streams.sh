#!/bin/bash
# PLAN_EXEMPLAR (docs/PLAN_EXEMPLAR_2026-09-21.md, pre-registered 2026-09-21), array B: re-extract and
# store the stage-pooled stream features of the 20 configurations of slurm/shape_texture.sh (Table 1,
# Fig. 2, ED Table 4), check that the paper's probe reproduces results/shape_texture/*.json exactly, and
# run the exemplar-held-out probes per stream. CPU only; sized as slurm/shape_texture.sh.
#
#   sbatch slurm/exemplar_streams.sh
#SBATCH --job-name=mpk_exstreams
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-19%6
#SBATCH --output=/scratch/byrong/mpk/exemplar/logs/exstreams_%A_%a.out

set -euo pipefail
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1

PY=/home/byrong/anaconda3/envs/mpk/bin/python
RUNS=/scratch/byrong/VST_data/mpk_runs
THESIS=/scratch/byrong/tsn_checkpoints
FEAT=/scratch/byrong/mpk/exemplar/features/2026-09-21_streams
OUT=/scratch/byrong/mpk/exemplar/results/2026-09-21_streams
cd "$SLURM_SUBMIT_DIR"

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
  "untrained_swap|-|--transform-target parvo"
)
IFS='|' read -r NAME CKPT EXTRA <<< "${CONFIG[$SLURM_ARRAY_TASK_ID]}"
CKPT_ARG=""
[[ "$CKPT" != "-" ]] && CKPT_ARG="--checkpoint $CKPT"
echo "[exstreams] task=$SLURM_ARRAY_TASK_ID $NAME ckpt=$CKPT extra=$EXTRA"
PYTHONPATH="$SLURM_SUBMIT_DIR" $PY -m src.analysis.behavior.stream_features \
    --model-name "$NAME" --feat-dir "$FEAT" --out-dir "$OUT" $CKPT_ARG $EXTRA
echo "[exstreams] done $NAME"
