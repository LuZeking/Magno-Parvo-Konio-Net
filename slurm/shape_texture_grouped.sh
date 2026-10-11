#!/bin/bash
# Feature-level grouped shape/texture probe: the paper's stream probe under
# its original split and two grouped splits (content photo / texture source
# held out). Pre-registration and results:
#   /scratch/byrong/TSN/fusion_stage1/results/2026-10-11_grouped_probe/README.md
# Tasks 0-7 are the paper's models through the paper's front end (the same
# arguments as slurm/shape_texture.sh and slurm/thesis_ckpt.sh), tasks 8-19
# the Stage-1 runs rebuilt from their own checkpoints and cue-conflict cache.
#
# Pure CPU, so `main` with no --gres. Earlier probe jobs peaked at 17.6 GB
# MaxRSS (26401311, 26399698); 24G is that plus ~25 %.
#
#   env -u TMPDIR sbatch --export=ALL,CODE=/home/byrong/projects/TSN/fusion_stage1_code/<sha> \
#       slurm/shape_texture_grouped.sh
#SBATCH --job-name=mpk_grpprobe
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --array=0-19%10
#SBATCH --output=/scratch/byrong/TSN/fusion_stage1/logs/grpprobe_%A_%a.out

set -euo pipefail
trap '[[ ${TMPDIR:-unset} == unset ]] || rm -rf $TMPDIR' EXIT
: "${CODE:?pass --export=ALL,CODE=<pinned code dir>}"
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$CODE"

PY=/home/byrong/anaconda3/envs/mpk/bin/python
RUNS=/scratch/byrong/VST_data/mpk_runs
THESIS=/scratch/byrong/tsn_checkpoints
S1=/scratch/byrong/TSN/fusion_stage1/checkpoints/2026-10-10_pilot
CC=/scratch/byrong/TSN/fusion_stage1/cache/cue_conflict_dkl_u8_v1.npz
RES=/scratch/byrong/TSN/fusion_stage1/results/2026-10-11_grouped_probe/json
FEAT=/scratch/byrong/TSN/fusion_stage1/features/2026-10-11_grouped_probe

# name | arguments
CONFIG=(
  "untrained_blur_cs|"
  "dkl_bcs_seed0_trained|--checkpoint $RUNS/dkl_bcs_seed0/checkpoint_last.pt"
  "dkl_bcs_seed1_trained|--checkpoint $RUNS/dkl_bcs_seed1/checkpoint_last.pt"
  "dkl_bcs_seed2_trained|--checkpoint $RUNS/dkl_bcs_seed2/checkpoint_last.pt"
  "dkl_bcs_parvo_highsf_seed0_trained|--checkpoint $RUNS/dkl_bcs_parvo_highsf_seed0/checkpoint_last.pt --parvo-highsf"
  "dkl_bcs_parvo_highsf_seed1_trained|--checkpoint $RUNS/dkl_bcs_parvo_highsf_seed1/checkpoint_last.pt --parvo-highsf"
  "dkl_bcs_parvo_highsf_seed2_trained|--checkpoint $RUNS/dkl_bcs_parvo_highsf_seed2/checkpoint_last.pt --parvo-highsf"
  "dkl_bcs_thesis_trained|--checkpoint $THESIS/dkl_blur_cs_dvd_trans/epoch=147-step=152292.ckpt --stream1-weight 0.5"
)
for r in fs1_bcs_avg fs1_bcs_add_w050 fs1_bcs_add_w070 fs1_bcs_add_w085 \
         fs1_hsf_add_w050 fs1_hsf_add_w070 fs1_hsf_add_w085 \
         fs1_hsf_hinge_k3_l2 fs1_hsf_hinge_k3_l4 \
         fs1_bcs_add_w070_mstem64 fs1_cs_add_w070 fs1_b2cs_add_w070; do
  CONFIG+=("${r}_seed0|--stage1-run $S1/${r}_seed0 --cue-conflict-npz $CC")
done

IFS='|' read -r NAME ARGS <<< "${CONFIG[$SLURM_ARRAY_TASK_ID]}"
echo "[$SLURM_ARRAY_TASK_ID] $NAME  args=$ARGS  code=$CODE  host=$(hostname)"
cd "$CODE"
# shellcheck disable=SC2086  # ARGS is a word list on purpose
"$PY" -m src.analysis.behavior.stream_shape_texture \
    --model-name "$NAME" --out "$RES" --save-features "$FEAT/$NAME.npz" $ARGS
