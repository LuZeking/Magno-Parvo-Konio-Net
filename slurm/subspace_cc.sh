#!/bin/bash
# PLAN_SUBSPACE D1-D2 (pre-registered 2026-09-15): shape- and texture-predictive
# subspaces of every stored network and the fixed-head subspace gain sweep, then the
# accuracy gate on held-out natural images where those features exist.
#
#   sbatch slurm/subspace_cc.sh
#
# Reads the stored cue-conflict / MiniEcoset / ILSVRC features of the 2026-09-14_gen
# run; writes <tag>_subspace.{json,npz} and <tag>_gain_accuracy_subspace.json to the
# same behaviour directory. Minutes per element.
#SBATCH --job-name=mpk_sub_cc
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --array=0-17
#SBATCH --output=slurm/logs/sub_cc_%A_%a.out

set -euo pipefail
mkdir -p slurm/logs
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

MODELS=(run:rgb_tsn_seed0 run:rgb_tsn_seed1 run:rgb_tsn_seed2
        run:dkl_bcs_single_seed0 run:dkl_bcs_single_seed1 run:dkl_bcs_single_seed2
        run:dkl_bcs_single34_seed0 run:dkl_bcs_single34_seed1 run:dkl_bcs_single34_seed2
        concat:dkl_bcs_seed0 concat:dkl_bcs_seed1 concat:dkl_bcs_seed2
        tv:resnet50 tv:resnet152 tv:vgg16 tv:vit_b_16 tv:convnext_small tv:convnext_base)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

PY=/scratch/byrong/anaconda3/envs/mpk/bin/python
RUNID=${GEN_RUNID:-2026-09-14_gen}
FEAT=/scratch/byrong/mpk/generality/features/$RUNID
BEH=/scratch/byrong/mpk/generality/behaviour/$RUNID
TAG=$(echo "$MODEL" | tr ':' '_')

echo "[sub_cc] task=$SLURM_ARRAY_TASK_ID model=$MODEL tag=$TAG"
cd "$SLURM_SUBMIT_DIR"
$PY -m src.analysis.generality.subspace_factor --features "$FEAT/${TAG}_cue_conflict.npz" --out-dir "$BEH" --n-splits 10
VAL=""
[[ -f "$FEAT/${TAG}_miniecoset.npz" ]] && VAL="$FEAT/${TAG}_miniecoset.npz"
[[ -f "$FEAT/${TAG}_imagenet_val.npz" ]] && VAL="$FEAT/${TAG}_imagenet_val.npz"
if [[ -n "$VAL" ]]; then
    $PY -m src.analysis.generality.gain_accuracy --factor "$BEH/${TAG}_subspace.npz" \
        --val "$VAL" --out-dir "$BEH" --out-suffix "_subspace"
fi
echo "[sub_cc] done $MODEL"
