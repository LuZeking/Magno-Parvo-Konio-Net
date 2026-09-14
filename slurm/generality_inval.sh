#!/bin/bash
# Generality test, accuracy gate for the ImageNet networks: penultimate features
# of a stratified 10-per-class ILSVRC-2012 validation subset (10,000 images),
# then top-1 under the fixed-head gain sweep.
#
#   sbatch slurm/generality_inval.sh
#SBATCH --job-name=mpk_gen_inval
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --array=0-5
#SBATCH --output=slurm/logs/gen_inval_%A_%a.out

set -euo pipefail
mkdir -p slurm/logs
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export TORCH_HOME=/scratch/byrong/caches/torch

MODELS=(tv:resnet50 tv:resnet152 tv:vgg16 tv:vit_b_16 tv:convnext_small tv:convnext_base)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
PY=/scratch/byrong/anaconda3/envs/mpk/bin/python
RUNID=${GEN_RUNID:-2026-09-14_gen}
FEAT=/scratch/byrong/mpk/generality/features/$RUNID
BEH=/scratch/byrong/mpk/generality/behaviour/$RUNID
TAG=$(echo "$MODEL" | tr ':' '_')

echo "[gen_inval] task=$SLURM_ARRAY_TASK_ID model=$MODEL"
cd "$SLURM_SUBMIT_DIR"
[[ -f "$BEH/${TAG}_unit_factor.npz" ]] || { echo "FATAL: unit factor missing for $TAG" >&2; exit 1; }
if [[ ! -f "$FEAT/${TAG}_imagenet_val.npz" ]]; then
    $PY -m src.analysis.generality.extract --model "$MODEL" --set imagenet_val --out-dir "$FEAT"
fi
$PY -m src.analysis.generality.gain_accuracy --factor "$BEH/${TAG}_unit_factor.npz" \
    --val "$FEAT/${TAG}_imagenet_val.npz" --out-dir "$BEH"
echo "[gen_inval] done $MODEL"
