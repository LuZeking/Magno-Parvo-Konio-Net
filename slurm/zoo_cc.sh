#!/bin/bash
# PLAN_ZOO (docs/PLAN_ZOO_2026-09-15.md, pre-registered 2026-09-15): the field's model zoo
# under the W1 subspace rule and the unit rule -- cue-conflict features, shape/texture
# subspaces, fixed-head gain sweeps, the ILSVRC-subset accuracy gate and the linear-probe
# representational bias, one network per element.
#
#   sbatch slurm/zoo_cc.sh
#
# The six networks of Fig. 6 reuse their stored 2026-09-14_gen features and sweeps
# (symlinked into the zoo run) and only add the probe. Everything else is extracted
# here: 1,200 cue-conflict images + the 10,000-image ILSVRC-2012 validation subset on
# CPU (18-20 min for ResNet-50-sized networks, ~1 h for ViT-L/16), minutes of analysis.
#SBATCH --job-name=mpk_zoo
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --array=0-41%6
#SBATCH --output=slurm/logs/zoo_%A_%a.out

set -euo pipefail
mkdir -p slurm/logs
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export TORCH_HOME=/scratch/byrong/caches/torch

MODELS=(
  tv:resnet50 tv:resnet152 tv:vgg16 tv:vit_b_16 tv:convnext_small tv:convnext_base   # Fig. 6 (stored)
  tv:alexnet tv:vgg11_bn tv:vgg16_bn tv:vgg19_bn tv:googlenet tv:inception_v3
  tv:resnet18 tv:resnet34 tv:resnet101 tv:resnext50_32x4d tv:resnext101_32x8d
  tv:wide_resnet50_2 tv:wide_resnet101_2 tv:densenet121 tv:densenet201
  tv:mobilenet_v2 tv:mobilenet_v3_large tv:shufflenet_v2_x1_0 tv:mnasnet1_0
  tv:efficientnet_b0 tv:efficientnet_b3 tv:efficientnet_v2_s
  tv:regnet_y_8gf tv:regnet_y_16gf tv:convnext_tiny tv:convnext_large
  tv:vit_b_32 tv:vit_l_16 tv:swin_t tv:swin_s tv:swin_b tv:maxvit_t
  tv:resnet50:IMAGENET1K_V2                                                          # recipe variant
  hub:resnet50_sin hub:resnet50_sin_in hub:resnet50_sin_in_ft                        # Geirhos 2019
)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

PY=/scratch/byrong/anaconda3/envs/mpk/bin/python
RUNID=${ZOO_RUNID:-2026-09-15_zoo}
FEAT=/scratch/byrong/mpk/generality/features/$RUNID
BEH=/scratch/byrong/mpk/generality/behaviour/$RUNID
OLDFEAT=/scratch/byrong/mpk/generality/features/2026-09-14_gen
OLDBEH=/scratch/byrong/mpk/generality/behaviour/2026-09-14_gen
TAG=$(echo "$MODEL" | tr ':' '_')
mkdir -p "$FEAT" "$BEH"

echo "[zoo] task=$SLURM_ARRAY_TASK_ID model=$MODEL tag=$TAG"
cd "$SLURM_SUBMIT_DIR"

if [[ -f "$OLDFEAT/${TAG}_cue_conflict.npz" ]]; then
    ln -sfn "$OLDFEAT/${TAG}_cue_conflict.npz" "$FEAT/${TAG}_cue_conflict.npz"
    ln -sfn "$OLDFEAT/${TAG}_imagenet_val.npz" "$FEAT/${TAG}_imagenet_val.npz"
    for f in "$OLDBEH/${TAG}"_*; do ln -sfn "$f" "$BEH/$(basename "$f")"; done
    echo "[zoo] reusing stored 2026-09-14_gen features and sweeps for $TAG"
fi

CC=$FEAT/${TAG}_cue_conflict.npz
VAL=$FEAT/${TAG}_imagenet_val.npz
[[ -f "$CC" ]]  || $PY -m src.analysis.generality.extract --model "$MODEL" --set cue_conflict --out-dir "$FEAT"
[[ -f "$VAL" ]] || $PY -m src.analysis.generality.extract --model "$MODEL" --set imagenet_val --out-dir "$FEAT"
[[ -f "$BEH/${TAG}_subspace.npz" ]]    || $PY -m src.analysis.generality.subspace_factor --features "$CC" --out-dir "$BEH" --n-splits 10
[[ -f "$BEH/${TAG}_unit_factor.npz" ]] || $PY -m src.analysis.generality.unit_factor --features "$CC" --out-dir "$BEH" --n-splits 10
[[ -f "$BEH/${TAG}_gain_accuracy_subspace.json" ]] || $PY -m src.analysis.generality.gain_accuracy --factor "$BEH/${TAG}_subspace.npz" --val "$VAL" --out-dir "$BEH" --out-suffix _subspace
[[ -f "$BEH/${TAG}_gain_accuracy.json" ]]          || $PY -m src.analysis.generality.gain_accuracy --factor "$BEH/${TAG}_unit_factor.npz" --val "$VAL" --out-dir "$BEH"
$PY -m src.analysis.generality.probe_bias --features "$CC" --out-dir "$BEH" --n-splits 10
echo "[zoo] done $MODEL"
