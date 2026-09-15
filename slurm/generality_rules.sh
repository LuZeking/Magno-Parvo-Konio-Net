#!/bin/bash
# Generality test, split-rule sensitivity (added 2026-09-15 at a referee's request;
# not pre-specified): the fixed-head gain sweep of unit_factor.py with the
# shape-leaning set S defined as the absolute set d_j > 0 ("positive") and as the
# top decile of d ("decile"), instead of the pre-specified median split.
#
#   sbatch slurm/generality_rules.sh
#
# Reads the stored cue-conflict features of slurm/generality_cc.sh; nothing is
# re-extracted. 18 networks x 2 rules = 36 elements, seconds to a minute each.
#SBATCH --job-name=mpk_gen_rules
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --array=0-35
#SBATCH --output=slurm/logs/gen_rules_%A_%a.out

set -euo pipefail
mkdir -p slurm/logs
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

MODELS=(run:rgb_tsn_seed0 run:rgb_tsn_seed1 run:rgb_tsn_seed2
        run:dkl_bcs_single_seed0 run:dkl_bcs_single_seed1 run:dkl_bcs_single_seed2
        run:dkl_bcs_single34_seed0 run:dkl_bcs_single34_seed1 run:dkl_bcs_single34_seed2
        concat:dkl_bcs_seed0 concat:dkl_bcs_seed1 concat:dkl_bcs_seed2
        tv:resnet50 tv:resnet152 tv:vgg16 tv:vit_b_16 tv:convnext_small tv:convnext_base)
RULES=(positive decile)
MODEL=${MODELS[$((SLURM_ARRAY_TASK_ID % 18))]}
RULE=${RULES[$((SLURM_ARRAY_TASK_ID / 18))]}

PY=/scratch/byrong/anaconda3/envs/mpk/bin/python
RUNID=${GEN_RUNID:-2026-09-14_gen}
FEAT=/scratch/byrong/mpk/generality/features/$RUNID
BEH=/scratch/byrong/mpk/generality/behaviour/$RUNID
TAG=$(echo "$MODEL" | tr ':' '_')

echo "[gen_rules] task=$SLURM_ARRAY_TASK_ID model=$MODEL rule=$RULE tag=$TAG"
cd "$SLURM_SUBMIT_DIR"
$PY -m src.analysis.generality.unit_factor --features "$FEAT/${TAG}_cue_conflict.npz" --out-dir "$BEH" \
    --n-splits 10 --rule "$RULE" --out-suffix "_$RULE"
# accuracy gate under the same rule, where held-out natural-image features exist
# (MiniEcoset validation for our runs, the ILSVRC-2012 subset for torchvision nets)
VAL=""
[[ -f "$FEAT/${TAG}_miniecoset.npz" ]] && VAL="$FEAT/${TAG}_miniecoset.npz"
[[ -f "$FEAT/${TAG}_imagenet_val.npz" ]] && VAL="$FEAT/${TAG}_imagenet_val.npz"
if [[ -n "$VAL" && -f "$BEH/${TAG}_unit_factor_$RULE.npz" ]]; then
    $PY -m src.analysis.generality.gain_accuracy --factor "$BEH/${TAG}_unit_factor_$RULE.npz" \
        --val "$VAL" --out-dir "$BEH" --out-suffix "_$RULE"
fi
echo "[gen_rules] done $MODEL $RULE"
