#!/bin/bash
# Generality test, refit (PLAN_GENERALITY D3): MiniEcoset train/val features of
# our nine single-trunk runs, then the lossy-sum sweep with refitted heads.
#
#   sbatch --dependency=afterok:<gen_cc jobid> slurm/generality_mini.sh
#
# 263,200 training images through a ResNet-18/34 trunk on CPU (1-3 h), then
# 2 directions x 21 w x 2 heads = 84 head refits (~1-2 h).
#SBATCH --job-name=mpk_gen_mini
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --array=0-8
#SBATCH --output=slurm/logs/gen_mini_%A_%a.out

set -euo pipefail
mkdir -p slurm/logs
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

MODELS=(run:rgb_tsn_seed0 run:rgb_tsn_seed1 run:rgb_tsn_seed2
        run:dkl_bcs_single_seed0 run:dkl_bcs_single_seed1 run:dkl_bcs_single_seed2
        run:dkl_bcs_single34_seed0 run:dkl_bcs_single34_seed1 run:dkl_bcs_single34_seed2)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

PY=/scratch/byrong/anaconda3/envs/mpk/bin/python
RUNID=${GEN_RUNID:-2026-09-14_gen}
FEAT=/scratch/byrong/mpk/generality/features/$RUNID
BEH=/scratch/byrong/mpk/generality/behaviour/$RUNID
TAG=$(echo "$MODEL" | tr ':' '_')

echo "[gen_mini] task=$SLURM_ARRAY_TASK_ID model=$MODEL tag=$TAG"
cd "$SLURM_SUBMIT_DIR"
[[ -f "$FEAT/${TAG}_cue_conflict.npz" ]] || { echo "FATAL: cue-conflict features missing for $TAG" >&2; exit 1; }
if [[ ! -f "$FEAT/${TAG}_miniecoset.npz" ]]; then
    $PY -m src.analysis.generality.extract --model "$MODEL" --set miniecoset --out-dir "$FEAT"
fi
$PY -m src.analysis.generality.refit_sum --cc "$FEAT/${TAG}_cue_conflict.npz" \
    --mini "$FEAT/${TAG}_miniecoset.npz" --out-dir "$BEH"
echo "[gen_mini] done $MODEL"
