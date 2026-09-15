#!/bin/bash
# PLAN_SUBSPACE D3 (pre-registered 2026-09-15): lossy-sum refit in block form,
# z(w) = [w*s + (1-w)*t, r], linear and two-layer heads refitted at every w on
# MiniEcoset training features, our nine single-trunk networks.
#
#   sbatch --dependency=afterok:<sub_cc jobid> slurm/subspace_refit.sh
#
# 2 directions x 21 w x 2 heads = 84 refits per network (1-3 h on 8 CPU cores).
#SBATCH --job-name=mpk_sub_refit
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-8
#SBATCH --output=slurm/logs/sub_refit_%A_%a.out

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

echo "[sub_refit] task=$SLURM_ARRAY_TASK_ID model=$MODEL tag=$TAG"
cd "$SLURM_SUBMIT_DIR"
[[ -f "$FEAT/${TAG}_miniecoset.npz" ]] || { echo "FATAL: MiniEcoset features missing for $TAG" >&2; exit 1; }
$PY -m src.analysis.generality.refit_sum --cc "$FEAT/${TAG}_cue_conflict.npz" \
    --mini "$FEAT/${TAG}_miniecoset.npz" --out-dir "$BEH" --subspace --out-suffix "_subspace"
echo "[sub_refit] done $MODEL"
