#!/bin/bash
# Generality test, cortex step 2 (PLAN_GENERALITY D4): banded ridge with the
# shape- and texture-leaning halves as bands, per (network, subject).
#
#   sbatch --dependency=afterok:<gen_cc>:<gen_nsd> slurm/generality_cortex.sh
#   GEN_MODE=w sbatch ... slurm/generality_cortex.sh      # the 21-w lossy-sum sweep instead
#
# 7 networks x 8 subjects = 56 elements. Float32 designs with small target
# batches: the same kind of fit measured 25.6 GB peak in the readout_w job.
#SBATCH --job-name=mpk_gen_ctx
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --array=0-55%14
#SBATCH --output=slurm/logs/gen_ctx_%A_%a.out

set -euo pipefail
mkdir -p slurm/logs
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

MODELS=(run:rgb_tsn_seed0 run:rgb_tsn_seed1 run:rgb_tsn_seed2
        run:dkl_bcs_single34_seed0 run:dkl_bcs_single34_seed1 run:dkl_bcs_single34_seed2
        tv:resnet50)
TASK=$SLURM_ARRAY_TASK_ID
SUBJECT=$(( TASK % 8 + 1 ))
MODEL=${MODELS[$(( TASK / 8 ))]}
TAG=$(echo "$MODEL" | tr ':' '_')

PY=/scratch/byrong/anaconda3/envs/mpk/bin/python
RUNID=${GEN_RUNID:-2026-09-14_gen}
FEAT=/scratch/byrong/mpk/generality/features/$RUNID
BEH=/scratch/byrong/mpk/generality/behaviour/$RUNID
OUT=/scratch/byrong/mpk/generality/cortex/$RUNID
MODE=${GEN_MODE:-vp}
if [[ "$MODE" == "w" ]]; then FLAG="--only-w-sweep --n-w ${GEN_NW:-21}"; else FLAG="--skip-w-sweep"; fi

echo "[gen_ctx] task=$TASK model=$MODEL subj=$SUBJECT mode=$MODE"
cd "$SLURM_SUBMIT_DIR"
[[ -f "$FEAT/${TAG}_nsd.npz" ]] || { echo "FATAL: NSD features missing for $TAG" >&2; exit 1; }
[[ -f "$BEH/${TAG}_unit_factor.npz" ]] || { echo "FATAL: unit factor missing for $TAG" >&2; exit 1; }
$PY -m src.analysis.generality.cortex_bands --subject "$SUBJECT" \
    --nsd-features "$FEAT/${TAG}_nsd.npz" --factor "$BEH/${TAG}_unit_factor.npz" \
    --out-dir "$OUT" $FLAG
echo "[gen_ctx] done $MODEL subj$SUBJECT"
