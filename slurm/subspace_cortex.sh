#!/bin/bash
# PLAN_SUBSPACE D4 (pre-registered 2026-09-15): three-band NSD encoding (S, T, R) per
# (network, subject), and the 11-point w sweep of the 15-d lossy sum.
#
#   sbatch --dependency=afterok:<sub_cc> --array=0-55%14 slurm/subspace_cortex.sh          # stored NSD features
#   sbatch --dependency=afterok:<sub_cc>:<sub_nsd> --array=56-71%8 slurm/subspace_cortex.sh # ViT-B/16, ConvNeXt-B
#   GEN_MODE=w sbatch ... slurm/subspace_cortex.sh                                            # the w sweep instead
#
# 9 networks x 8 subjects = 72 elements. Five fits per element (S, T, T+R, S+R, S+T+R):
# ~2 h for 512-d trunks, up to ~6 h for the 2048-d ResNet-50. The same kind of fit
# measured <= 14 GB for two 1024-d bands; 48G is headroom, not a measurement.
#SBATCH --job-name=mpk_sub_ctx
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-71%14
#SBATCH --output=slurm/logs/sub_ctx_%A_%a.out

set -euo pipefail
mkdir -p slurm/logs
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

MODELS=(run:rgb_tsn_seed0 run:rgb_tsn_seed1 run:rgb_tsn_seed2
        run:dkl_bcs_single34_seed0 run:dkl_bcs_single34_seed1 run:dkl_bcs_single34_seed2
        tv:resnet50 tv:vit_b_16 tv:convnext_base)
TASK=$SLURM_ARRAY_TASK_ID
SUBJECT=$(( TASK % 8 + 1 ))
MODEL=${MODELS[$(( TASK / 8 ))]}
TAG=$(echo "$MODEL" | tr ':' '_')

PY=/scratch/byrong/anaconda3/envs/mpk/bin/python
RUNID=${GEN_RUNID:-2026-09-14_gen}
OUTID=${SUB_RUNID:-2026-09-15_sub}
FEAT=/scratch/byrong/mpk/generality/features/$RUNID
BEH=/scratch/byrong/mpk/generality/behaviour/$RUNID
OUT=/scratch/byrong/mpk/generality/cortex/$OUTID
MODE=${GEN_MODE:-vp}
if [[ "$MODE" == "w" ]]; then FLAG="--only-w-sweep --n-w ${GEN_NW:-11}"; else FLAG="--skip-w-sweep"; fi

echo "[sub_ctx] task=$TASK model=$MODEL subj=$SUBJECT mode=$MODE out=$OUT"
cd "$SLURM_SUBMIT_DIR"
[[ -f "$FEAT/${TAG}_nsd.npz" ]] || { echo "FATAL: NSD features missing for $TAG" >&2; exit 1; }
[[ -f "$BEH/${TAG}_subspace.npz" ]] || { echo "FATAL: subspace factor missing for $TAG" >&2; exit 1; }
$PY -m src.analysis.generality.cortex_bands --subject "$SUBJECT" \
    --nsd-features "$FEAT/${TAG}_nsd.npz" --factor "$BEH/${TAG}_subspace.npz" \
    --out-dir "$OUT" --subspace --n-iter 20 --n-iter-joint 30 $FLAG
echo "[sub_ctx] done $MODEL subj$SUBJECT"
