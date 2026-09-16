#!/bin/bash
# Unit-rule cortex fits (PLAN_GENERALITY D4 machinery) for the two ImageNet networks whose NSD
# features were extracted for the subspace plan but never fitted under the unit rule: ViT-B/16 and
# ConvNeXt-B. Prompted by the v4 referee simulation (docs/NMI_REVIEW_v4_2026-09-16.md, Advance
# lever): does the unit-level cortical result of Fig. 6c-d hold in ImageNet networks other than
# ResNet-50? Not pre-registered -- reported as a post hoc extension of P3-P5 in the plan outcomes.
#
#   sbatch slurm/generality_cortex_vit_cnx.sh                       # variance partition (S, T bands)
#   GEN_MODE=w GEN_NW=11 sbatch slurm/generality_cortex_vit_cnx.sh  # lossy-sum w sweep
#
# Same pipeline, inputs and output directory as slurm/generality_cortex.sh (run 2026-09-14_gen),
# so tmp_analysis/figs_v2/generality_cortex.py picks the new families up unchanged.
#SBATCH --job-name=mpk_gen_ctx2
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --array=0-15%8
#SBATCH --output=slurm/logs/gen_ctx2_%A_%a.out

set -euo pipefail
mkdir -p slurm/logs
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

MODELS=(tv:vit_b_16 tv:convnext_base)
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
echo "[gen_ctx2] task=$TASK model=$MODEL subj=$SUBJECT mode=$MODE"
cd "$SLURM_SUBMIT_DIR"
[[ -f "$FEAT/${TAG}_nsd.npz" ]] || { echo "FATAL: NSD features missing for $TAG" >&2; exit 1; }
[[ -f "$BEH/${TAG}_unit_factor.npz" ]] || { echo "FATAL: unit factor missing for $TAG" >&2; exit 1; }
$PY -m src.analysis.generality.cortex_bands --subject "$SUBJECT" \
    --nsd-features "$FEAT/${TAG}_nsd.npz" --factor "$BEH/${TAG}_unit_factor.npz" \
    --out-dir "$OUT" $FLAG
echo "[gen_ctx2] done $MODEL subj$SUBJECT"
