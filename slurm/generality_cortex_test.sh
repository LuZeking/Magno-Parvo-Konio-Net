#!/bin/bash
# One quick-mode element of the generality cortex job (memory/time probe before the array).
#   sbatch slurm/generality_cortex_test.sh
#SBATCH --job-name=mpk_gen_ctx_test
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/gen_ctx_test_%j.out

set -euo pipefail
mkdir -p slurm/logs
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/scratch/byrong/anaconda3/envs/mpk/bin/python
RUNID=${GEN_RUNID:-2026-09-14_gen}
FEAT=/scratch/byrong/mpk/generality/features/$RUNID
BEH=/scratch/byrong/mpk/generality/behaviour/$RUNID
OUT=/scratch/byrong/mpk/generality/cortex/${RUNID}_test
TAG=${GEN_TAG:-run_rgb_tsn_seed0}
cd "$SLURM_SUBMIT_DIR"
# quick: 2 sessions, 500 voxels, 3 w -- exercises every code path
$PY -m src.analysis.generality.cortex_bands --subject 1 --nsd-features "$FEAT/${TAG}_nsd.npz" \
    --factor "$BEH/${TAG}_unit_factor.npz" --out-dir "$OUT" --quick
# full-size variance partition for one subject (the memory/time probe), no w sweep
$PY -m src.analysis.generality.cortex_bands --subject 1 --nsd-features "$FEAT/${TAG}_nsd.npz" \
    --factor "$BEH/${TAG}_unit_factor.npz" --out-dir "$OUT" --skip-w-sweep
echo "[gen_ctx_test] done"
