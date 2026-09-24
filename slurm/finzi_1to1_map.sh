#!/bin/bash
# 1-to-1 fixed-quota mapping (docs/PLAN_FINZI_1TO1_2026-09-24.md §4-§8).
# Task = network x subject. MODE=main (default) or MODE=pc (LOSO controls; needs
# every subject's main output for that network first).
#
#   sbatch slurm/finzi_1to1_map.sh                                  # 17 networks x 8
#   MODE=pc NETS="tdann_a025_s0 diet_s0" sbatch --array=0-15 slurm/finzi_1to1_map.sh
#
# CPU only. Peak ~6 GB (TDANN: 25k units x 9.5k images float32, z-scored copy,
# two hemisphere correlation blocks); 12G leaves headroom.
#SBATCH --job-name=f1to1_map
#SBATCH --partition=main
#SBATCH --qos=inefficient
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --array=0-135%10
#SBATCH --exclude=c025
#SBATCH --output=slurm/logs/f1to1_map_%A_%a.out

set -euo pipefail
unset SLURM_CPU_BIND SLURM_CPU_BIND_LIST SLURM_CPU_BIND_TYPE SLURM_CPU_BIND_VERBOSE
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4

MODE=${MODE:-main}
NETS=${NETS:-"tdann_a025_s0 tdann_a0_s0 diet_s0 diet_s1 diet_s2 nodiet_s0 nodiet_s1 nodiet_s2 swap_s0 swap_s1 swap_s2 untrained_diet_s0 untrained_diet_s1 untrained_diet_s2 untrained_nodiet_s0 untrained_nodiet_s1 untrained_nodiet_s2"}
read -r -a NET_ARR <<< "$NETS"
I=$SLURM_ARRAY_TASK_ID
NET=${NET_ARR[$(( I / 8 ))]}
SUBJECT=$(( I % 8 + 1 ))
PY=/scratch/byrong/anaconda3/envs/mpk/bin/python
cd "$SLURM_SUBMIT_DIR"
echo "[f1to1_map] mode=$MODE net=$NET subject=$SUBJECT host=$(hostname) start=$(date -Is)"
$PY -m src.analysis.finzi_1to1.mapping --network "$NET" --subject "$SUBJECT" --mode "$MODE"
echo "[f1to1_map] done $(date -Is)"
