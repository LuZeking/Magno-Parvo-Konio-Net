#!/bin/bash
# Fig. 4b on the TSN of record (2026-10-19), step 2 of 2: refit the affine,
# linear and MLP heads at each of 21 readout weights on the cached features
# (tmp_analysis/nonlinear_readout_sweep.py), as slurm/nonlinear_sweep.sh, for
# each run in RUN_LIST (tag <run>_final), from the pinned code. CPU by design
# (a 512-unit MLP over cached features); submit with
# --dependency=aftercorr:<slurm/paper_fs2_nonlinear.sh array> and the same RUN_LIST.
#
#   mkdir -p /scratch/byrong/mpk/nonlinear_readout/logs/2026-10-19_fs2
#   env -u TMPDIR sbatch --array=0-$((N-1)) --dependency=aftercorr:<nlfeat> slurm/paper_fs2_nonlinear_sweep.sh
#
# Writes /scratch/byrong/TSN/nonlinear_readout/results/2026-09-10/<run>_final_nonlinear_sweep.json (the BA's directory).
#SBATCH --job-name=fs2_nlsweep
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --array=0-3
#SBATCH --output=/scratch/byrong/mpk/nonlinear_readout/logs/2026-10-19_fs2/nlsweep_%A_%a.out

set -euo pipefail
trap '[[ ${TMPDIR:-unset} == unset ]] || rm -rf $TMPDIR' EXIT
CODE=${CODE:-/home/byrong/projects/TSN/fusion_stage1_code/c543636}
export OPENBLAS_NUM_THREADS=16 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$CODE:$CODE/tmp_analysis"
PY=/home/byrong/anaconda3/envs/mpk/bin/python

read -r -a RUNS <<< "${RUN_LIST:-fs2_dvd300_avg_w050_ls_seed0 fs2_dvd300_avg_w050_ls_seed1 fs2_dvd300_avg_w050_ls_seed2 fs2_dvdfix300_avg_w050_ls_seed0}"
RUN=${RUNS[$SLURM_ARRAY_TASK_ID]}
TAG=${RUN}_final
FEAT=/scratch/byrong/TSN/nonlinear_readout/features/2026-09-10
OUT=/scratch/byrong/TSN/nonlinear_readout/results/2026-09-10
DEST=$OUT/${TAG}_nonlinear_sweep.json
if [[ -e $DEST && ${OVERWRITE:-0} != 1 ]]; then
  echo "FATAL: $DEST exists; refusing to overwrite" >&2; exit 1
fi
[[ -f $FEAT/${TAG}_features.npz ]] || { echo "FATAL: $FEAT/${TAG}_features.npz missing" >&2; exit 1; }
mkdir -p "$OUT"
echo "[fs2_nlsweep] task=$SLURM_ARRAY_TASK_ID run=$RUN -> $DEST code=$(cat "$CODE/COMMIT") $(hostname) $(date -Is)"
cd "$CODE"
$PY tmp_analysis/nonlinear_readout_sweep.py --features "$FEAT/${TAG}_features.npz" --out-dir "$OUT"
echo "[fs2_nlsweep] done $RUN $(date -Is)"
