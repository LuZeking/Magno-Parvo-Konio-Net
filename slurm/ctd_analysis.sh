#!/bin/bash
# Cortex-to-decision analysis (docs/PLAN_CORTEX_DECISION_2026-10-26.md §2-§4):
# reads the refit encoders of slurm/ctd_encode.sh and the observers' per-image
# profiles of tmp_analysis/human_perimage.py, writes ctd_analysis.json and
# ctd_summary.md, then the plan's two sensitivity runs (Spearman over voxels;
# all voxels). Submitted afterok the encoding array (see ctd_encode.sh header).
#
#   env -u TMPDIR sbatch --export=ALL,CODE=<pinned code dir> \
#       --dependency=afterok:<ctd_encode job id> slurm/ctd_analysis.sh
#SBATCH --job-name=mpk_ctd_ana
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/byrong/mpk/encoding/logs/2026-10-26_ctd/analysis_%j.out
set -euo pipefail
trap '[[ ${TMPDIR:-unset} == unset ]] || rm -rf $TMPDIR' EXIT
: "${CODE:?pass --export=ALL,CODE=<pinned code dir>}"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$CODE:$CODE/tmp_analysis"
PY=/home/byrong/anaconda3/envs/mpk/bin/python
FITS=/scratch/byrong/mpk/encoding/results/2026-10-26_ctd
HUMAN=/scratch/byrong/mpk/perimage_behavior/results/2026-09-07_geirhos_perimage
OUT=$FITS/analysis
[[ -f $HUMAN/human_perimage.npz && -f $HUMAN/human_ceiling.json ]] || { echo "FATAL: observer profiles missing in $HUMAN" >&2; exit 1; }
echo "[ctd_ana] code=$(cat "$CODE/COMMIT") $(hostname) $(date -Is)"
cd "$CODE"
$PY tmp_analysis/test_cortex_decision.py
$PY -m src.analysis.encoding.cortex_decision --fits-dir "$FITS" --human-dir "$HUMAN" --out-dir "$OUT"
$PY -m src.analysis.encoding.cortex_decision --fits-dir "$FITS" --human-dir "$HUMAN" --out-dir "$OUT/sens_spearman" --method spearman
$PY -m src.analysis.encoding.cortex_decision --fits-dir "$FITS" --human-dir "$HUMAN" --out-dir "$OUT/sens_v1_streams" --v1-streams
$PY -m src.analysis.encoding.cortex_decision --fits-dir "$FITS" --human-dir "$HUMAN" --out-dir "$OUT/sens_all_voxels" --all-voxels
echo "[ctd_ana] done $(date -Is); copy $OUT/ctd_analysis.json + ctd_summary.md into results/cortex_decision/ of the worktree and write the ledger"
