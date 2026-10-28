#!/bin/bash
# Cortex-to-decision analysis, v2 record (results/cortex_decision/EXECUTION_LEDGER.md §3–§4):
# the registered analysis of slurm/ctd_analysis.sh rerun unchanged on the same 88
# refit files, from a code snapshot whose cortex_decision.py masks the six
# cue-conflict images no observer resolved (h = NaN) in the permutation null and
# the exemplar bootstrap, as the observed correlation always did (defect found
# 2026-10-28 after the v1 run; ledger §3). Everything else is the v1 code path,
# so v2 must equal the v1 record (analysis/) except the shape-target nulls and
# CIs, which must equal results/cortex_decision/ctd_h_null_fix.json. Writes to a
# new directory (analysis_v2/); the v1 record is not touched. --mem raised from
# 16G to 24G (v1 MaxRSS 16.0 GB).
#
#   env -u TMPDIR sbatch --export=ALL,CODE=<pinned code dir> slurm/ctd_analysis_v2.sh
#SBATCH --job-name=mpk_ctd_ana2
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/byrong/mpk/encoding/logs/2026-10-26_ctd/analysis_v2_%j.out
set -euo pipefail
trap '[[ ${TMPDIR:-unset} == unset ]] || rm -rf $TMPDIR' EXIT
: "${CODE:?pass --export=ALL,CODE=<pinned code dir>}"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$CODE:$CODE/tmp_analysis"
PY=/home/byrong/anaconda3/envs/mpk/bin/python
FITS=/scratch/byrong/mpk/encoding/results/2026-10-26_ctd
HUMAN=/scratch/byrong/mpk/perimage_behavior/results/2026-09-07_geirhos_perimage
OUT=$FITS/analysis_v2
[[ -f $HUMAN/human_perimage.npz && -f $HUMAN/human_ceiling.json ]] || { echo "FATAL: observer profiles missing in $HUMAN" >&2; exit 1; }
[[ -e $OUT/ctd_analysis.json ]] && { echo "FATAL: $OUT/ctd_analysis.json exists; v2 is written once" >&2; exit 1; }
echo "[ctd_ana2] code=$(cat "$CODE/COMMIT") $(hostname) $(date -Is)"
cd "$CODE"
$PY tmp_analysis/test_cortex_decision.py
$PY -m src.analysis.encoding.cortex_decision --fits-dir "$FITS" --human-dir "$HUMAN" --out-dir "$OUT"
$PY -m src.analysis.encoding.cortex_decision --fits-dir "$FITS" --human-dir "$HUMAN" --out-dir "$OUT/sens_spearman" --method spearman
$PY -m src.analysis.encoding.cortex_decision --fits-dir "$FITS" --human-dir "$HUMAN" --out-dir "$OUT/sens_v1_streams" --v1-streams
$PY -m src.analysis.encoding.cortex_decision --fits-dir "$FITS" --human-dir "$HUMAN" --out-dir "$OUT/sens_all_voxels" --all-voxels
echo "[ctd_ana2] done $(date -Is); compare $OUT with $FITS/analysis and results/cortex_decision/ctd_h_null_fix.json (ledger §4)"
