#!/bin/bash
# Completeness gate for the cortex-to-decision analysis: exits 0 only when all 88
# refit files of slurm/ctd_encode.sh exist, so slurm/ctd_analysis.sh (afterok this
# job) never runs on a partial set. Chain it afterany the encode array and any
# rerun arrays of timed-out elements:
#   G=$(env -u TMPDIR sbatch --parsable --dependency=afterany:<enc>:<rerun> slurm/ctd_gate.sh)
#   env -u TMPDIR sbatch --export=ALL,CODE=<snapshot> --dependency=afterok:$G slurm/ctd_analysis.sh
# A missing file is listed; rerun that element (same script, longer --time) and resubmit gate + analysis.
#SBATCH --job-name=mpk_ctd_gate
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/byrong/mpk/encoding/logs/2026-10-26_ctd/gate_%j.out
set -uo pipefail
trap '[[ ${TMPDIR:-unset} == unset ]] || rm -rf $TMPDIR' EXIT
OUT=/scratch/byrong/mpk/encoding/results/2026-10-26_ctd
missing=0; n=0
for RUN in fs2_dvd300_avg_w050_ls_seed{0,1,2} fs2_dvdfix300_avg_w050_ls_seed{0,1,2} fs2_dvd300swap_avg_w050_ls_seed{0,1,2} untrained_dvd300_seed0 gabor_dvd300; do
  for s in 01 02 03 04 05 06 07 08; do
    n=$((n+1)); f=$OUT/${RUN}_subj$s.npz
    [[ -s $f ]] || { echo "MISSING $f"; missing=$((missing+1)); }
  done
done
echo "[ctd_gate] $((n-missing))/$n refit files present $(date -Is)"
[[ $missing -eq 0 ]] || exit 1
