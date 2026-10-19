#!/bin/bash
# Exemplar-held-out spans (E0-E3) and probes on the two-stream concatenation of
# the TSN of record (2026-10-19), as slurm/exemplar_zoo.sh does for
# concat_dkl_bcs_seed*, for each run in RUN_LIST (tag concat_<run>), from the
# pinned code. Reads the features and subspace/gate files that
# slurm/paper_fs2_generality.sh wrote; nothing is extracted or retrained.
#
#   mkdir -p /scratch/byrong/mpk/exemplar/logs/2026-10-19_fs2
#   env -u TMPDIR sbatch --array=0-$((N-1)) slurm/paper_fs2_exemplar_zoo.sh
#
# Writes /scratch/byrong/mpk/exemplar/results/2026-09-21_exemplar/concat_<run>_exemplar.{json,npz}
# (the BA's directory, read by tmp_analysis/figs_v2/exemplar_aggregate.py).
#SBATCH --job-name=fs2_exzoo
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --array=0-3
#SBATCH --output=/scratch/byrong/mpk/exemplar/logs/2026-10-19_fs2/exzoo_%A_%a.out

set -euo pipefail
trap '[[ ${TMPDIR:-unset} == unset ]] || rm -rf $TMPDIR' EXIT
CODE=${CODE:-/home/byrong/projects/TSN/fusion_stage1_code/c543636}
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$CODE"
PY=/home/byrong/anaconda3/envs/mpk/bin/python

read -r -a RUNS <<< "${RUN_LIST:-fs2_dvd300_avg_w050_ls_seed0 fs2_dvd300_avg_w050_ls_seed1 fs2_dvd300_avg_w050_ls_seed2 fs2_dvdfix300_avg_w050_ls_seed0}"
RUN=${RUNS[$SLURM_ARRAY_TASK_ID]}
TAG=concat_$RUN
RUNID=${GEN_RUNID:-2026-09-14_gen}
FEAT=/scratch/byrong/mpk/generality/features/$RUNID
BEH=/scratch/byrong/mpk/generality/behaviour/$RUNID
OUT=/scratch/byrong/mpk/exemplar/results/2026-09-21_exemplar
DEST=$OUT/${TAG}_exemplar.json
if [[ -e $DEST && ${OVERWRITE:-0} != 1 ]]; then
  echo "FATAL: $DEST exists; refusing to overwrite" >&2; exit 1
fi
for f in "$FEAT/${TAG}_cue_conflict.npz" "$BEH/${TAG}_subspace.json" "$BEH/${TAG}_gain_accuracy_subspace.json"; do
  [[ -f $f ]] || { echo "FATAL: $f missing (run slurm/paper_fs2_generality.sh first)" >&2; exit 1; }
done
mkdir -p "$OUT"
echo "[fs2_exzoo] task=$SLURM_ARRAY_TASK_ID tag=$TAG code=$(cat "$CODE/COMMIT") $(hostname) $(date -Is)"
cd "$CODE"
$PY -m src.analysis.generality.exemplar --features "$FEAT/${TAG}_cue_conflict.npz" --behaviour-dir "$BEH" --out-dir "$OUT"
echo "[fs2_exzoo] done $TAG $(date -Is)"
