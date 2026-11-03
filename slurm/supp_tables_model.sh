#!/bin/bash
# Supplementary Tables S1-S10 for the model of record (docs/PLAN_SUPP_MODEL_2026-11-03.md, section 5, Stage A).
# One CPU job on main: runs the generator's tests, then tmp_analysis/figs_v3/supp_tables_model.py, which reads
# the stored encoding fits (/scratch, 2026-10-16_fs2), the committed behavioural results and the NSD pRF/atlas
# files, and writes supp_tables_model.{json,md} + provenance.json. Nothing is trained, refit or re-encoded.
#
#   SHA=$(git rev-parse HEAD); D=/home/byrong/projects/TSN/fusion_stage1_code/$SHA
#   mkdir -p $D && git archive $SHA | tar -x -C $D && echo "$SHA fusion-stage1 (Magno-Parvo-Konio-Net)" > $D/COMMIT
#   # pin the output and log directories to a healthy OST first (lfs setstripe -i <ost> <dir>)
#   env -u TMPDIR sbatch --export=ALL,CODE=$D slurm/supp_tables_model.sh
#SBATCH --job-name=mpk_supp_v3
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/byrong/mpk/encoding/logs/2026-11-03_supp_v3/supp_%j.out
set -euo pipefail
trap '[[ ${TMPDIR:-unset} == unset ]] || rm -rf $TMPDIR' EXIT
: "${CODE:?pass --export=ALL,CODE=<pinned code dir>}"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$CODE:$CODE/tmp_analysis:$CODE/tmp_analysis/figs_v2"
export FS2_ENCODING_DIR=/scratch/byrong/mpk/encoding/results/2026-10-16_fs2
export FS2_VPREC_DIR="$CODE/results/vp_recovery_realnc_v3"
export FS2_CKPT_DIR=/scratch/byrong/TSN/fusion_stage1/checkpoints/2026-10-12_stage2
PY=/home/byrong/anaconda3/envs/mpk/bin/python
OUT=/scratch/byrong/mpk/encoding/results/2026-11-03_supp_v3
[[ -e $OUT/supp_tables_model.json ]] && { echo "FATAL: $OUT/supp_tables_model.json exists; the record is written once" >&2; exit 1; }
mkdir -p "$OUT"
echo "[supp] code=$(cat "$CODE/COMMIT") $(hostname) $(date -Is) out=$OUT"
cd "$CODE"
$PY tmp_analysis/test_supp_tables_model.py
$PY tmp_analysis/figs_v3/supp_tables_model.py --out "$OUT"
ls -la "$OUT"
echo "[supp] done $(date -Is)"
