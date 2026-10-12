#!/bin/bash
# Fusion Stage 2 smoke test (CPU, `main`): model-level checks
# (tmp_analysis/smoke_fusion_stage2.py), 2 steps of every fs2_ condition, a
# resume of one, and the unchanged Stage-1 A0 path. Same layout as
# slurm/smoke_fusion_stage1.sh: runs in node-local $TMPDIR, copied to /scratch.
#
#   env -u TMPDIR sbatch slurm/smoke_fusion_stage2.sh
#SBATCH --job-name=fs2_smoke
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --exclude=c029,c009
#SBATCH --output=/scratch/byrong/TSN/fusion_stage1/logs/smoke2_%j.out

set -euo pipefail
trap '[[ ${TMPDIR:-unset} == unset ]] || rm -rf $TMPDIR' EXIT
: "${TMPDIR:?no node-local TMPDIR}"
cd "$SLURM_SUBMIT_DIR"
unset SLURM_CPU_BIND SLURM_CPU_BIND_LIST SLURM_CPU_BIND_TYPE SLURM_CPU_BIND_VERBOSE
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16

PY=/home/byrong/anaconda3/envs/mpk/bin/python
DKL=/scratch/byrong/VST_data/VST_datasets/miniecoset_256px_dkl_uint8.h5
CC=/scratch/byrong/TSN/fusion_stage1/cache/cue_conflict_dkl_u8_v1.npz
LOCAL=$TMPDIR/smoke
OUT=/scratch/byrong/TSN/fusion_stage1/smoke/$SLURM_JOB_ID
echo "commit $(git rev-parse --short HEAD) on $(hostname) $(date -Is)"

"$PY" tmp_analysis/smoke_fusion_stage2.py

SMOKE=(--dataset "$DKL" --seed 0 --out-dir "$LOCAL" --epochs 2 --batch-size 8
       --num-workers 0 --val-batches 2 --cue-conflict-npz "$CC")
for c in $("$PY" -c "from src.training.standalone_trainer import STAGE2_CONDITIONS as S; print(' '.join(S))") fs1_bcs_avg; do
    echo "== $c $(date +%T)"
    "$PY" -m src.training.standalone_trainer --condition "$c" "${SMOKE[@]}" --max-steps 2 2>&1 | grep -v "pin_memory\|super().__init__"
done
echo "== resume fs2_bcs_avg_w070 $(date +%T)"
"$PY" -m src.training.standalone_trainer --condition fs2_bcs_avg_w070 "${SMOKE[@]}" --max-steps 4 --resume 2>&1 | grep -v "pin_memory\|super().__init__" | cut -c1-200

"$PY" - "$LOCAL" <<'EOF'
import json, sys, os, torch
out = sys.argv[1]
for run in sorted(os.listdir(out)):
    h = json.load(open(os.path.join(out, run, "history.json")))
    cfg = json.load(open(os.path.join(out, run, "config.json")))
    r = h[-1]
    ck = torch.load(os.path.join(out, run, "checkpoint_last.pt"), map_location="cpu", weights_only=False)
    print(f"{run:<26} epochs {[x['epoch'] for x in h]} val_top1 {r['val_top1']:.3f}"
          f" | joint SB {r['cc']['joint']['sb']:.2f} | heads {sorted(r['val'])}"
          f" | train_ce {r.get('train_ce')} | step {ck['step']}")
EOF
mkdir -p "$OUT"
rsync -a "$LOCAL/" "$OUT/" && echo "copied to $OUT"
echo "smoke OK $(date -Is)"
