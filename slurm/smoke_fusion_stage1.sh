#!/bin/bash
# Fusion Stage 1 smoke test (CPU, `main`): model-level checks, the cue-conflict
# cache (built once, with its parity check against cue_conflict.py's front
# end), 2 steps of every fs1_ condition, a resume of two of them, and the
# unchanged dkl_bcs path. Runs write to node-local $TMPDIR and are copied to
# /scratch at the end: the first smoke (26606696) lost 85 min to one stalled
# /scratch checkpoint write.
#
#   env -u TMPDIR sbatch slurm/smoke_fusion_stage1.sh
#SBATCH --job-name=fs1_smoke
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/byrong/TSN/fusion_stage1/logs/smoke_%j.out

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

"$PY" tmp_analysis/smoke_fusion_stage1.py
[ -s "$CC" ] || "$PY" scripts/fusion_stage1_cue_conflict_cache.py --out "$CC"

SMOKE=(--dataset "$DKL" --seed 0 --out-dir "$LOCAL" --epochs 2 --batch-size 8
       --num-workers 0 --val-batches 2 --cue-conflict-npz "$CC")
for c in $("$PY" -c "from src.training.standalone_trainer import STAGE1_CONDITIONS as S; print(' '.join(S))"); do
    echo "== $c $(date +%T)"
    "$PY" -m src.training.standalone_trainer --condition "$c" "${SMOKE[@]}" --max-steps 2 2>&1 | grep -v "pin_memory\|super().__init__"
done
for c in fs1_bcs_avg fs1_hsf_hinge_k3_l4; do
    echo "== resume $c $(date +%T)"
    "$PY" -m src.training.standalone_trainer --condition "$c" "${SMOKE[@]}" --max-steps 4 --resume 2>&1 | grep -v "pin_memory\|super().__init__" | cut -c1-200
done
echo "== regression dkl_bcs $(date +%T)"
"$PY" -m src.training.standalone_trainer --condition dkl_bcs "${SMOKE[@]}" --max-steps 2 2>&1 | grep -v "pin_memory\|super().__init__"

"$PY" - "$LOCAL" <<'EOF'
import json, sys, os, torch
out = sys.argv[1]
for run in sorted(os.listdir(out)):
    h = json.load(open(os.path.join(out, run, "history.json")))
    r = h[-1]
    extra = ""
    if "cc" in r:
        extra = (f" | joint SB {r['cc']['joint']['sb']:.2f} cover {r['cc']['joint']['cover']:.3f}"
                 f" | heads {sorted(r['val'])} | sweeps {list(r['sweep'])} | lr {r['lr']:.2e}")
        ck = torch.load(os.path.join(out, run, "checkpoint_last.pt"), map_location="cpu", weights_only=False)
        extra += f" | step {ck['step']} sched {ck['scheduler']['last_epoch']}"
    print(f"{run:<34} epochs {[x['epoch'] for x in h]} val_top1 {r['val_top1']:.3f}{extra}")
EOF
mkdir -p "$OUT"
rsync -a "$LOCAL/" "$OUT/" && echo "copied to $OUT"
echo "smoke OK $(date -Is)"
