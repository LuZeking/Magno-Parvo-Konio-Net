#!/bin/bash
# Verify the standalone trainer against the CONVERTED DKL dataset.
#
# The earlier smoke test used the RGB source h5, where the trainer's DKL
# branch reads channels that are still sRGB. This checks the real training
# input: a file whose three channels already are the normalized DKL axes.
#
#SBATCH --job-name=mpk_smoke_dkl
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/smoke_dkl_%j.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1

PY=/scratch/byrong/anaconda3/envs/mpk/bin/python
DKL=/scratch/byrong/VST_data/VST_datasets/miniecoset_256px_dkl_uint8.h5
OUT=$SLURM_SUBMIT_DIR/tmp_analysis/smoke_runs_dkl

echo "=== dataset structure ==="
$PY - <<'PYEOF'
import h5py
import numpy as np
p = "/scratch/byrong/VST_data/VST_datasets/miniecoset_256px_dkl_uint8.h5"
with h5py.File(p, "r") as f:
    print("top-level:", sorted(f.keys()))
    for split in ("train", "val"):
        g = f[split]
        print(f"  {split}: " + ", ".join(
            f"{k}{f[split][k].shape}" for k in sorted(g.keys())))
    n = int(np.unique(f["train"]["labels"][:]).size)
    print("  n classes:", n)
    print("  dkl min:", f["dkl_channel_min"][:])
    print("  dkl max:", f["dkl_channel_max"][:])
    # per-channel means of a small sample: lum should sit well above the
    # near-zero-centred chromatic axes after [0,1] normalization
    x = f["train"]["data"][:64].astype("float32") / 255.0
    print("  channel means (lum, rg, by):", x.mean(axis=(0, 1, 2)).round(4))
PYEOF

for COND in dkl_base dkl_bcs dkl_bcs_mheavy dkl_bcs_parvo_highsf; do
  echo "=== $COND ==="
  $PY -m src.training.standalone_trainer \
      --dataset "$DKL" --condition "$COND" --seed 0 --out-dir "$OUT" \
      --epochs 1 --max-steps 3 --batch-size 8 --num-workers 2 \
      --val-batches 2 --ckpt-every 1 2>&1 |
      grep -aE "^\[|Traceback|Error" | tail -4
done
echo "=== smoke_train_dkl finished ==="
