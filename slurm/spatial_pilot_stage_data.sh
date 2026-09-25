#!/bin/bash
# Stage the miniecoset DKL dataset on /home for the spatial-loss pilot's g017 runs
# (PLAN_TSN_SPATIAL_PILOT_2026-09-25). g017 is the only free GPU node; it does not mount
# /scratch, and compute nodes cannot ssh to the login node, so /home is the only path.
# /home truncates silently when full, so the copy is checked byte for byte with cmp.
# The staged copy is removed by slurm/spatial_pilot_migrate.sh once both runs finish.
#SBATCH --job-name=sl_stage
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --exclude=c025,c094,c072,c098,c150,c088
#SBATCH --output=slurm/logs/sl_stage_%j.out

set -euo pipefail
SRC=/scratch/byrong/VST_data/VST_datasets/miniecoset_256px_dkl_uint8.h5
DST_DIR=/home/byrong/_staging/TSN/spatial_pilot_data
DST=$DST_DIR/miniecoset_256px_dkl_uint8.h5
mkdir -p "$DST_DIR"
df -h /home | tail -1
t0=$(date +%s)
rsync -a --info=stats2 "$SRC" "$DST.partial"
echo "copied in $(( $(date +%s) - t0 )) s"
cmp "$SRC" "$DST.partial"
[ "$(stat -c %s "$SRC")" = "$(stat -c %s "$DST.partial")" ]
mv "$DST.partial" "$DST"
echo "verified: $(stat -c %s "$DST") bytes, cmp identical"
/home/byrong/anaconda3/envs/mpk/bin/python -c "
import h5py; f = h5py.File('$DST', 'r')
print({k: (f[k]['data'].shape if isinstance(f[k], h5py.Group) else f[k].shape) for k in f})"
echo "total $(( $(date +%s) - t0 )) s"
