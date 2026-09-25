#!/bin/bash
# Removes the pilot's node-local copy of the dataset (and any unverified leftovers)
# from g017's NVMe once the training array has ended (afterany). The runs themselves
# live on /scratch; a failed task keeps its /scratch copy, so nothing is lost here.
#SBATCH --job-name=sl_local_clean
#SBATCH --partition=agcichy
#SBATCH --qos=standard
#SBATCH --nodelist=g017
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=slurm/logs/sl_local_clean_%j.out

du -sh /localscratch/byrong/tsn_spatial_pilot 2>/dev/null
rm -rf /localscratch/byrong/tsn_spatial_pilot
echo "[local-clean] done $(date -Is)"
