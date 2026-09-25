#!/bin/bash
# Stops the pilot's rsync gateway once the training array has ended (afterany), so it
# never idles for its full time limit. GW_JOB comes through the environment.
#SBATCH --job-name=sl_gw_stop
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --exclude=c025,c094,c072,c098,c150,c088
#SBATCH --output=slurm/logs/sl_gw_stop_%j.out

echo "[gw-stop] cancelling gateway ${GW_JOB:?}"
scancel "$GW_JOB"
rm -f /home/byrong/_staging/TSN/sl_gateway.txt
