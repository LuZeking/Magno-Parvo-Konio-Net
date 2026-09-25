#!/bin/bash
# Network gateway between g017 (no /scratch) and /scratch for the spatial-loss pilot
# (PLAN_TSN_SPATIAL_PILOT_2026-09-25 §10). /home cannot hold the 59 GB dataset (hidden
# ~400G quota; the first launch died with EDQUOT), so an rsync daemon on a `main` node,
# which mounts /scratch, serves:
#   data  = /scratch/byrong/VST_data/VST_datasets        read-only
#   pilot = /scratch/byrong/TSN/spatial_pilot             read-write (runs/)
# Auth: user `pilot`, secrets in /home/byrong/.config/tsn_pilot_rsync (mode 600), and
# only g017 may connect. Writes "<host> <port>" to $GW_FILE; runs until cancelled by
# slurm/spatial_pilot_gateway_stop.sh (chained afterany on the training array).
#SBATCH --job-name=sl_gateway
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=3-00:00:00
#SBATCH --exclude=c025,c094,c072,c098,c150,c088
#SBATCH --output=slurm/logs/sl_gateway_%j.out

set -euo pipefail
GW_FILE=/home/byrong/_staging/TSN/sl_gateway.txt
SECRETS=/home/byrong/.config/tsn_pilot_rsync/secrets
CONF=${TMPDIR:-/tmp}/sl_rsyncd_$SLURM_JOB_ID.conf
mkdir -p /scratch/byrong/TSN/spatial_pilot/runs "$(dirname "$GW_FILE")"
PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1])")
cat > "$CONF" <<EOF
use chroot = no
max connections = 8
pid file = ${TMPDIR:-/tmp}/sl_rsyncd_$SLURM_JOB_ID.pid
lock file = ${TMPDIR:-/tmp}/sl_rsyncd_$SLURM_JOB_ID.lock
log file = /home/byrong/projects/TSN/Magno-Parvo-Konio-Net/.claude/worktrees/finzi-1to1/slurm/logs/sl_gateway_${SLURM_JOB_ID}_rsyncd.log
reverse lookup = yes
hosts allow = 160.45.4.200 g017 g017.* g017-*
hosts deny = *
auth users = pilot
secrets file = $SECRETS
strict modes = yes
[data]
    path = /scratch/byrong/VST_data/VST_datasets
    read only = yes
[pilot]
    path = /scratch/byrong/TSN/spatial_pilot
    read only = no
EOF
echo "$(hostname -f) $PORT" > "$GW_FILE.tmp"
mv "$GW_FILE.tmp" "$GW_FILE"
echo "[gateway] $(hostname -f):$PORT up $(date -Is)"
exec rsync --daemon --no-detach --port="$PORT" --config="$CONF"
