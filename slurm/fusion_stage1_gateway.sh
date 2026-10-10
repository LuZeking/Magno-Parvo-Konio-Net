#!/bin/bash
# Network gateway between g017 (no /scratch mount) and /scratch for the Stage-1
# pilots moved there (slurm/train_fusion_stage1_g017.sh). Same design as
# slurm/spatial_pilot_gateway.sh, which fed the g017 spatial pilot at ~300 MB/s:
# an rsync daemon on a `main` node, which mounts /scratch, serves
#   data = /scratch/byrong/VST_data/VST_datasets   read-only (training h5)
#   fs1  = /scratch/byrong/TSN/fusion_stage1        read-write (cache, checkpoints)
# Auth: user `pilot`, secrets in /home/byrong/.config/tsn_pilot_rsync (mode 600),
# and only g017 may connect. Writes "<host> <port>" to $GW_FILE; runs until
# slurm/fusion_stage1_g017_finish.sh cancels it (chained after the g017 array).
# The daemon runs as a waited-on child, not exec'd, so the EXIT trap still
# removes $TMPDIR when the job is cancelled.
#SBATCH --job-name=fs1_gateway
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=3-00:00:00
#SBATCH --exclude=c025,c094,c072,c098,c150,c088
#SBATCH --output=/scratch/byrong/TSN/fusion_stage1/logs/gateway_%j.out

set -euo pipefail
trap '[[ ${TMPDIR:-unset} == unset ]] || rm -rf $TMPDIR' EXIT
trap 'exit 143' TERM INT
: "${TMPDIR:?no node-local TMPDIR}"
GW_FILE=/home/byrong/_staging/TSN/fs1_gateway.txt
SECRETS=/home/byrong/.config/tsn_pilot_rsync/secrets
CONF=$TMPDIR/rsyncd.conf
mkdir -p "$(dirname "$GW_FILE")"
PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1])")
cat > "$CONF" <<EOF
use chroot = no
max connections = 16
pid file = $TMPDIR/rsyncd.pid
lock file = $TMPDIR/rsyncd.lock
log file = /scratch/byrong/TSN/fusion_stage1/logs/gateway_${SLURM_JOB_ID}_rsyncd.log
reverse lookup = yes
hosts allow = 160.45.4.200 g017 g017.* g017-*
hosts deny = *
auth users = pilot
secrets file = $SECRETS
strict modes = yes
[data]
    path = /scratch/byrong/VST_data/VST_datasets
    read only = yes
[fs1]
    path = /scratch/byrong/TSN/fusion_stage1
    read only = no
EOF
echo "$(hostname -f) $PORT" > "$GW_FILE.tmp"
mv "$GW_FILE.tmp" "$GW_FILE"
echo "[gateway] $(hostname -f):$PORT up $(date -Is)"
rsync --daemon --no-detach --port="$PORT" --config="$CONF" &
wait $!
