#!/bin/bash
# Create and pin the finzi_1to1 output leaf directories to OSTs that answered a
# 4 MB write+fsync probe the same day (see memory: lustre-ost-speed-variance).
# Every leaf gets its own `lfs setstripe -c 1 -i`, because -o/-i are not
# inherited by subdirectories created later.
#
#   bash slurm/finzi_1to1_pin_dirs.sh "30 31 32 35 37" dir1 dir2 ...
set -uo pipefail
read -r -a OSTS <<< "$1"; shift
k=0
for d in "$@"; do
  mkdir -p "$d"
  o=${OSTS[$((k % ${#OSTS[@]}))]}
  if lfs setstripe -c 1 -i "$o" "$d"; then
    echo "pinned $d -> OST $o"
  else
    echo "setstripe FAILED for $d (OST $o)" >&2
  fi
  k=$((k + 1))
done
