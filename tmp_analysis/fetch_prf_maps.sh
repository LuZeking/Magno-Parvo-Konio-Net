#!/bin/bash
# Fetch NSD's continuous pRF maps for all 8 subjects (§7.2 item 2).
#
# Lands in the existing NSD tree on /scratch, where nsd_data.py already looks
# and where the rest of the dataset lives -- not /home, which has the quota
# that truncates silently.
#
# Each file is verified against the server's Content-Length after download. A
# truncated nii.gz would decompress to garbage or fail late in an analysis,
# and a silent partial download is exactly the class of failure this project
# has been burned by; better to fail here, loudly, on 1.4 MB.
set -uo pipefail

BASE="https://natural-scenes-dataset.s3.amazonaws.com/nsddata/ppdata"
DEST="/scratch/byrong/encoding/datasets/nsd/nsddata/ppdata"
VOLS="prf_eccentricity prf_size prf_angle prf_R2"

fail=0
for s in 01 02 03 04 05 06 07 08; do
    d="$DEST/subj$s/func1pt8mm"
    mkdir -p "$d"
    for v in $VOLS; do
        url="$BASE/subj$s/func1pt8mm/${v}.nii.gz"
        out="$d/${v}.nii.gz"
        want=$(curl -sI --max-time 60 "$url" | grep -i '^content-length' \
               | tr -d '\r' | awk '{print $2}')
        if [ -z "$want" ]; then
            echo "subj$s $v: NO CONTENT-LENGTH (url unreachable?)"
            fail=1; continue
        fi
        if [ -f "$out" ] && [ "$(stat -c%s "$out")" = "$want" ]; then
            echo "subj$s $v: already present ($want bytes)"
            continue
        fi
        curl -s --max-time 300 -o "$out" "$url"
        got=$(stat -c%s "$out" 2>/dev/null || echo 0)
        if [ "$got" != "$want" ]; then
            echo "subj$s $v: SIZE MISMATCH got=$got want=$want"
            fail=1; continue
        fi
        # a valid .nii.gz must at least be readable gzip
        if ! gzip -t "$out" 2>/dev/null; then
            echo "subj$s $v: NOT VALID GZIP"
            fail=1; continue
        fi
        echo "subj$s $v: ok ($got bytes)"
    done
done

echo "---"
du -sh "$DEST" 2>/dev/null
if [ "$fail" -ne 0 ]; then
    echo "DOWNLOAD INCOMPLETE -- do not run the analysis on this"
    exit 1
fi
echo "ALL 32 VOLUMES VERIFIED"
