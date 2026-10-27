#!/bin/bash
# Resubmit ONE timed-out CTD encode element unchanged — same slurm/ctd_encode.sh, same CODE snapshot, same
# settings — with a 30 h wall limit, and extend the completeness gate's afterany dependency so the analysis
# keeps waiting for it. This is a submit helper (run in the login/allocation shell), not a Slurm script.
#
#   slurm/ctd_rerun_element.sh <task_id> [<gate_jobid>]        e.g.  slurm/ctd_rerun_element.sh 25
#   DRY=1 slurm/ctd_rerun_element.sh 25                         print the checks and the sbatch line, submit nothing
#   env: CODE (default snapshot 376fb05), EXCL (nodes to exclude), TIME (default 30:00:00)
#
# Refuses to submit while any element with this task id is still PENDING/RUNNING in the gate's chain
# (two writers of the same output file), or if the output file already exists (the encode script would refuse
# to overwrite it anyway). Ledger: results/cortex_decision/EXECUTION_LEDGER.md §2a/§2b.
set -euo pipefail
TASK=${1:?usage: ctd_rerun_element.sh <task_id> [<gate_jobid>]}
GATE=${2:-26719988}
CODE=${CODE:-/home/byrong/projects/TSN/fusion_stage1_code/376fb05}
EXCL=${EXCL:-c116,c042,c162,c012}
TIME=${TIME:-30:00:00}
RES=/scratch/byrong/mpk/encoding/results/2026-10-26_ctd

# only the live Dependency= field — SubmitLine= also contains the original --dependency string
DEPS=$(scontrol show job "$GATE" | grep -oE '(^|[[:space:]])Dependency=[^[:space:]]+' | grep -oE 'afterany:[0-9]+' | cut -d: -f2 | awk '!seen[$0]++' | paste -sd,)
[[ -n $DEPS ]] || { echo "gate $GATE has no afterany dependencies (already ran?)"; exit 2; }
echo "[rerun] gate $GATE chain: $DEPS"

live=$(squeue -r -h -j "$DEPS" -o "%i %t %M %R" | awk -v t="_${TASK}" 'substr($1, length($1)-length(t)+1) == t')
if [[ -n $live ]]; then echo "[rerun] REFUSED — task $TASK still live:"; echo "$live"; exit 3; fi

# task -> run/subject mapping copied from slurm/ctd_encode.sh (GROUP = TASK/8, SUBJECT = TASK%8+1)
G=$((TASK/8)); SS=$(printf '%02d' $((TASK%8+1)))
case $G in
  0|1|2) RUN=fs2_dvd300_avg_w050_ls_seed$G ;;
  3|4|5) RUN=fs2_dvdfix300_avg_w050_ls_seed$((G-3)) ;;
  6|7|8) RUN=fs2_dvd300swap_avg_w050_ls_seed$((G-6)) ;;
  9)     RUN=untrained_dvd300_seed0 ;;
  10)    RUN=gabor_dvd300 ;;
  *) echo "bad task $TASK"; exit 2 ;;
esac
DEST="$RES/${RUN}_subj${SS}.npz"
echo "[rerun] task $TASK = $RUN subj$SS -> $DEST"
[[ -e $DEST ]] && { echo "[rerun] REFUSED — output already exists: $DEST"; exit 4; }

hist=$(sacct -j "$DEPS" -n -P --format=JobID,State,Elapsed,NodeList | grep -E "^[0-9]+_${TASK}\|" || true)
echo "[rerun] history of task $TASK:"; echo "${hist:-  (none)}"

CMD="env -u TMPDIR sbatch --parsable --export=ALL,CODE=$CODE --time=$TIME --exclude=$EXCL --array=$TASK slurm/ctd_encode.sh"
echo "[rerun] cd $CODE && $CMD"
if [[ ${DRY:-0} == 1 ]]; then echo "[rerun] DRY=1 — nothing submitted"; exit 0; fi
cd "$CODE"
NEW=$($CMD)
echo "[rerun] submitted $NEW"
scontrol update jobid="$GATE" Dependency="afterany:$(echo "$DEPS" | tr ',' ':'):$NEW"
echo "[rerun] gate $GATE now: $(scontrol show job "$GATE" | grep -oE '(^|[[:space:]])Dependency=[^[:space:]]+')"
