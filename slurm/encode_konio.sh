#!/bin/bash
# The konio arm: is the null colour result a failed colour prediction, or is
# L-M simply the wrong model of colour cortex?
#
#   sbatch slurm/encode_konio.sh
#
# WHY THIS ARM (PAPER_PLAN B3 / §65). §7.1 attached the Kastner2015 atlas to
# the existing fits and found the colour prediction fails at VO: median
# (Um-Up) = +0.0044, p = .46, i.e. VO is if anything MAGNO-preferring. hV4
# (-0.012, 2/8) is the only parvo-preferring area in 20 ROIs. The repository
# is called Magno-Parvo-Konio-Net and the third channel has never been run.
#
# The DKL dataset already stores all three opponent axes (lum, rg, by -- see
# dkl_frontend.rgb_to_dkl3), so the koniocellular S-(L+M) input needs NO new
# data conversion and NO training: an untrained stream is the reference model
# for the colour claim anyway (§8 item 4). Cost is one array, no GPU.
#
# WHAT CHANGES, AND ONLY THIS. Stream 2 receives the by channel instead of
# rg. Same untrained network at the same seed (weights bit-identical --
# verified), same luminance stream (blur sigma=4 + CS cut at 48 months, the
# defaults `untrained_seed0` was submitted with), same architecture, same CV
# splits, same nsdgeneral mask. tmp_analysis/smoke_konio.py verified before
# submission that the magno FEATURE MATRIX is bit-identical across the two
# arms, so any change in unique chromatic variance cannot come from the
# luminance side.
#
# THE CONTRAST CONFOUND, MEASURED FIRST -- and it points the wrong way.
# Every DKL channel is normalised to [0,1] with its OWN global min/max, and
# those spans differ (rg 1.048, by 8.189 of cone contrast). That does not
# equalise anything. On 100 shared-515 NSD images the by channel arrives with
# 2.05x rg's per-image RMS contrast (0.0941 vs 0.0460) and 3.5x its absolute
# power above 1/8 cyc/px. So an unmatched konio WIN would be uninterpretable:
# chromatic axis and input contrast confounded, with the confound flattering
# the new arm. Hence two cells, and the asymmetry is the point:
#
#   cell 0  untrained_konio_matched   by, per-image RMS matched to rg
#           -> the PRIMARY arm. The only remaining difference from the
#              published parvo fit is which chromatic axis stream 2 carries.
#   cell 1  untrained_konio_raw       by as the pipeline delivers it
#           -> the sensitivity check. A LOSS here is decisive (konio fails
#              WITH a 2x contrast advantage); a WIN here proves nothing on
#              its own and must be read against cell 0.
#
# The reference cell needs no run: `untrained_seed0` (rg, 8 subjects) is
# already on disk in results/encoding, produced by
# `sbatch slurm/encode_subjects.sh untrained_seed0` with no extra flags
# (recovered via sacct SubmitLine), which is exactly the front end used here.
#
# NOT DONE, DELIBERATELY: no blur on the konio stream. B3 specified "heavy
# blur" from the K pathway's low spatial acuity, but the measured radial
# power spectrum of this pipeline's by channel has almost the same SHAPE as
# rg's (0.614 vs 0.570 of power in the lowest octave band). The low-acuity
# property is not in the input, so blurring would impose an assumption rather
# than reproduce a data property, and it would break the single-factor swap.
# Reported as a limitation in §65.7, not smuggled in as a free parameter.
#
# ONE ARRAY, NOT TWO SBATCHES: 2 cells x 8 subjects = 16 tasks, indexed off
# $SLURM_ARRAY_TASK_ID per project policy. No CUDA is used (the fits are
# primal banded ridge over ~1k features per band), so this runs on `main`
# with NO --gres. Resources copied from slurm/encode_subjects.sh, whose
# header records the measurements behind them: 160G because the measured peak
# with chunk=all is 137G, and 160G (163840 MB) fits on 68 nodes where 190G
# fits on only 18.
#SBATCH --job-name=mpk_konio
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --array=0-15%4
#SBATCH --output=slurm/logs/konio_%A_%a.out

set -euo pipefail
mkdir -p slurm/logs
export OPENBLAS_NUM_THREADS=16 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16

CELL=$((SLURM_ARRAY_TASK_ID / 8))
SUBJECT=$((SLURM_ARRAY_TASK_ID % 8 + 1))

case "$CELL" in
    0) MODEL_NAME=untrained_konio_matched
       EXTRA=(--chroma-channel by --chroma-rescale match-rg) ;;
    1) MODEL_NAME=untrained_konio_raw
       EXTRA=(--chroma-channel by) ;;
    *) echo "bad cell $CELL"; exit 1 ;;
esac

PY=/scratch/byrong/anaconda3/envs/mpk/bin/python
REPO=$SLURM_SUBMIT_DIR
OUT=$REPO/results/encoding
# Regenerable and ~1.2 GB per subject per model, so it must not live in the
# repo: $REPO is on quota'd /home and a cache write there once killed a run
# with `Errno 122 Disk quota exceeded` mid-fit.
CACHE_DIR=${MPK_FEATURE_CACHE:-/scratch/byrong/VST_data/mpk_feature_cache}
CACHE=$CACHE_DIR/${MODEL_NAME}_subj$(printf %02d "$SUBJECT").npz

cd "$REPO"
echo "[konio] task=$SLURM_ARRAY_TASK_ID cell=$CELL model=$MODEL_NAME " \
     "subj=$SUBJECT on $(hostname)"

# No --checkpoint: this arm is untrained on purpose. The models were never
# trained on a by channel, so there is no konio checkpoint to load and none
# is implied -- see §65.7 on what this arm therefore cannot claim.
$PY -m src.analysis.encoding.run_encoding \
    --subject "$SUBJECT" \
    --model-name "$MODEL_NAME" \
    --out-dir "$OUT" \
    --feature-cache "$CACHE" \
    "${EXTRA[@]}"

echo "[konio] done $MODEL_NAME subj$SUBJECT"
