#!/bin/bash
# OOD robustness for thesis_dvd_trans -- the arm with an actual PROGRESSIVE age
# transition, as opposed to this paper's static 48-month front end.
#
# WHY. §2.6.1 concludes that "a developmental route to comparable shape bias
# delivers NONE of the robustness". The diet it tests is static (§4.1: "a fixed
# age of 48 months... Age is fixed rather than scheduled"), but the active
# ingredient of the developmental-diet result it is implicitly contradicting
# (Lu et al. 2026, Nat Mach Intell) is the ORDERING of training. So as it stands
# §2.6.1 may be a failed replication of the closest prior work, mislabelled as a
# dissociation. thesis_dvd_trans has the ordering and has never been run through
# the OOD battery -- it appears in results/shape_texture and results/encoding but
# not in results/ood or results/ood_bestep.
#
# ONE job, so no array (the array rule applies at N>1). CPU only -- `main`,
# no --gres, matching slurm/ood_robustness_bestep.sh.
#SBATCH --job-name=mpk_ood_dvdtrans
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/ood_dvdtrans_%j.out

set -euo pipefail
mkdir -p slurm/logs results/ood_bestep
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
PY=/scratch/byrong/anaconda3/envs/mpk/bin/python
REPO=$SLURM_SUBMIT_DIR
cd "$REPO"

# Lightning checkpoint from the thesis codebase; diet on luminance, so the same
# front-end flags as dkl_bcs (none), per slurm/shape_texture.sh:47.
CKPT=/scratch/byrong/tsn_checkpoints/dkl_blur_cs_dvd_trans/epoch=147-step=152292.ckpt
[[ -f "$CKPT" ]] || { echo "FATAL: missing $CKPT"; exit 1; }
echo "[thesis_dvd_trans] progressive-age-transition arm -> $CKPT"

PYTHONPATH=. $PY tmp_analysis/ood_robustness.py \
    --model-name thesis_dvd_trans \
    --checkpoint "$CKPT" \
    --out-dir "$REPO/results/ood_bestep" \
    --stream1-weight 0.5
