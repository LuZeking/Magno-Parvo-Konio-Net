#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes 1
#SBATCH --mem 100GB
#SBATCH -c 8
#SBATCH -p klab-gpu
#SBATCH --qos klab
#SBATCH --gres=gpu:1
#SBATCH --job-name=final_ts_dkl_contrast
#SBATCH --error=final_ts_dkl_contrast.error.o%j
#SBATCH --output=final_ts_dkl_contrast.output.o%j
#SBATCH --signal=SIGTERM@180

echo "running in shell: " "$SHELL"
export NCCL_SOCKET_IFNAME=lo
export TMPDIR='/share/klab/lzejin/mbenito/two_stream_dvd/tmp_files'

spack load cuda@11.8.0
spack load cudnn@8.6.0.163-11.8
spack load miniconda3
eval "$(conda shell.bash hook)"
conda activate hpc-env


srun --unbuffered python3 -m scripts.train_dvd --config-name "scenario_train_dvd_supervised"
