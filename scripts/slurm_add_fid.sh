#!/bin/bash
#SBATCH --job-name=mls_fid
#SBATCH --partition=sganguli
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=results/mls_inception/logs/%x_%j.out
# Usage: sbatch [--dependency=afterok:<id>] scripts/slurm_add_fid.sh <args for scripts/add_fid.py>
# Computes FID/KID from the samples each eval already saved, so it never regenerates anything.
ml purge
ml devel math python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312 py-scipy/1.12.0_py312 py-numpy/1.26.3_py312
source $HOME/diffusion_venv_py312/bin/activate
export TORCH_HOME=$SCRATCH/.cache/torch
cd $HOME/convolutional_diffusion
python scripts/add_fid.py "$@"
