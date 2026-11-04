#!/bin/bash
#SBATCH --job-name=rhm_ent2d
#SBATCH --partition=sganguli
#SBATCH --time=02:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch/users/hshunt/convolutional_diffusion/logs/rhm_ent2d_%j.out

# 2D entropy, second attempt.  The first (51073161) selected the 8 highest-
# success peaks, a subset whose published x* median is 0.088 against the L=2
# population's 0.193, and spent most of its wall clock on four L=3 cells whose
# P* is 2e5-1.3e6 -- far above the ladder, so they could never bracket.
# This takes all 25 bracketed L=2 cells, whose P* all sit under 3685, on a
# ladder to 16384 that brackets every one of them.

set -euo pipefail
cd /home/users/hshunt/convolutional_diffusion
ml purge
ml devel math python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312 py-numpy/1.26.3_py312
source "$HOME/diffusion_venv_py312/bin/activate"

python scripts/rhm_entropy_2d.py \
  --L 2 --max_cells 25 \
  --n_gen 2048 --gen_batch 1024 \
  --P_min 1 --P_max 16384 \
  --out results/rhm_entropy/2026-11-04/entropy_2d_L2.json
