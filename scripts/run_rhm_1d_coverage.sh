#!/bin/bash
#SBATCH --job-name=rhm1d_cov
#SBATCH --partition=sganguli
#SBATCH --time=04:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/scratch/users/hshunt/convolutional_diffusion/logs/rhm1d_cov_%j.out

# Does coverage explain the 1D/2D gap in x* = P*/|R_L| (0.050 vs 0.193 at n=16)?
# Sweep the window k at *fixed* n -- the lever neither depth sweep could pull,
# since there k and n moved together.  L=4 (n=16) runs k = 3..17, covering
# 0.19..1.0, which brackets 2D's whole range; k=9 sits at 9/16, exactly 2D
# L=2's coverage, so the cross-dimension prediction is a single number.
#
# 32G, not the depth probe's 64G: that job's measurement phase peaked at ~1 GB
# host and 1.2 GB GPU, and the OOM was in exact_RL, which is now capped. The
# new cost here is the k=17 table, bounded by --max_patches.

set -euo pipefail
cd /home/users/hshunt/convolutional_diffusion

ml purge
ml devel math python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312 py-numpy/1.26.3_py312
source "$HOME/diffusion_venv_py312/bin/activate"

OUT=results/rhm_1d/2026-11-04
mkdir -p "$OUT" logs

python scripts/rhm_1d_coverage.py \
  --L 3 4 \
  --v 2 3 \
  --m 2 3 4 \
  --s 2 \
  --rule_seeds 3 \
  --n_gen 2048 \
  --gen_batch 1024 \
  --P_min 1 --P_max 262144 \
  --R_min 200 --R_max 1e5 \
  --max_patches 3000000 \
  --out "$OUT/coverage.json"
