#!/bin/bash
#SBATCH --job-name=rhm_entropy
#SBATCH --partition=sganguli
#SBATCH --time=04:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=/scratch/users/hshunt/convolutional_diffusion/logs/rhm_entropy_%j.out

# Does a posterior-entropy budget set the window optimum and the 1D/2D gap?
# Prediction registered in the two scripts' docstrings: novelty dies where the
# trajectory entropy crosses the novelty floor, ln P = H_tot, so the effective
# support is exp(H_tot) and x* = exp(H_tot)/|R_L| with no fitted constant.
#
# Both dimensions in one job, deliberately: the 1D half alone runs in minutes,
# and the question is a cross-dimension one anyway.
#
# 4 cpus, not 8: sganguli was at 3 of 384 cores free this morning, so a wider
# request queues behind the GPU it actually needs.  48G against the 1D sweep's
# measured 829 MB peak -- the headroom is for the 2D L=3 k=7 patch table.

set -euo pipefail
cd /home/users/hshunt/convolutional_diffusion

ml purge
ml devel math python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312 py-numpy/1.26.3_py312
source "$HOME/diffusion_venv_py312/bin/activate"

OUT=results/rhm_entropy/2026-11-04
mkdir -p "$OUT" logs

echo "=============== 1D: k-sweep with trajectory entropy ==============="
python scripts/rhm_entropy_window.py \
  --L 3 4 --v 2 3 --m 2 3 4 --s 2 --rule_seeds 3 \
  --n_gen 2048 --gen_batch 1024 \
  --P_min 1 --P_max 262144 \
  --R_min 200 --R_max 1e5 --max_chance 0.05 \
  --max_patches 3000000 \
  --out "$OUT/entropy_1d.json"

echo "=============== 2D: published peaks, same instrument ==============="
python scripts/rhm_entropy_2d.py \
  --L 2 3 --max_cells 8 \
  --n_gen 2048 --gen_batch 512 \
  --P_min 1 --P_max 131072 \
  --max_patches 3000000 \
  --out "$OUT/entropy_2d.json"

echo "done"
