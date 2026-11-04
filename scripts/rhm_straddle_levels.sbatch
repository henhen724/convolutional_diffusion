#!/bin/bash
#SBATCH --job-name=rhm_strad_lv
#SBATCH --partition=sganguli
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=logs/rhm_straddle_levels_%j.out

# How many levels up the tree can a straddling window see?
#
# Three stages, cheapest first:
#   1-2. L=2, v=2 and v=3 separately so each can be cross-checked against the
#        exact enumeration in straddle_v{2,3}.json (a_{j*} must reproduce
#        lv_pooled / lv_blocks).  This is the validation, not the result.
#   3.   L=3, where the question actually lives: with j* = 1, does the window
#        still reject grids that are coherent through level 2?  k is capped at
#        5 because a k=7 window code would overflow int64 (the script refuses).
#
# The window set is grown until it stops changing, so k=3 cells cost ~25 s and
# the budget goes to the k=5 cells that need millions of grids.  a_L = 1 is the
# saturation test and is recorded per cell.

set -euo pipefail
cd "$HOME/convolutional_diffusion"

ml purge
ml devel math python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312 py-numpy/1.26.3_py312
source "$HOME/diffusion_venv_py312/bin/activate"

OUT=results/rhm_sampling/2026-11-03
mkdir -p "$OUT" logs

for V in 2 3; do
  python scripts/rhm_straddle_levels.py --v $V --m 2 3 4 5 6 --L 2 --k 3 5 \
      --rule_seeds 5 --n_test 200000 --n_wset 2000000 --chunk 50000 \
      --check "$OUT/straddle_v$V.json" --out "$OUT/straddle_levels_L2_v$V.json"
done

python scripts/rhm_straddle_levels.py --v 2 3 4 --m 2 3 --L 3 --k 3 5 \
    --rule_seeds 5 --n_test 200000 --n_wset 6000000 --chunk 50000 \
    --out "$OUT/straddle_levels_L3.json"

echo "done $(date)"
