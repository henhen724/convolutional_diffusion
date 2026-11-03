#!/bin/bash
#SBATCH --job-name=rhm_straddle
#SBATCH --partition=sganguli
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/rhm_straddle_%j.out

# What the straddling windows are worth, measured exactly rather than assumed
# away. The ceiling used through 2026-11-02, C(k) = |R_L| / |R_j*|^n_blocks,
# counts a grid as locally valid when every *aligned* s^j* block is a legal
# rendering -- i.e. it assumes a window that spans a block boundary constrains
# nothing. hshunt's point is that it does: the boundary between two blocks can
# reveal that they descend from different parent words, and how often it does
# depends on the synonym rules.
#
# At L=2 the grid is 16 sites, so the whole space v^16 is enumerable and the
# locally-valid set of each machine can be *counted*. The smoke test at v=2
# already shows the assumption is wrong -- straddling shrinks the locally-valid
# set by 1.6x to 4.0x at k=3, and at v=2,m=2,seed1 it shrinks it to exactly
# |R_L|, ceiling 1.0 for a strictly local window. This run maps that across v
# and m to see what controls the gain.
#
# k=3 only at v=3: k=5 at L=2 is the memorization regime (a 5x5 window centred
# at (1,1) already covers the whole 4x4 grid), so its ceiling of 1 is trivial
# and it is kept only at v=2 as the documented boundary of that regime.

set -euo pipefail
cd "$HOME/convolutional_diffusion"

ml purge
ml devel math python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312 py-numpy/1.26.3_py312
source "$HOME/diffusion_venv_py312/bin/activate"

OUT=results/rhm_sampling/2026-11-03
mkdir -p "$OUT" logs
git rev-parse --short HEAD > "$OUT/provenance_straddle.txt"

# v=2: 65536 grids per grammar, so every m and both k are free.
python scripts/rhm_straddle.py --v 2 --m 2 3 4 5 6 --L 2 --k 3 5 \
    --rule_seeds 5 --chunk 1048576 --out "$OUT/straddle_v2.json"

# v=3: 43M grids per grammar, k=3 only.
python scripts/rhm_straddle.py --v 3 --m 2 3 4 5 6 --L 2 --k 3 \
    --rule_seeds 5 --chunk 1048576 --max_enum 1e8 --out "$OUT/straddle_v3.json"

echo "done $(date)"
