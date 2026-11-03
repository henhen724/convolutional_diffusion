#!/bin/bash
#SBATCH --job-name=rhm_sweep
#SBATCH --partition=sganguli
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --output=logs/rhm_sweep_%j.out

# Sampling complexity of the LS machine on the RHM: how many training grids
# before generated grids are simultaneously novel and grammatical.
#
# A window of side k centred on a site reaches only k // 2, so it contains the
# whole aligned block of side b through that site only when k >= 2b - 1. For
# s = 2, L = 3 (an 8x8 grid) that makes k = 3 enough for the 2x2 leaf blocks,
# k = 7 enough for the 4x4 blocks, and k = 15 = 2d - 1 the full-window
# memorization control. Stage A walks that ladder; B-D vary one RHM parameter
# at a time at the cheap end of it.

set -euo pipefail
cd "$HOME/convolutional_diffusion"

ml purge
ml devel math python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312 py-numpy/1.26.3_py312
source "$HOME/diffusion_venv_py312/bin/activate"

OUT=results/rhm_sampling/$(date +%Y-%m-%d)
mkdir -p "$OUT" logs
git rev-parse --short HEAD > "$OUT/provenance_sweep.txt"

python -m pytest tests/test_rhm_sampling.py -q --no-cov -p no:cacheprovider

PS="30 100 300 1000 3000 10000 30000 100000"
COMMON="--n_gen 128 --alpha_smooth 1e-3 --max_patches 1200000"

# A1: the window ladder at the reference grammar, phase-aware.
python scripts/rhm_sampling_complexity.py --v 4 --s 2 --L 3 --m 3 \
    --k 3 5 7 --block_align 2 --P $PS --rule_seeds 2 $COMMON \
    --out "$OUT/A1_window.json"

# A2: the repo's own machine -- translation-invariant, zero-padded -- against
# the phase-aware one, to price the two assumptions separately.
python scripts/rhm_sampling_complexity.py --v 4 --s 2 --L 3 --m 3 \
    --k 3 5 --block_align 1 --P $PS --rule_seeds 2 $COMMON \
    --out "$OUT/A2_pooled.json"
python scripts/rhm_sampling_complexity.py --v 4 --s 2 --L 3 --m 3 \
    --k 3 5 --block_align 1 --pad_mode zeros --P $PS --rule_seeds 2 $COMMON \
    --out "$OUT/A3_repo_machine.json"

# A4: memorization ceiling, k = 2d - 1. Expect grammatical 1, novel 0.
python scripts/rhm_sampling_complexity.py --v 4 --s 2 --L 3 --m 3 \
    --k 15 --block_align 1 --P 100 300 1000 --rule_seeds 1 --n_gen 128 \
    --alpha_smooth 1e-3 --max_patches 1200000 --out "$OUT/A4_memorize.json"

# B: vocabulary size, at the window the data can fill.
python scripts/rhm_sampling_complexity.py --v 2 --s 2 --L 3 --m 3 \
    --k 3 --block_align 2 --P $PS --rule_seeds 3 $COMMON --out "$OUT/B_v2.json"
python scripts/rhm_sampling_complexity.py --v 3 --s 2 --L 3 --m 3 \
    --k 3 --block_align 2 --P $PS --rule_seeds 3 $COMMON --out "$OUT/B_v3.json"
python scripts/rhm_sampling_complexity.py --v 6 --s 2 --L 3 --m 3 \
    --k 3 --block_align 2 --P $PS --rule_seeds 3 $COMMON --out "$OUT/B_v6.json"
python scripts/rhm_sampling_complexity.py --v 8 --s 2 --L 3 --m 3 \
    --k 3 --block_align 2 --P $PS --rule_seeds 3 $COMMON --out "$OUT/B_v8.json"

# C: rules per symbol.
for M in 2 4 5; do
  python scripts/rhm_sampling_complexity.py --v 4 --s 2 --L 3 --m $M \
      --k 3 --block_align 2 --P $PS --rule_seeds 3 $COMMON --out "$OUT/C_m$M.json"
done

# D: depth. L=2 is 4x4, where k=7 is already the full window.
python scripts/rhm_sampling_complexity.py --v 4 --s 2 --L 2 --m 3 \
    --k 3 5 7 --block_align 2 --P $PS --rule_seeds 3 $COMMON --out "$OUT/D_L2.json"

echo "done $(date)"
