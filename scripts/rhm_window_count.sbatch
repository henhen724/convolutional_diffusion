#!/bin/bash
#SBATCH --job-name=rhm_nk
#SBATCH --partition=sganguli
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/rhm_nk_%j.out

# The cost leg, exactly. With the U=1 ceiling now measured at Theta(1) rather
# than f0^U, what sets the sampling complexity is coupon collection over N_k,
# the saturated number of distinct k x k windows the grammar can emit.
#
# Sampling does not reach it at depth -- pooled N_7 at L=3 was still climbing at
# P=3e5 (job 50895001) -- but the grammar is finite, so N_k can be counted over
# the whole of R_L wherever R_L is enumerable. At L=3 that is v m^21: 4.2e6
# grids at v=2,m=2 and 6.3e6 at v=3,m=2. The enumerator reproduces the sampled
# saturation values at L=2 exactly (274, 526, 475, 1890), which is the check
# that it agrees with the machine's own table.
#
# k=7 is the U=1 member at L=3 (k = 2 s^(L-1) - 1), so N_3 at L=2 and N_7 at
# L=3 are the two depth points of the family in which the ceiling stays Theta(1)
# and only the data cost moves. k=3 and k=5 come along for the ladder.

set -euo pipefail
cd "$HOME/convolutional_diffusion"

ml purge
ml devel math python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312 py-numpy/1.26.3_py312
source "$HOME/diffusion_venv_py312/bin/activate"

OUT=results/rhm_sampling/2026-11-03
mkdir -p "$OUT" logs
git rev-parse --short HEAD > "$OUT/provenance_nk.txt"

# L=1 and L=2: every grammar is free, so the family has its small-L anchors
# across v and m, not just at m=2.
python scripts/rhm_window_count_exact.py --v 2 3 4 6 8 --m 2 3 4 5 6 --L 2 --k 1 3 5 \
    --rule_seeds 3 --max_grids 2e7 --out "$OUT/nk_exact_L2.json"

# L=3, m=2 only: v m^21 is 4.2e6 at v=2 and 6.3e6 at v=3, and 3^21 is already
# out of reach at m=3.
python scripts/rhm_window_count_exact.py --v 2 3 --m 2 --L 3 --k 3 5 7 \
    --rule_seeds 3 --max_grids 2e7 --out "$OUT/nk_exact_L3.json"

echo "done $(date)"
