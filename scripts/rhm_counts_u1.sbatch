#!/bin/bash
#SBATCH --job-name=rhm_counts_u1
#SBATCH --partition=sganguli
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/rhm_counts_u1_%j.out

# Cost leg of the U=1 family, which is the one family in which the LS machine
# actually succeeds at more than one depth.
#
# U = (s^(2(L-j*)) - 1)/(s^2 - 1) = 1 means j* = L-1, i.e. k = 2 s^(L-1) - 1:
# (L=2, k=3), (L=3, k=7), (L=4, k=15).  The ceiling is f0 = m/v^(s^2-1) at every
# member, so success stays Theta(1) as the grammar deepens and the only thing
# that moves is the data cost N_k.  That makes P*(L) along this family the
# experimental sampling complexity, free of the ceiling collapsing to zero.
#
# This job measures N_k(P) only (distinct windows seen; no generation), to size
# the P ladder for the generation sweep that follows.  v=2,3 and m=2,3 keep N_7
# small enough that the coupon-collector threshold is reachable at all.

set -euo pipefail
cd "$HOME/convolutional_diffusion"

ml purge
ml devel math python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312 py-numpy/1.26.3_py312
source "$HOME/diffusion_venv_py312/bin/activate"

OUT=results/rhm_sampling/2026-11-03
mkdir -p "$OUT" logs
git rev-parse --short HEAD > "$OUT/provenance_counts_u1.txt"

python scripts/rhm_patch_counts.py --v 2 3 --m 2 3 --L 2 --k 3 \
    --P 100 300 1000 3000 10000 30000 100000 300000 \
    --rule_seeds 3 --out "$OUT/counts_u1_L2.json"

python scripts/rhm_patch_counts.py --v 2 3 --m 2 3 --L 3 --k 7 \
    --P 100 300 1000 3000 10000 30000 100000 300000 \
    --rule_seeds 3 --out "$OUT/counts_u1_L3.json"

echo "done $(date)"
