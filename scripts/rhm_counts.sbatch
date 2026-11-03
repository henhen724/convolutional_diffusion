#!/bin/bash
#SBATCH --job-name=rhm_counts
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/rhm_counts_%j.out

# N_k(P): how many distinct k x k windows the RHM emits, and how fast the
# training set fills them in. This is the table the LS machine's posterior is
# indexed by, so where it saturates is where the machine stops being data-bound.
# Pure numpy, no sampler, so P can go orders of magnitude beyond the generation
# sweep. Tests A_j = v * m**((s**(2j)-1)/(s**2-1)) -- linear in v, power of m.

set -euo pipefail
cd "$HOME/convolutional_diffusion"

ml purge
ml devel math python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312 py-numpy/1.26.3_py312
source "$HOME/diffusion_venv_py312/bin/activate"

OUT=results/rhm_sampling/$(date +%Y-%m-%d)
mkdir -p "$OUT" logs
git rev-parse --short HEAD > "$OUT/provenance.txt"

SMALLP="100 300 1000 3000 10000 30000 100000 300000"
BIGP="100 300 1000 3000 10000 30000"

# m-dependence at fixed v, both depths, small windows (cheap, wide P range).
python scripts/rhm_patch_counts.py --v 4 --s 2 --L 2 3 --m 2 3 4 5 --k 3 5 \
    --P $SMALLP --rule_seeds 3 --out "$OUT/counts_m.json"

# v-dependence at fixed m.
python scripts/rhm_patch_counts.py --v 2 3 6 8 --s 2 --L 2 3 --m 3 --k 3 5 \
    --P $SMALLP --rule_seeds 3 --out "$OUT/counts_v.json"

# the larger windows, where the table is big and P cannot be pushed as far.
python scripts/rhm_patch_counts.py --v 4 --s 2 --L 3 --m 2 3 --k 7 \
    --P $BIGP --rule_seeds 2 --out "$OUT/counts_k7.json"

echo "done $(date)"
