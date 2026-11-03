#!/bin/bash
#SBATCH --job-name=rhm_ceil_L4
#SBATCH --partition=sganguli
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/rhm_ceiling_L4_%j.out

# The L=4 endpoint of the depth law, which OOM-killed inside 50861297: j_max was
# taken over every requested k, so the full-window k=31 (j*=L, ceiling 1 by
# definition) dragged it up to L-1 and the run tried to enumerate
# |R_3| = v m^21 = 4e10 renderings. rhm_ceiling_exact.py now takes j_max over the
# k that actually need a pool, and k=15 (j*=3 at L=4) is dropped as genuinely
# unenumerable. The L=1,2,3 stages of 50861297 wrote correctly and are not redone.
#
# The row worth having: L=4, k=7 has U=5, the same U as L=3, k=3, so the two
# ceilings must agree exactly -- the sharpest test that C depends on U alone.

set -euo pipefail
cd "$HOME/convolutional_diffusion"

ml purge
ml devel math python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312 py-numpy/1.26.3_py312
source "$HOME/diffusion_venv_py312/bin/activate"

OUT=results/rhm_sampling/2026-11-02
git rev-parse --short HEAD > "$OUT/provenance_ceiling_L4.txt"

python scripts/rhm_ceiling_exact.py --v 2 4 --m 3 --L 4 --k 3 7 31 \
    --rule_seeds 3 --mc 400000 --out "$OUT/ceiling_L4.json"

echo "done $(date)"
