#!/bin/bash
#SBATCH --job-name=rhm_ceiling
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/rhm_ceiling_%j.out

# The grammaticality ceiling of a k x k LS machine, with no reference to any
# training set: the fraction of the locally-valid set that is actually
# grammatical,
#     C(k) = |R_L| / |R_j*| ** (s ** (2 * (L - j*))),
# exactly by enumeration where R_L is small enough and by Monte Carlo over
# uniform draws from R_j* otherwise. This is the curve the measured plateaus of
# rhm_sweep2.sbatch are compared against; it is derivation, not measurement, so
# it is the half of the result that does not depend on how much data the machine
# got. Validated against the closed form f0**U, f0 = m / v**(s**2-1).

set -euo pipefail
cd "$HOME/convolutional_diffusion"

ml purge
ml devel math python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312 py-numpy/1.26.3_py312
source "$HOME/diffusion_venv_py312/bin/activate"

OUT=results/rhm_sampling/$(date +%Y-%m-%d)
mkdir -p "$OUT" logs
git rev-parse --short HEAD > "$OUT/provenance_ceiling.txt"

# L=2: R_L is enumerable, so the ratio is exact and the MC is a check on it.
python scripts/rhm_ceiling_exact.py --v 2 3 4 6 8 --m 2 3 4 5 6 --L 2 --k 3 5 7 \
    --rule_seeds 3 --mc 200000 --enumerate_top --out "$OUT/ceiling_L2.json"

# L=3: R_L ~ v * m**21 is far too large to enumerate; MC only.
python scripts/rhm_ceiling_exact.py --v 2 3 4 6 8 --m 2 3 4 5 --L 3 --k 3 7 15 \
    --rule_seeds 3 --mc 400000 --out "$OUT/ceiling_L3.json"

# L=1 and L=4 pin the two ends of the depth law at the reference grammar.
python scripts/rhm_ceiling_exact.py --v 2 4 --m 3 --L 1 --k 3 \
    --rule_seeds 3 --mc 200000 --enumerate_top --out "$OUT/ceiling_L1.json"
# k=15 is dropped at L=4: j*=3 there and |R_3| = v m^21 cannot be enumerated.
# k=31 is the full window (j*=L), so it costs nothing.
python scripts/rhm_ceiling_exact.py --v 2 4 --m 3 --L 4 --k 3 7 31 \
    --rule_seeds 3 --mc 400000 --out "$OUT/ceiling_L4.json"

echo "done $(date)"
