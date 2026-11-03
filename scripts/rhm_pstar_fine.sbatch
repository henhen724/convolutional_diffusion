#!/bin/bash
#SBATCH --job-name=rhm_pstar
#SBATCH --partition=sganguli
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/rhm_pstar_%j.out

# EXPERIMENTAL sampling complexity P*(v, m) at fixed depth, measured properly.
#
# The 2026-11-02 sweeps located P* on a 7-point ladder spaced by 3x and 3 rule
# seeds, which pins P* only to within half a decade: the fit of ln P* against
# ln |R_L| came out at slope 0.78 with a residual factor of 2.1, too loose to
# call it the predicted 1. This run puts a 25-point ladder (1.5x spacing) and 5
# rule seeds under the same nine grammars, with n_gen 8192 so that the smallest
# peak (succ* ~ 0.006 at v=8) is still ~50 hits rather than ~20.
#
# L=2, k=3 throughout: j*=1 there, so U=1 and the ceiling is f0 = m/v^(s^2-1),
# large enough to measure. k=5 is strictly worse at the same j*, and k=7 = 2d-1
# is the full window where novelty is identically 0, so k=3 is the best window
# for every grammar here and P*(k=3) is the sampling complexity.
#
# Prediction under test: P* = Theta(|R_L|) = Theta(v m^((n-1)/(s^2-1))), i.e.
# ln P* = ln v + 5 ln m at L=2, s=2.
#
# block_align 1 is the measurement and 2 is only a control. The LS machine is
# translation invariant: one pooled table, no knowledge of where the RHM's block
# grid sits. block_align=2 hands it the phase of every site, which is strictly
# more than it has. The 2026-11-02 sweeps used block_align=2 as primary, so every
# P* in them is the privileged machine's; this run redoes them with the real one
# on the same seeds and ladder, so the two differ in nothing but the bookkeeping.

set -euo pipefail
cd "$HOME/convolutional_diffusion"

ml purge
ml devel math python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312 py-numpy/1.26.3_py312
source "$HOME/diffusion_venv_py312/bin/activate"

OUT=results/rhm_sampling/2026-11-03
mkdir -p "$OUT" logs
git rev-parse --short HEAD > "$OUT/provenance_pstar.txt"

LADDER="10 15 20 30 50 70 100 150 200 300 500 700 1000 1500 2000 3000 5000 7000 10000 15000 20000 30000 50000 70000 100000"

for v in 2 3 4 6 8; do
    python scripts/rhm_sampling_complexity.py --v $v --m 3 --L 2 --k 3 \
        --P $LADDER --n_gen 8192 --gen_batch 4096 --rule_seeds 5 \
        --block_align 1 2 --pad_mode outside --out "$OUT/P_v${v}_m3.json"
done

for m in 2 4 5 6; do
    python scripts/rhm_sampling_complexity.py --v 4 --m $m --L 2 --k 3 \
        --P $LADDER --n_gen 8192 --gen_batch 4096 --rule_seeds 5 \
        --block_align 1 2 --pad_mode outside --out "$OUT/P_v4_m${m}.json"
done

echo "done $(date)"
