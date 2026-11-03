#!/bin/bash
#SBATCH --job-name=rhm_sweep2
#SBATCH --partition=sganguli
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --output=logs/rhm_sweep2_%j.out

# Sampling complexity of the LS machine on the RHM. Redesign of rhm_sweep.sbatch
# (job 50859792, cancelled): its v- and m-sweeps sat at L=3, k=3, where the
# predicted grammaticality ceiling is f^5 ~ 2e-7, so with n_gen=128 every cell
# reads exactly 0.000 and measures nothing.
#
# The theory being tested. A k x k window centred on a site reaches only k // 2,
# so it contains the aligned block of side b through that site only when
# k >= 2b - 1. Write j* = floor(log_s((k+1)/2)) for the deepest level it can
# police. The rules above j* are unconstrained, there are
#     U = (s^(2(L-j*)) - 1) / (s^2 - 1)
# of them, and a random s x s block of symbols is a word of some symbol with
# probability
#     f = 1 - (1 - v^(-s^2))^(v m)   (~ m / v^(s^2-1) when that is small).
# So grammaticality is capped at f^U *no matter how large P is*, and reaching
# that cap costs the coupon-collector time for the N_k distinct windows.
#
# Each stage below isolates one factor of that formula in a regime where it is
# actually resolvable: n_gen is set so the predicted ceiling is many times
# 1 / n_gen.

set -euo pipefail
cd "$HOME/convolutional_diffusion"

ml purge
ml devel math python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312 py-numpy/1.26.3_py312
source "$HOME/diffusion_venv_py312/bin/activate"

OUT=results/rhm_sampling/$(date +%Y-%m-%d)
mkdir -p "$OUT" logs
git rev-parse --short HEAD > "$OUT/provenance_sweep2.txt"

python -m pytest tests/test_rhm_sampling.py -q --no-cov -p no:cacheprovider

BIG="--alpha_smooth 1e-3 --max_patches 1200000 --gen_batch 2048"

# ---------------------------------------------------------------- A: window
# The k ladder at the reference grammar, L=3 (8x8). Ceiling is unresolvable at
# k=3 (U=5) but the per-level parse fractions are not, and they are what shows
# the window failing scale by scale. k=15 = 2d-1 is the memorization control.
PS="30 100 300 1000 3000 10000 30000 100000"
python scripts/rhm_sampling_complexity.py --v 4 --s 2 --L 3 --m 3 \
    --k 3 5 7 --block_align 2 --P $PS --rule_seeds 2 --n_gen 512 $BIG \
    --out "$OUT/A1_window.json"
python scripts/rhm_sampling_complexity.py --v 4 --s 2 --L 3 --m 3 \
    --k 15 --block_align 1 --P 100 300 1000 --rule_seeds 1 --n_gen 512 $BIG \
    --out "$OUT/A4_memorize.json"

# ------------------------------------------------- A2/A3: the two assumptions
# The repo's machine pools patches over all offsets and zero-pads; the RHM is
# block-aligned and has a hard edge. Price each separately against A1's k=3.
SPARSE="100 1000 10000 100000"
python scripts/rhm_sampling_complexity.py --v 4 --s 2 --L 3 --m 3 \
    --k 3 --block_align 1 --P $SPARSE --rule_seeds 2 --n_gen 512 $BIG \
    --out "$OUT/A2_pooled.json"
python scripts/rhm_sampling_complexity.py --v 4 --s 2 --L 3 --m 3 \
    --k 3 --block_align 1 --pad_mode zeros --P $SPARSE --rule_seeds 2 --n_gen 512 $BIG \
    --out "$OUT/A3_repo_machine.json"

# ------------------------------------------------------------- E: the ceiling
# L=2 (4x4), k=3: j*=1, so U=1 and the ceiling is f itself -- one unenforced
# rule, the root. Sweeping v and m here reads off f(v, m) directly and tests the
# v^-(s^2-1) = v^-3 exponent. n_gen=4096 resolves down to f ~ 0.006 (v=8).
EPS="30 100 300 1000 3000 10000 30000"
for V in 2 3 4 6 8; do
  python scripts/rhm_sampling_complexity.py --v $V --s 2 --L 2 --m 3 \
      --k 3 --block_align 2 --P $EPS --rule_seeds 3 --n_gen 4096 $BIG \
      --out "$OUT/E_v$V.json"
done
for M in 2 3 4 5 6; do
  python scripts/rhm_sampling_complexity.py --v 4 --s 2 --L 2 --m $M \
      --k 3 --block_align 2 --P $EPS --rule_seeds 3 --n_gen 4096 $BIG \
      --out "$OUT/E_m$M.json"
done

# --------------------------------------------------------------- F: the depth
# v=2, m=3, k=3 fixed, so f is fixed at 1-(1-1/16)^6 = 0.321 and only U moves:
# L=1 -> U=0 -> ceiling 1 (and k=3=2d-1, pure memorization)
# L=2 -> U=1 -> ceiling 0.321
# L=3 -> U=5 -> ceiling 0.0034
# Three points over three decades; the slope of log G against U is log f.
for LL in 1 2 3; do
  python scripts/rhm_sampling_complexity.py --v 2 --s 2 --L $LL --m 3 \
      --k 3 --block_align 2 --P $EPS --rule_seeds 3 --n_gen 8192 $BIG \
      --out "$OUT/F_L$LL.json"
done

# ------------------------------------------------------- G: P* and the novelty
# tradeoff, at L=2 where both ends are reachable. Fine P ladder: grammaticality
# climbs to its ceiling, then novelty collapses once P approaches the grammar's
# whole support (4*3^5 = 972 grids at m=3, 4*5^5 = 12500 at m=5). The peak of
# success = novel AND grammatical is the sampling complexity hshunt asked for.
GPS="10 20 30 50 70 100 150 200 300 500 700 1000 2000 3000 5000 10000 30000"
python scripts/rhm_sampling_complexity.py --v 4 --s 2 --L 2 --m 3 \
    --k 3 5 7 --block_align 2 --P $GPS --rule_seeds 3 --n_gen 4096 $BIG \
    --out "$OUT/G_L2_pstar.json"
python scripts/rhm_sampling_complexity.py --v 4 --s 2 --L 2 --m 5 \
    --k 3 5 --block_align 2 --P $GPS --rule_seeds 3 --n_gen 4096 $BIG \
    --out "$OUT/G_L2_m5.json"

echo "done $(date)"
