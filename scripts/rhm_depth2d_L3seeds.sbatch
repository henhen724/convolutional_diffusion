#!/bin/bash
#SBATCH --job-name=rhm_L3seed
#SBATCH --partition=sganguli
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=10:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/rhm_L3seed_%A_%a.out

# Widens the L=3 side of the 2D depth test, which is what limits it.
#
# 51244089 gives eight rule seeds for 2D L=2 m=2 at v=2,3.  Against them the
# published L=3 m=2 peaks are three seeds each, and the first paired estimate
# came out x*(L=2)/x*(L=3) = 1.07x with a 95% interval of [0.52, 2.34] -- which
# covers both 1 and the ~2.8x depth effect that 1D shows, so it decides nothing.
# The per-cell bootstrap spread at L=3 is only 1.11-1.17x on the well-measured
# cells, so n_gen is not the binding constraint; the seed count is.  Going from
# 3 to 10 seeds at v=2 should shrink the interval by about sqrt(10/3) = 1.8x,
# enough to separate "no effect" from the 1D-sized one.
#
# Settings are the published ones exactly (n_gen 1024, nsteps 64, k=7=U1,
# block_align 1, pad_mode outside, max_patches 25e6) so the new seeds pool with
# L3peak_v*_m2_k7.json rather than forming a separate population -- the mistake
# this whole line of work exists to undo.  --rule_seed_start is new: the
# grammar seed is 1000+rs and the sampler seed 7e6+1000*rs+P, both functions of
# rs alone, so seeds 3+ are genuinely new grammars and seeds 0-2 are untouched.
#
# Ladder changes, both deliberate:
#   * 600000 and 2000000 added at v=2.  The published 3.16x spacing puts only
#     one point inside the peak, and a 3-point parabola on a 10x-wide bracket is
#     most of why the L=3 estimates are soft.
#   * v=3 now runs to 6000000 instead of stopping at 3000000.  |R_L| is 6.29e6
#     for two of the three published v=3 seeds, so the old ladder reached only
#     P/|R_L| = 0.48 and left rs0 unbracketed -- it never got past the peak.
#     The table size is capped by N_7 = 1.8e7 distinct windows, not by P, so
#     P=6e6 costs no more memory than the 3e6 cell already did.
#
# |R_L| is recomputed per seed rather than carried over: at L=3 it already
# ranges 20x across the published seeds (v3m2: 6291456 / 301440 / 6291456)
# because rules are drawn with replacement and collide, and x* is a ratio to it.

set -euo pipefail
cd "$HOME/convolutional_diffusion"

ml purge
ml devel math python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312 py-numpy/1.26.3_py312
source "$HOME/diffusion_venv_py312/bin/activate"

OUT=results/rhm_sampling/2026-11-05
mkdir -p "$OUT" logs
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

COMMON="--L 3 --s 2 --m 2 --k 7 --n_gen 1024 --gen_batch 512 --nsteps 64 \
        --block_align 1 --pad_mode outside --max_patches 25000000 \
        --rule_seed_start 3"

case "$SLURM_ARRAY_TASK_ID" in
  0)
    V=2; NSEED=7
    LAD="10000 30000 100000 300000 600000 1000000 2000000 4000000"
    ;;
  1)
    V=3; NSEED=5
    LAD="10000 30000 100000 300000 600000 1000000 2000000 3000000 6000000"
    ;;
  *) echo "bad task id"; exit 1 ;;
esac

git rev-parse --short HEAD > "$OUT/provenance_L3seed_v${V}.txt"

python scripts/rhm_window_count_exact.py --v $V --m 2 --L 3 --k 7 \
    --rule_seed_start 3 --rule_seeds $NSEED \
    --out "$OUT/nk_exact_L3_ext_v${V}.json"

python scripts/rhm_sampling_complexity.py --v $V --rule_seeds $NSEED $COMMON \
    --P $LAD --out "$OUT/L3seed_v${V}_m2_k7.json"

echo "done $(date)"
