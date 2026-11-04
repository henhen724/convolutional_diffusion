#!/bin/bash
#SBATCH --job-name=rhm_adapt_s3
#SBATCH --partition=sganguli
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/rhm_adapt_s3_%j.out

# The first fair test of hshunt's patch-size criterion: a branching factor where
# a schedule has somewhere to go.
#
# Everything measured so far used s = 2, and there U1 = 2 s^(L-1) - 1 = 2^L - 1
# is exactly (grid side - 1). The only odd window in [U1, side] is U1 itself, at
# every depth: 4x4 -> {3}, 8x8 -> {7}, 16x16 -> {15}. Every larger window is
# bigger than the grid and is mostly padding. So "largest k above an entropy
# floor" was being scored on a choice set of one real rung plus the whole grid,
# which is why 51098252 could only conclude that U1 already is the schedule.
# (The deck built on 2026-11-04 blamed the int64 overflow in
# scripts/rhm_straddle.py:window_codes for this. That is wrong -- that overflow
# limits the a_j *support* instrument, not the sampler, which already ran k=7 at
# L=3 on 2026-11-03. The real limit is the arithmetic above.)
#
# s = 3, L = 2 is a 9x9 grid with U1 = 5 and three genuine rungs, 5 / 7 / 9. A
# smoke test (job 51106057, v2 m2, n_gen 256) shows they are not
# interchangeable: at P = 1000, grammaticality is 0.000 at k=5, 0.086 at k=7 and
# 0.312 at k=9. Unlike s = 2, the best fixed window is not U1.
#
# Prediction, registered before the run: with three rungs the adaptive machine
# beats the best fixed k in a majority of the 12 cells, because the trade the
# floor is built to make -- start wide while the grid is empty and the match is
# loose, fall back as it fills and the posterior sharpens -- is finally
# available to it. The null: the best fixed k still wins, in which case the
# criterion loses on a fair test too, and the answer to hshunt's question is
# that the window should simply be fixed at the best k rather than scheduled.
#
# Secondary reading: whether the induced k(t) now spans more than two values,
# and whether the fixed optimum sits at U1, above it, or at the whole grid.
#
# |R_L| = v * m^(1 + s^2) = v * m^10 here, so the m=2 cells peak near a few
# hundred and v2 m3 near 25k; the ladders below bracket both.

set -euo pipefail
cd "$HOME/convolutional_diffusion"

ml purge
ml devel math python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312 py-numpy/1.26.3_py312
source "$HOME/diffusion_venv_py312/bin/activate"

OUT=results/rhm_sampling/2026-11-04
mkdir -p "$OUT" logs
git rev-parse --short HEAD > "$OUT/provenance_adapt_s3.txt"

LAD_SMALL="10 30 100 300 1000 3000 10000"
LAD_BIG="100 300 1000 3000 10000 30000 100000"

run_cell () {          # v m ladder
    local v=$1 m=$2 lad=$3
    python scripts/rhm_sampling_complexity.py --v "$v" --m "$m" --s 3 --L 2 \
        --k 5 7 9 --P $lad --n_gen 4096 --gen_batch 4096 --rule_seeds 3 \
        --block_align 1 --pad_mode outside \
        --out "$OUT/s3_fixed_L2_v${v}_m${m}.json"
    python scripts/rhm_sampling_complexity.py --v "$v" --m "$m" --s 3 --L 2 \
        --k_adapt 5 7 9 --h_mode match --h_min 0 0.1 0.5 1.0 \
        --P $lad --n_gen 4096 --gen_batch 4096 --rule_seeds 3 \
        --block_align 1 --pad_mode outside \
        --out "$OUT/s3_adapt_L2_v${v}_m${m}.json"
}

run_cell 2 2 "$LAD_SMALL"
run_cell 3 2 "$LAD_SMALL"
run_cell 4 2 "$LAD_SMALL"
run_cell 2 3 "$LAD_BIG"

echo "done $(date)"
