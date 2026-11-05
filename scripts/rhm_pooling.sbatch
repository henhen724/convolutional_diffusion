#!/bin/bash
#SBATCH --job-name=rhm_pooling
#SBATCH --partition=sganguli
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/rhm_pooling_%j.out

# Pooling costs the sampler, not the support -- widening the measurement.
#
# scripts/rhm_pooling_cost.py decomposes the per-phase-vs-pooled grammaticality
# gap as  gram_2/gram_1 = pool_cost x sampler, where pool_cost = lv_pooled /
# lv_aligned is the exact factor by which pooling inflates the locally-valid set
# (rhm_straddle.py:165, from enumerating all v^(d*d) grids). On the data in hand
# it gives 3.46x = 1.05x x 3.18x: pooling barely weakens the *constraint*, it
# wrecks the *weights*. The pooled sampler realises 0.22 of its own support
# ceiling, the per-phase one 0.72.
#
# That rests on 10 cells, all v in {2,3}, m = 3, k = 3. The exact support exists
# for 50 such cells (straddle_v2.json, straddle_v3.json: v in {2,3},
# m in {2,3,4,5,6}, 5 seeds, k = 3) and the other 40 have no run at all at
# s=2, L=2, k=3, pad_mode=outside. Arm A fills them in, both alignments in one
# invocation so the pair is exactly matched, and lets the sampler term be read
# against m as well as against v.
#
# k = 3 is not a restriction here, it is the only interior window: L=2, s=2 is a
# 4x4 grid and U1 = 3, so k = 5 is already mostly padding.
#
# Arm B tests what the sampler term scales with. Pooling mixes s^2 phases into
# one table -- 4 at s=2, 9 at s=3. Prediction, registered before the run: the
# per-phase/pooled grammaticality ratio at s=3, L=2 exceeds the 3.46x measured
# at s=2, because there is more to mix. The null is that it is unchanged, which
# would mean the cost is per-site weight error and not phase count. No exact
# support at s=3 (v^81 grids), so this arm reads the measured ratio only.
# Phase-aware means block_align = s (rhm_sampling.py:54), hence 3 here, and the
# pooled half already exists from 51106614 at identical settings, so only the
# per-phase half is run and it pairs row for row.

set -euo pipefail
cd "$HOME/convolutional_diffusion"

ml purge
ml devel math python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312 py-numpy/1.26.3_py312
source "$HOME/diffusion_venv_py312/bin/activate"

OUT=results/rhm_sampling/2026-11-04
mkdir -p "$OUT" logs
git rev-parse --short HEAD > "$OUT/provenance_pooling.txt"

# |R_L| = v * m^5 here, 64 (v2 m2) to 23328 (v3 m6), so P* = 0.21 |R_L| runs
# from ~13 to ~4900; this ladder brackets every one of them.
LAD="3 10 30 100 300 1000 3000 10000 30000"

for v in 2 3; do
    for m in 2 4 5 6; do
        python scripts/rhm_sampling_complexity.py --v "$v" --m "$m" --s 2 --L 2 \
            --k 3 --P $LAD --n_gen 4096 --gen_batch 4096 --rule_seeds 5 \
            --block_align 1 2 --pad_mode outside \
            --out "$OUT/pool_L2_v${v}_m${m}.json"
    done
done

LAD_S3="10 30 100 300 1000 3000 10000"
for v in 2 3; do
    python scripts/rhm_sampling_complexity.py --v "$v" --m 2 --s 3 --L 2 \
        --k 5 7 9 --P $LAD_S3 --n_gen 4096 --gen_batch 4096 --rule_seeds 3 \
        --block_align 3 --pad_mode outside \
        --out "$OUT/s3_phase_L2_v${v}_m2.json"
done

echo "done $(date)"
