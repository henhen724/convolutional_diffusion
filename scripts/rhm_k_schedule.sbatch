#!/bin/bash
#SBATCH --job-name=rhm_ksched
#SBATCH --partition=sganguli
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=05:00:00
#SBATCH --output=logs/rhm_ksched_%j.out

# Is the adaptive window's 50x win a k(t) schedule, or per-site adaptivity?
#
# The s=3 fair test left the criterion "largest k with any support" unbeaten and
# every floor on top of it refuted -- entropy (h_min best at 0 in 8/8 cells) and
# now count (n_min monotone down: succ .414 -> .148 as n_min 1 -> 10 at v2m2
# rs0 P=300, buying novelty +0.14 against grammaticality -0.40). So the whole
# contribution is the choice of k, and this job splits that choice in two.
#
# scripts/rhm_k_schedule.py runs four matched arms and derives the control
# schedule from the adaptive arm's own recorded trace, so the comparison cannot
# be rigged by picking a bad schedule by hand.
#
# Single-cell result at v2 m2 rs0 P=300, n_gen=4096, which this job tests for
# generality:
#     adaptive  succ 0.4363  gram 0.610  mean k 8.297
#     cap       succ 0.4412  gram 0.608  mean k 7.867
#     sched     succ 0.0925  gram 0.259  mean k 8.309
#     fixed k=7 succ 0.0444  gram 0.071  mean k 7.000
# cap ~ adaptive says the criterion never wanted a window above its own
# schedule. sched at the *same mean k* losing 4.7x says the size that matters is
# the one at each site, not the one on average at that time. Prediction for the
# other cells, registered before the run: the sched/adaptive ratio stays well
# below 1 and falls as v grows, because a larger vocabulary makes support
# sparser and more unevenly spread across sites. The null is sched ~ adaptive at
# some cell, which would make the answer to "how do you set the patch size over
# time" a curve after all.
#
# |R_L| = v * m^10 = 2048 / 3072 / 4096 at m=2, v=2/3/4, so P* ~ 0.21 |R_L| is
# ~430 / ~645 / ~860 and this ladder brackets all three.

set -euo pipefail
cd "$HOME/convolutional_diffusion"

ml purge
ml devel math python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312 py-numpy/1.26.3_py312
source "$HOME/diffusion_venv_py312/bin/activate"

OUT=results/rhm_sampling/2026-11-04
mkdir -p "$OUT" logs
git rev-parse --short HEAD > "$OUT/provenance_ksched.txt"

for v in 2 3 4; do
    python scripts/rhm_k_schedule.py --v "$v" --m 2 --s 3 --L 2 \
        --rule_seeds 3 --P 100 300 1000 3000 \
        --k_adapt 5 7 9 --k_fixed 7 --n_gen 4096 \
        --out "$OUT/ksched_L2_v${v}_m2.json"
done

echo "done $(date)"
