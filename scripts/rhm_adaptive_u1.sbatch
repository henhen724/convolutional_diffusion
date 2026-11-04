#!/bin/bash
#SBATCH --job-name=rhm_adapt_u1
#SBATCH --partition=sganguli
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=logs/rhm_adapt_u1_%j.out

# One variable against the 2026-11-03 adaptive sweep: the window set the floor
# is allowed to choose from.
#
# That sweep gave the machine k in {1, 3, 5} and swept the floor. Re-read on
# novel-and-grammatical (scripts/rhm_adaptive_score.py, 2026-11-04) it loses to
# the best fixed k in 20 of 21 cells, but *beats* the fixed k=5 it imitates in
# 11 of 12 -- so the adaptivity is not the problem, the choice set is. The
# induced k(t) says why: raising h_min from 0 to 1.0 drives the mean k at the
# last reveal from 4.13 down to 1.10, which lifts novelty 0.487 -> 0.975 exactly
# as the criterion intends, while grammaticality falls 0.679 -> 0.065 and
# success with it, 0.166 -> 0.040.
#
# k = 1 and 2 are below U1 = 2 s^(L-1) - 1 = 3, the smallest window that can
# contain a complete top-level child block. A window under U1 cannot see the
# root, so it cannot be grammatical at any P, and letting the entropy floor
# select one spends the whole grammaticality budget to buy novelty that is then
# worthless. Removing k=1 leaves the floor free to trade 5 against 3 and no
# lower.
#
# Prediction: with k_set = {3, 5} a binding floor beats both the h_min=0 control
# and fixed k=3, because the novelty gain survives and the collapse does not
# happen. If it instead just reproduces fixed k=3, the criterion adds nothing at
# L=2 and the honest answer is that U1 already is the schedule.
#
# Same grammars, same ladder, same seeds, same n_gen as the 2026-11-03 run, so
# every cell pairs exactly. h_mode=match only: token was shown on 2026-11-03 to
# reject windows that are confident and right.

set -euo pipefail
cd "$HOME/convolutional_diffusion"

ml purge
ml devel math python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312 py-numpy/1.26.3_py312
source "$HOME/diffusion_venv_py312/bin/activate"

OUT=results/rhm_sampling/2026-11-04
mkdir -p "$OUT" logs
git rev-parse --short HEAD > "$OUT/provenance_adapt_u1.txt"

LAD2="10 30 100 300 1000 3000 10000 30000 100000"

for spec in "2 3" "3 3" "4 2" "4 3" "4 5"; do
    set -- $spec
    v=$1; m=$2
    python scripts/rhm_sampling_complexity.py --v $v --m $m --L 2 \
        --k_adapt 3 5 --h_mode match --h_min 0 0.1 0.5 1.0 \
        --P $LAD2 --n_gen 4096 --gen_batch 4096 --rule_seeds 3 \
        --block_align 1 --pad_mode outside \
        --out "$OUT/adaptu1_L2_v${v}_m${m}.json"
done

echo "done $(date)"
