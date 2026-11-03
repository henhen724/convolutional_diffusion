#!/bin/bash
#SBATCH --job-name=rhm_adapt
#SBATCH --partition=sganguli
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/rhm_adapt_%j.out

# The window size as a function of time, which every run up to 2026-11-03 got
# wrong by not having one: k was fixed for the whole trajectory and identical at
# every site.
#
# hshunt's criterion is to take the largest length scale whose posterior entropy
# stays above a floor. Along a reverse-erasure trajectory the observed sites
# inside a window only accumulate, so for fixed k the posterior only sharpens; a
# window that is informative while the grid is mostly masked ends up matching a
# single training patch, at which point the machine is copying -- grammatical,
# since training grids are, and never novel. The floor is what keeps it off that
# branch, and because entropy falls with both k and time it makes k *shrink*
# along the trajectory: coarse scales first, fine scales last.
#
# Two readings of "posterior entropy" are measured rather than guessed between.
# h_mode=token is the literal one, the entropy of the predicted-token posterior;
# a 2026-11-03 smoke test shows a 0.2 nat floor on it drops grammaticality from
# 0.71 to 0.03, because it rejects windows that are confident *and right*.
# h_mode=match is the entropy of the posterior over which training patch
# matched, which is zero exactly when one training patch explains the window --
# the memorization signal proper. h_min=0 is the control: always the largest
# window that matched anything.
#
# block_align 1 throughout: the real LS machine is translation invariant.

set -euo pipefail
cd "$HOME/convolutional_diffusion"

ml purge
ml devel math python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312 py-numpy/1.26.3_py312
source "$HOME/diffusion_venv_py312/bin/activate"

OUT=results/rhm_sampling/2026-11-03
mkdir -p "$OUT" logs
git rev-parse --short HEAD > "$OUT/provenance_adapt.txt"

LAD2="10 30 100 300 1000 3000 10000 30000 100000"

# L=2: cheap enough to sweep the floor densely against the fixed-k baselines
# that 50903090 is measuring on the same grammars and the same ladder.
for v in 2 3 4; do
    python scripts/rhm_sampling_complexity.py --v $v --m 3 --L 2 \
        --k_adapt 1 3 5 --h_mode match token --h_min 0 0.1 0.5 1.0 \
        --P $LAD2 --n_gen 4096 --gen_batch 4096 --rule_seeds 3 \
        --block_align 1 --pad_mode outside --out "$OUT/adapt_L2_v${v}_m3.json"
done
for m in 2 5; do
    python scripts/rhm_sampling_complexity.py --v 4 --m $m --L 2 \
        --k_adapt 1 3 5 --h_mode match token --h_min 0 0.1 0.5 1.0 \
        --P $LAD2 --n_gen 4096 --gen_batch 4096 --rule_seeds 3 \
        --block_align 1 --pad_mode outside --out "$OUT/adapt_L2_v4_m${m}.json"
done

# L=3 is the test that matters: fixed k=3 gives grammaticality 0 at every P, so
# any nonzero success here is the adaptive machine doing something a fixed
# window cannot. k=7 is left out -- its table is 1.4e6 distinct windows at
# P=1e5, which the generation conv cannot carry -- so the machine chooses among
# 1, 3, 5 and max_patches skips any cell that still overruns.
for v in 2 3; do
    for m in 2 3; do
        python scripts/rhm_sampling_complexity.py --v $v --m $m --L 3 \
            --k_adapt 1 3 5 --h_mode match --h_min 0 0.1 0.5 1.0 \
            --P 100 300 1000 3000 10000 30000 --n_gen 1024 --gen_batch 1024 \
            --rule_seeds 2 --block_align 1 --pad_mode outside \
            --max_patches 300000 --out "$OUT/adapt_L3_v${v}_m${m}.json"
    done
done

echo "done $(date)"
