#!/bin/bash
#SBATCH --job-name=rhm_depth2d
#SBATCH --partition=sganguli
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/rhm_depth2d_%j.out

# Is x* = P*/|R_L| depth-stable in 2D *at fixed grammar shape*?
#
# This is the load-bearing assumption behind the headline P* = 0.21 |R_L|, and
# right now it rests on pooling L=2 (0.1925 over nine (v,m) combos) against L=3
# (0.2075 over two).  Those two sets of combos are DISJOINT, and x* varies 7.6x
# across (v,m) medians -- so that comparison cannot separate depth from shape.
# The same disjoint-pooling reasoning is what hid a real depth effect on the 1D
# side, where pairing within (v,m) shows x* falling with depth in 15/15 combos
# (median ~2.8x).  If 2D does the same, 0.21 is not a constant and the headline
# needs a depth-dependent statement instead.
#
# 2D already has L=3 m=2 peaks at v=2 and v=3 (L3peak_v*_m2_k7.json).  What is
# missing is their L=2 partners: m=2 was only ever run at L=2 for v=4.  These
# are cheap -- |R_L| is 64 / 96 / 128, so the whole peak lives below P=1000 --
# which is why eight rule seeds and n_gen=16384 are affordable here.  v=4 is
# included as a control: it is the one m=2 cell that already exists at L=2
# (n_gen=8192, two seeds), so agreement there checks that nothing about the
# denser ladder or the extra seeds moves the answer on its own.
#
# n_gen=16384 is also the "raise n_gen first" step STATUS asks for: the peak
# estimator's noise is binomial in n_gen, and the bootstrap in
# scripts/rhm_peak_noise.py shows that noise, not the grammar, is the entire
# source of the 1.5x across-seed scatter at n_gen=4096.
#
# k = U1 = 2s^(L-1) - 1 = 3 at L=2, matching the U=1 window the published peaks
# use.  block_align 1 / pad_mode outside throughout: the real LS machine is
# translation invariant, and every published peak on both sides uses these.

set -euo pipefail
cd "$HOME/convolutional_diffusion"

ml purge
ml devel math python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312 py-numpy/1.26.3_py312
source "$HOME/diffusion_venv_py312/bin/activate"

OUT=results/rhm_sampling/2026-11-05
mkdir -p "$OUT" logs
git rev-parse --short HEAD > "$OUT/provenance_depth2d.txt"

# |R_L| for the five extra rule seeds.  It is 64/96/128 for seeds 0-2 at every
# v here, so this is expected to be flat, but x* is a ratio to |R_L| and the
# ratio should never be taken on an assumed denominator.
python scripts/rhm_window_count_exact.py --v 2 3 4 --m 2 --L 2 --k 3 \
    --rule_seeds 8 --out "$OUT/nk_exact_L2_rs8.json"

# ~1.26x spacing through the region where the peak has to be: x* spans roughly
# 0.05-0.6 across known grammars, so P* is somewhere in 3-77 for these |R_L|.
LAD="2 3 4 5 6 8 10 13 16 20 25 32 40 50 64 80 100 128 160 200 265 350 500 700 1000"

for v in 2 3 4; do
    python scripts/rhm_sampling_complexity.py --v $v --m 2 --L 2 --k 3 \
        --P $LAD --n_gen 16384 --gen_batch 4096 --rule_seeds 8 \
        --block_align 1 --pad_mode outside \
        --out "$OUT/depth2d_L2_v${v}_m2.json"
done

echo "done $(date)"
