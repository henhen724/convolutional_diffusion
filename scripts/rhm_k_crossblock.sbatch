#!/bin/bash
#SBATCH --job-name=rhm_kxblock
#SBATCH --partition=sganguli
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/rhm_kxblock_%j.out
#SBATCH --error=logs/rhm_kxblock_%j.out

# Does consistency BETWEEN blocks inform the per-site window choice?
#
# This is hshunt's straddling question made into a measurement.  n_words
# (rhm_k_content.py, +0.107 median R^2 over nrev across 18 cells) is blind
# outside a site's own block by construction.  n_sym_root is the same block's
# admissible level-1 symbol count AFTER every other block has constrained it
# through the root production -- the part only a straddling window can see.
#
# Registered before this run:
#
#   1. n_sym_root adds >= 0.02 R^2 over the joint (nrev, nw, n_sym_local)
#      baseline, NET of a permutation null (nsr shuffled within step).  r2 is
#      a group-means fit, so any extra column raises it just by splitting the
#      data finer; the null measures that floor.  This also puts a caveat on
#      rhm_k_content.py's +0.107, which had no such null -- only the
#      matched-cardinality nrevb control (+0.006) bounding it.
#   2. Substrate check, reported either way: n_sym_root < n_sym_local at more
#      than 10% of site-observations.  A null in 1 with a null here means
#      "nothing to see"; a null in 1 with a large value here means the machine
#      has the signal available and does not use it.  Different findings.
#   3. Control: n_sym_local on top of (nrev, nw) adds < 0.02, being a
#      coarsening of nw (symbols rather than words).
#
# Second pass (2026-11-05), after the first 18/18 came back: all three held
# (net gain median +0.196, range +0.116..+0.294, null <= 0.001; tighter
# 45.8-70.7%; ctrl <= 0.002).  But frac_dead -- no root production surviving
# every block at once -- runs 19.4% at v2 to 65.9% at v4, so n_sym_root may be
# acting as a binary "already globally doomed" flag rather than carrying graded
# cross-block information.  This pass adds r2_dead (that flag in nsr's place)
# and splits the gain into gain_dead + gain_graded.  No direction registered:
# either split is a real finding, and I want the number before I claim which.
#
# Standing caution from the last two rounds: my stated *mechanisms* have been
# refuted twice (the n_words sign, and "gain grows with v"), while the
# predictive claims held.  So k_by_tighter is printed as a diagnostic, not as
# a prediction -- I am not registering a direction for it.

set -euo pipefail
cd "$HOME/convolutional_diffusion"
mkdir -p logs

ml purge
ml devel math python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312 py-numpy/1.26.3_py312
source "$HOME/diffusion_venv_py312/bin/activate"

for v in 2 3 4; do
  python3 scripts/rhm_k_crossblock.py \
    --v "$v" --m 2 --s 3 --L 2 \
    --rule_seeds 3 --P 300 1000 --k_adapt 5 7 9 --n_gen 512 \
    --out "results/rhm_sampling/2026-11-05/kxblock_v${v}_m2.json"
done

echo "done"
