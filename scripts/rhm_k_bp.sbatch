#!/bin/bash
#SBATCH --job-name=rhm_kbp
#SBATCH --partition=sganguli
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/rhm_kbp_%j.out
#SBATCH --error=logs/rhm_kbp_%j.out

# Is the criterion's cross-block dependence really only one bit?
#
# rhm_k_crossblock.py (18/18) found the window depends on whether a site's
# level-1 block is still consistent with the OTHER blocks through the root:
# median +0.196 R^2, null +0.000.  But a bare binary "no root production
# survives every block at once" flag reproduced that gain to +0.000 in all 18,
# even though the survivor COUNT was >= 2 at 10.6-27.0% of observations.  A
# hard count is a coarse probe -- it asks which symbols are possible, never how
# probable.  BP is exact on this tree, so the graded version is in closed form.
#
# Registered before this run:
#
#   1. h_root (exact H(block symbol | ALL observed leaves)) adds >= 0.02 R^2
#      over (nrev, n_words, h_local, dead), NET of a null that shuffles it
#      within each (step, dead) stratum -- preserving the time coupling and the
#      dead/alive split, destroying only the graded content.  No confident
#      direction: the count said zero, and a NULL is the stronger finding here,
#      because "the Bayes-optimal posterior adds nothing over one bit" is a far
#      harder statement than "a coarse count adds nothing".
#   2. Correctness cross-check, not a prediction: BP deadness (zero posterior
#      mass at level 1) agrees with the enumeration's n_sym_root == 0 at > 99%.
#      Independent implementations of one predicate -- disagreement means one is
#      wrong and the run is void.
#   3. Substrate, reported either way: among ALIVE observations h_root resolves
#      into more than one bin.  Without this, 1 collapses to "h_root is
#      constant" and says nothing about grading.
#
# Found while smoke-testing, before any full run: check 3 is sensitive to the
# binning, and plain quantile bins were failing it for the wrong reason.
# h_root is exactly 0 on 71-80% of ALIVE observations (the root pins the
# block's symbol), with the graded content in a thin positive tail -- 33 values
# at v2, 66 at v4.  Plain quantile cuts all landed on the zero atom, np.unique
# collapsed them, and the tail got 2 bins.  qbin now gives the atom its own bin
# and quantiles the positive part, which is 7-8 bins on the same cells.  The
# measured gain was +0.000 both before and after, so this did not move the
# answer; it removes the objection that a null was my own discretization.
#
# Carried UNREGISTERED and labelled so: h_leaf, the exact posterior entropy at
# the site's own token.  That is the quantity hshunt named for this criterion
# ("the maximum length scale which stays above the minimum posterior entropy"),
# where h_root is about the block's symbol.  Free from the same BP pass, and
# the natural place to look if h_root comes back null -- that would mean the
# right idea measured at the wrong level of the tree.  Smoke: it explains
# 0.34-0.47 of k on its own and adds +0.000 over the baseline.
#
# Standing caution: my stated MECHANISMS have now been refuted three times (the
# n_words sign, "gain grows with v", and the C(k) = f0^U ceiling) while the
# predictive claims held.  Nothing here registers a mechanism.

set -euo pipefail
cd "$HOME/convolutional_diffusion"
mkdir -p logs

ml purge
ml devel math python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312 py-numpy/1.26.3_py312
source "$HOME/diffusion_venv_py312/bin/activate"

for v in 2 3 4; do
  python3 scripts/rhm_k_bp.py \
    --v "$v" --m 2 --s 3 --L 2 \
    --rule_seeds 3 --P 300 1000 --k_adapt 5 7 9 --n_gen 512 \
    --out "results/rhm_sampling/2026-11-05/kbp_v${v}_m2.json"
done

echo "done"
