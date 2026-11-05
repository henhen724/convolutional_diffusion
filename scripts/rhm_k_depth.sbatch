#!/bin/bash
#SBATCH --job-name=rhm_kdepth
#SBATCH --partition=sganguli
#SBATCH --array=0-2
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/rhm_kdepth_%A_%a.out
#SBATCH --error=logs/rhm_kdepth_%A_%a.out

# Near coupling, far coupling, or one bit?  STATUS next step 5.
#
# Every cross-block result so far (rhm_k_crossblock.py, rhm_k_bp.py, 18/18
# each) was measured at s=3 L=2, where a site's level-1 block has exactly one
# thing above it: the root.  There "consistent with the other blocks" and
# "consistent through the root" are the same sentence, so the one-bit finding
# cannot tell a local consistency check from a global one.  L=3 splits them --
# the parent is a 9x9 region, the root couples nine of those.
#
# The registered predictions are in the script docstring (items 1-6) and are
# not restated here; what IS recorded here is what the sizing probe found
# first, so the operating point below is auditable:
#
#   * P does not move the far bit's substrate.  dead_root ran 98 / 96 / 96% at
#     P = 50 / 300 / 1000 (v2 m2, n_gen 32).  Deadness at L=3 is about the
#     sampler almost never emitting a globally parseable 27x27 grid, not about
#     how much training data it saw, so there is no P that buys range.  The
#     ladder stays at the BP run's 300 / 1000 for comparability instead of
#     chasing one.
#   * That is why item 6 exists.  Deadness is monotone up the tree and came out
#     30 / 78 / 96% local / parent / root, so the far bit is nearly constant
#     marginally -- check 5 passes (18% disagreement) but a marginal null would
#     still be half unreadable.  Among blocks their parent has not killed it
#     splits ~82/18, and that is where item 6 scores it.
#   * Cost and footprint, measured, per hshunt's standing "measure the
#     footprint before you just raise --mem":  at n_gen 32, P=300 took 5 min
#     and P=1000 17 min; peak RSS hit 28.2 GB at P=3000 against a 32 G cap, and
#     RSS tracks patch count linearly, so P=1000 is ~9.4 GB and P=300 ~3 GB.
#     24 G is ~2.5x the largest cell here.  P=3000 is deliberately NOT run: it
#     would have OOMed and buys nothing (first bullet).
#   * n_gen 64, not the L=2 run's 512.  The L=3 grid is 27x27, nine times the
#     sites, so 64 x 729 = 46656 site-draws per step matches L=2's 512 x 81 =
#     41472.  Matching n_obs is the point; matching n_gen would have been a 9x
#     overspend for a tighter error bar on one depth than the other.
#   * Budget: 3 seeds x (5 + 17) min x 2 for n_gen 64 = ~2.2 h per v, inside
#     the 4 h wall with room for v=3/v=4 being slower than the v=2 probe.
#
# Split by v as an array rather than looped in one job: the three v are
# independent, 3 x 2.2 h does not fit one 4 h wall, and the probe only measured
# v=2 -- if a higher v is slower than expected it costs one array task, not the
# whole run.  Nothing here excludes nodes (unsupported on Sherlock).
#
# Standing caution, unchanged: my stated MECHANISMS have been refuted three
# times (the n_words sign, "gain grows with v", the C(k) = f0^U ceiling) while
# the predictive claims held.  Item 1 is the only one I am confident in, and it
# is confident only because it is the L=2 effect re-read at depth.

set -euo pipefail
cd "$HOME/convolutional_diffusion"
mkdir -p logs results/rhm_sampling/2026-11-05

ml purge
ml devel math python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312 py-numpy/1.26.3_py312
source "$HOME/diffusion_venv_py312/bin/activate"

V=$(( SLURM_ARRAY_TASK_ID + 2 ))          # array 0,1,2 -> v = 2,3,4

python3 scripts/rhm_k_depth.py \
  --v "$V" --m 2 --s 3 --L 3 \
  --rule_seeds 3 --P 300 1000 --k_adapt 5 11 17 --n_gen 64 --nbins 8 \
  --out "results/rhm_sampling/2026-11-05/kdepth_v${V}_m2_L3.json"

echo "done v=$V"
