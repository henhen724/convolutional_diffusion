#!/bin/bash
#SBATCH --job-name=rhm_kcontent
#SBATCH --partition=sganguli
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/rhm_kcontent_%j.out
#SBATCH --error=logs/rhm_kcontent_%j.out

# Does a grammar-side feature name the >=53% that the reveal count leaves?
#
# Registered before this run (the smoke cell v2 m2 rs0 P=300 n_gen=64 is the
# only evidence so far, and one cell is not a result):
#
#   1. n_words adds >= 0.05 R^2 over the K x K reveal count alone.  Smoke
#      cell gave +0.109, so this is expected to hold.
#   2. The gain GROWS WITH v, because v*m words means more to tell apart and
#      so more resolution: 4 words at v2, 6 at v3, 8 at v4.
#   3. The nrev_block control stays small (<= ~0.03).  If it does not, the
#      gain is the block's own fill level under a grammatical name rather
#      than grammatical ambiguity, and claim 1 means nothing.
#
# Already refuted by the smoke cell and recorded as such: the mechanism I
# stated with the prediction -- "a block pinned to one word supports a larger
# window" -- has the sign backwards.  Mean k RISES with n_words (8.50 at
# nw=1 to 8.98 at nw=4).  That raw marginal is confounded with fill level
# (nw = v*m exactly when the block is untouched, which is early), which is
# why the gain is measured over nrev rather than on its own.

set -euo pipefail
cd "$HOME/convolutional_diffusion"
mkdir -p logs

ml purge
ml devel math python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312 py-numpy/1.26.3_py312
source "$HOME/diffusion_venv_py312/bin/activate"

for v in 2 3 4; do
  python3 scripts/rhm_k_content.py \
    --v "$v" --m 2 --s 3 --L 2 \
    --rule_seeds 3 --P 300 1000 --k_adapt 5 7 9 --n_gen 512 \
    --out "results/rhm_sampling/2026-11-04/kcontent_v${v}_m2.json"
done

echo "done"
