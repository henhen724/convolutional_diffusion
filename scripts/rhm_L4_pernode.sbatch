#!/bin/bash
#SBATCH --job-name=rhm_L4
#SBATCH --partition=sganguli
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/rhm_L4_%j.out

# A third depth, for the per-node feasibility rather than for grid grammaticality.
#
# The 2026-11-03 sweeps show grammaticality is multiplicative over the parse
# tree: at L=3, k<=5 the level-2 nodes parse at 0.65 and the level-1 nodes at
# 0.165, against 0.65^4 = 0.178 if the four children were independent; at L=2,
# k=3 the level-1 nodes parse at 0.734 against grammaticality 0.222, versus
# 0.734^4 = 0.290. So the object that scales is the per-node feasibility q_l,
# and grid grammaticality is q^M over the M = (n-1)/(s^2-1) internal nodes --
# which is why it collapses with depth whatever the data.
#
# The law needs to know whether q_l(k) depends on the depth of the model or only
# on the window and the node's height above the leaves. L=4 (d=16, n=256) gives
# the third point: if q at one level above the leaves is the same at L=2, L=3
# and L=4 for the same k, then gram = prod_l q_l^(n_l) is a law and not a fit.
#
# k is capped at 5 here deliberately. The U=1 member at L=4 is k=15, whose table
# is out of reach, and grid grammaticality at k<=5 will be 0 at every P -- that
# is expected and is not what this job measures. level_parse_fraction averages
# over 64 level-3 nodes per grid, so n_gen=64 already resolves q_l well even
# though it resolves grammaticality not at all.

set -euo pipefail
cd "$HOME/convolutional_diffusion"

ml purge
ml devel math python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312 py-numpy/1.26.3_py312
source "$HOME/diffusion_venv_py312/bin/activate"

OUT=results/rhm_sampling/2026-11-03
mkdir -p "$OUT" logs
git rev-parse --short HEAD > "$OUT/provenance_L4.txt"

for vm in "2 2" "2 3" "3 2"; do
    set -- $vm
    python scripts/rhm_sampling_complexity.py --v $1 --m $2 --L 4 \
        --k 3 5 --P 100 300 1000 3000 10000 --n_gen 64 --gen_batch 64 \
        --rule_seeds 2 --block_align 1 --pad_mode outside \
        --max_patches 1500000 --out "$OUT/L4_v$1_m$2.json"
done

echo "done $(date)"
