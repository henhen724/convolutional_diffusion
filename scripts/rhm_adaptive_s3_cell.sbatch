#!/bin/bash
#SBATCH --job-name=rhm_s3_cell
#SBATCH --partition=sganguli
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/rhm_s3_cell_%j.out

# One cell of the s=3 fair test, so a slow cell cannot eat the others.
#
# 51106614 ran all four cells in one 3 h job and timed out inside cell 3:
# v2m2 and v3m2 landed whole, v4m2 got 47 of 84 adaptive points, v2m3 never
# started. Measured cost per adaptive arm there was 2876 s (v2m2), 4196 s
# (v3m2) and ~5480 s extrapolated (v4m2) -- per-point time saturates in P
# rather than growing linearly (v2m2: 9.7 s at P=10 to 69.8 s at P=10000),
# because the distinct-patch table saturates. MaxRSS was 1.6 GB, so 32G was
# over-provisioned by 20x; 8G here.
#
# Usage:  sbatch --time=HH:MM:SS scripts/rhm_adaptive_s3_cell.sbatch <v> <m> <small|big>
#
# Same grammars, ladder, seeds, n_gen and alignment as 51106614, so the cells
# pair exactly with the eight already scored. Overwrites that cell's two JSON
# files; the partial v4m2 adaptive file is reproduced from scratch.

set -euo pipefail
cd "$HOME/convolutional_diffusion"

V=$1; M=$2; LAD_NAME=$3
case "$LAD_NAME" in
    small) LAD="10 30 100 300 1000 3000 10000" ;;
    big)   LAD="100 300 1000 3000 10000 30000 100000" ;;
    *)     echo "ladder must be small or big" >&2; exit 2 ;;
esac

ml purge
ml devel math python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312 py-numpy/1.26.3_py312
source "$HOME/diffusion_venv_py312/bin/activate"

OUT=results/rhm_sampling/2026-11-04
mkdir -p "$OUT" logs
git rev-parse --short HEAD > "$OUT/provenance_adapt_s3.txt"

python scripts/rhm_sampling_complexity.py --v "$V" --m "$M" --s 3 --L 2 \
    --k 5 7 9 --P $LAD --n_gen 4096 --gen_batch 4096 --rule_seeds 3 \
    --block_align 1 --pad_mode outside \
    --out "$OUT/s3_fixed_L2_v${V}_m${M}.json"

python scripts/rhm_sampling_complexity.py --v "$V" --m "$M" --s 3 --L 2 \
    --k_adapt 5 7 9 --h_mode match --h_min 0 0.1 0.5 1.0 \
    --P $LAD --n_gen 4096 --gen_batch 4096 --rule_seeds 3 \
    --block_align 1 --pad_mode outside \
    --out "$OUT/s3_adapt_L2_v${V}_m${M}.json"

echo "done v${V} m${M} $(date)"
