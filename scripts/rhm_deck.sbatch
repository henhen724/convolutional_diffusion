#!/bin/bash
#SBATCH --job-name=rhm_deck
#SBATCH --partition=sganguli
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=logs/rhm_deck_%j.out

# Builds the corrected slide PDF once 50935971 has written its last cell.
#
# This needs a GPU only for the trace page: it rebuilds three LS machines
# (k=3,5,7) on P=20000 L=3 grids and draws 64 samples from each, so the deck
# shows what the machine actually emits rather than asserting it. Everything
# else on the deck is replotted from the result JSONs. ~1.3e6 windows in the
# k=7 table, so 64G and a few minutes.

set -euo pipefail
cd "$HOME/convolutional_diffusion"

ml purge
ml devel math python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312 py-numpy/1.26.3_py312
source "$HOME/diffusion_venv_py312/bin/activate"

OUT=results/rhm_sampling/2026-11-03
mkdir -p "$OUT" logs

# refresh the peak table from whatever the sweep finished with
python scripts/rhm_peak_analysis.py --out "$OUT" --png "$OUT/collapse.png" \
    | tee "$OUT/peak_analysis.txt"

python scripts/build_rhm_peak_slides.py --results "$OUT" \
    --out "$OUT/rhm_sampling_complexity_slides.pdf" \
    --png_dir "$OUT/slide_pages" --trace_P 20000 --trace_n 64

echo "done $(date)"
