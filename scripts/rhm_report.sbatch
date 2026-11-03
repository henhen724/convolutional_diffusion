#!/bin/bash
#SBATCH --job-name=rhm_report
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/rhm_report_%j.out

# Reporting pass for the RHM sampling-complexity result: rebuild the slide deck
# and execute the results notebook in place, from the same JSONs and the same
# page functions, so deck and notebook cannot drift apart. Both re-run the live
# example traces (a k=15 patch-count machine on 600 grids), which is a few
# minutes of CPU -- hence a job rather than a login node.

set -euo pipefail
cd "$HOME/convolutional_diffusion"

ml purge
ml devel math python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312 py-numpy/1.26.3_py312
source "$HOME/diffusion_venv_py312/bin/activate"

OUT=results/rhm_sampling/2026-11-02
git rev-parse --short HEAD > "$OUT/provenance_report.txt"

python scripts/build_rhm_slides.py --results "$OUT" \
    --out "$OUT/rhm_sampling_complexity_slides.pdf"

jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=1800 \
    notebooks/rhm_sampling_complexity_results.ipynb

ls -la "$OUT/rhm_sampling_complexity_slides.pdf"
echo "done $(date)"
