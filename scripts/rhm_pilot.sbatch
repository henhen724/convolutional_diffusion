#!/bin/bash
#SBATCH --job-name=rhm_pilot
#SBATCH --partition=sganguli
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/rhm_pilot_%j.out

# Which LS machines can be grammatical at all? Varies the window k and the phase
# bookkeeping before any scaling sweep: the k=3 pooled machine scores 0 even with
# the grammar's whole support in the training set, so the ceiling has to be
# located first. k = 8 = s**L is the memorization control (expect gram 1, novel 0).

set -euo pipefail
cd "$HOME/convolutional_diffusion"

ml purge
ml devel math python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312 py-numpy/1.26.3_py312
source "$HOME/diffusion_venv_py312/bin/activate"

OUT=results/rhm_sampling/pilot
mkdir -p "$OUT" logs
git rev-parse --short HEAD | tee "$OUT/provenance.txt"

python -m pytest tests/test_rhm_sampling.py -q

python scripts/rhm_sampling_complexity.py --v 4 --s 2 --L 3 --m 3 \
    --k 3 5 7 8 --block_align 1 2 4 8 --P 100 1000 10000 \
    --n_gen 128 --rule_seeds 1 --alpha_smooth 1e-3 --out "$OUT/window_scan.json"

echo "done $(date)"
