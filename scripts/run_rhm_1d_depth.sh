#!/bin/bash
#SBATCH --job-name=rhm1d_depth
#SBATCH --partition=sganguli
#SBATCH --time=06:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/scratch/users/hshunt/convolutional_diffusion/logs/rhm1d_depth_%j.out

# Third depth for the exp(Theta(n)) law.  In 2D the root-reaching window at L=4
# is a 15x15 patch that no table fills; in 1D it is 15 sites, so n = 4, 8, 16
# are all reachable and P*/|R_L| gets a real lever arm.  m >= 3 at L=4 and
# m >= 5 at L=3 are skipped automatically by --R_max: their grammars are larger
# than the P ladder can bracket, and an unbracketed peak is not a measurement.

set -euo pipefail
cd /home/users/hshunt/convolutional_diffusion

ml purge
ml devel math python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312 py-numpy/1.26.3_py312
source "$HOME/diffusion_venv_py312/bin/activate"

OUT=results/rhm_1d/2026-11-03
mkdir -p "$OUT" logs

python scripts/rhm_1d_depth.py \
  --L 2 3 4 \
  --v 2 3 4 \
  --m 2 3 4 5 6 \
  --s 2 \
  --rule_seeds 3 \
  --n_gen 2048 \
  --gen_batch 1024 \
  --P_min 1 --P_max 262144 \
  --R_max 1.5e5 \
  --out "$OUT/sweep.json" \
  --rl_out "$OUT/RL_exact_1d.json"
