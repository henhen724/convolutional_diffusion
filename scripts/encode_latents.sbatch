#!/bin/bash
#SBATCH --job-name=encode_vae
#SBATCH --partition=sganguli
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=results/logs/encode_vae_%j.out
#SBATCH --error=results/logs/encode_vae_%j.err

set -e  # Exit on error so SLURM captures the real exit code

# Activate environment
source ~/activate_diffusion.sh

# Create log directory
mkdir -p results/logs

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

cd /home/users/hshunt/convolutional_diffusion

python scripts/encode_celebahq_latents.py \
    --outdir /scratch/users/hshunt/celeba_hq_latents \
    --data_dir /scratch/users/hshunt/celeba_hq_latents/raw \
    --batch_size 64 \
    --image_size 256 \
    --num_workers 4

echo "Job finished at $(date)"
