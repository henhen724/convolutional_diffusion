#!/bin/bash
#SBATCH --job-name=lat_resnet
#SBATCH --partition=sganguli
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=results/logs/train_latent_resnet_%j.out
#SBATCH --error=results/logs/train_latent_resnet_%j.err

set -e  # Exit on error

# Activate environment
source ~/activate_diffusion.sh
export PYTHONPATH="/home/users/hshunt/convolutional_diffusion:$PYTHONPATH"

mkdir -p results/logs

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null

cd /home/users/hshunt/convolutional_diffusion

# Train ResNet on CelebA-HQ latents
# Architecture matches existing CelebA ResNet: 8 layers, emb_dim=256, mode=zeros
python scripts/training_script_latent.py \
    --latent_path /scratch/users/hshunt/celeba_hq_latents/celeba_hq_latents.pt \
    --resnet \
    --epochs 500 \
    --batchsize 128 \
    --lr 0.0001 \
    --mode zeros \
    --layers 8 \
    --mult 2 \
    --saveinterval 25 \
    --gamma 0.999965 \
    --homedir ./checkpoints

echo "Job finished at $(date)"
