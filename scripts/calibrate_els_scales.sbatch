#!/bin/bash
#SBATCH --job-name=els_calib
#SBATCH --partition=sganguli
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=results/logs/calibrate_els_scales_%j.out
#SBATCH --error=results/logs/calibrate_els_scales_%j.err

set -e

source ~/activate_diffusion.sh
export PYTHONPATH="/home/users/hshunt/convolutional_diffusion:$PYTHONPATH"

mkdir -p results/logs

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null

cd /home/users/hshunt/convolutional_diffusion

python scripts/calibrate_els_scales.py \
    --unet_path   checkpoints/backbone_CelebAHQ_latent_UNet_zeros_nonorm_final.pt \
    --resnet_path  checkpoints/backbone_CelebAHQ_latent_ResNet_zeros_nonorm_final.pt \
    --latent_path  /scratch/users/hshunt/celeba_hq_latents/celeba_hq_latents.pt \
    --stats_path   /scratch/users/hshunt/celeba_hq_latents/celeba_hq_latent_stats.pt \
    --output       checkpoints/els_calibrated_scales.pt \
    --nsteps       50 \
    --n_calib      4 \
    --max_samples  2000 \
    --batch_size   64 \
    --candidate_ks 3,5,7,9,11,15,19,23,27,31

echo "Job finished at $(date)"
