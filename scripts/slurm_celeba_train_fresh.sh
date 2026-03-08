#!/bin/bash
#SBATCH --job-name=celeba_fresh
#SBATCH --output=celeba_fresh_%j.out
#SBATCH --error=celeba_fresh_%j.err
#SBATCH --time=72:00:00
#SBATCH --partition=sganguli
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# Fresh CelebA ResNet DDIM training from scratch.

ROOT="${SLURM_SUBMIT_DIR:-/home/users/hshunt/convolutional_diffusion}"
cd "$ROOT"
export PYTHONPATH="$ROOT:$PYTHONPATH"
source ~/activate_diffusion.sh

CKPT_DIR="${CKPT_DIR:-./checkpoints}"
SAVENAME="${SAVENAME:-backbone_CelebA_ResNet_fresh}"
EPOCHS="${EPOCHS:-500}"

echo "Starting fresh CelebA ResNet training for ${EPOCHS} epochs"

python scripts/training_script.py \
  --dataset celeba \
  --resnet \
  --mode zeros \
  --epochs "$EPOCHS" \
  --saveinterval 10 \
  --homedir "$CKPT_DIR" \
  --savename "$SAVENAME" \
  --batchsize "${BATCHSIZE:-128}" \
  --lr "${LR:-0.0001}"

echo "Done."

