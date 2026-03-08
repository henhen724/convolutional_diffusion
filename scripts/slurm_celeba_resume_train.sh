#!/bin/bash
#SBATCH --job-name=celeba_resume
#SBATCH --output=celeba_resume_%j.out
#SBATCH --error=celeba_resume_%j.err
#SBATCH --time=72:00:00
#SBATCH --partition=sganguli
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# Resume CelebA ResNet training from epoch 39 up to 500.
ROOT="${SLURM_SUBMIT_DIR:-/home/users/hshunt/convolutional_diffusion}"
cd "$ROOT"
export PYTHONPATH="$ROOT:$PYTHONPATH"
source ~/activate_diffusion.sh

RESUME_EPOCH="${RESUME_EPOCH:-39}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-500}"
CKPT_DIR="${CKPT_DIR:-./checkpoints}"
SAVENAME="${SAVENAME:-backbone_CelebA_ResNet}"
RESUME_PATH="${CKPT_DIR}/${SAVENAME}_epoch${RESUME_EPOCH}.pt"

if [[ ! -f "$RESUME_PATH" ]]; then
  echo "ERROR: Resume checkpoint not found: $RESUME_PATH"
  exit 1
fi

echo "Resuming from $RESUME_PATH, training to epoch $TOTAL_EPOCHS"

python scripts/training_script.py \
  --dataset celeba \
  --resnet \
  --epochs 1 \
  --saveinterval 10 \
  --homedir "$CKPT_DIR" \
  --savename "$SAVENAME" \
  --resume "$RESUME_PATH" \
  --total_epochs "$TOTAL_EPOCHS" \
  --batchsize "${BATCHSIZE:-128}" \
  --lr "${LR:-0.0001}"

echo "Done."
