#!/bin/bash
#SBATCH --job-name=k_cal_vs_k_mem
#SBATCH --partition=sganguli
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=results/logs/k_cal_vs_k_mem_%j.out
#SBATCH --error=results/logs/k_cal_vs_k_mem_%j.err

set -e

ROOT="${SLURM_SUBMIT_DIR:-/home/users/hshunt/convolutional_diffusion}"
cd "$ROOT"
export PYTHONPATH="$ROOT:$PYTHONPATH"

ml load py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312 viz py-matplotlib/3.10.3_py312

mkdir -p results/logs

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null

# DATASET / MODEL_PATH / CONDITIONAL are set by the submitting command (see below);
# fall back to CIFAR10 (conditional) so the job is runnable with no overrides.
DATASET="${DATASET:-cifar10}"
MODEL_PATH="${MODEL_PATH:-checkpoints/backbone_CIFAR10_UNet_zeros_conditional.pt}"
CONDITIONAL_FLAG="${CONDITIONAL_FLAG:---conditional}"

python scripts/k_cal_vs_k_mem_sweep.py \
    --dataset "$DATASET" \
    --model_path "$MODEL_PATH" \
    $CONDITIONAL_FLAG \
    --nsteps "${NSTEPS:-20}" \
    --nsamps "${NSAMPS:-4}" \
    --max_samples "${MAX_SAMPLES:-300}" \
    --score_batch_size "${SCORE_BATCH_SIZE:-64}"

echo "Job finished at $(date)"
