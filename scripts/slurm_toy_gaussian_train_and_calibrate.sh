#!/bin/bash
#SBATCH --job-name=toy_gaussian_bbresnet
#SBATCH --output=toy_gaussian_bbresnet_%j.out
#SBATCH --error=toy_gaussian_bbresnet_%j.err
#SBATCH --time=72:00:00
#SBATCH --partition=sganguli
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/home/users/hshunt/convolutional_diffusion}"
cd "$ROOT"
export PYTHONPATH="$ROOT:$PYTHONPATH"
source ~/activate_diffusion.sh

# -------------------------
# Config
# -------------------------
DATASET="${DATASET:-toy_gaussian_field}"
DATA_ROOT="${DATA_ROOT:-./data}"
DATA_DIRNAME="${DATA_DIRNAME:-toy_gaussian_field}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-200000}"
VALID_SAMPLES="${VALID_SAMPLES:-10000}"
IMAGE_SIZE="${IMAGE_SIZE:-32}"
CHANNELS="${CHANNELS:-3}"
ALPHA="${ALPHA:-3.0}"
AMPLITUDE="${AMPLITUDE:-}"  # empty => use default integral normalization
CHUNK_SIZE="${CHUNK_SIZE:-2048}"

HOMEDIR="${HOMEDIR:-./checkpoints}"
SAVENAME="${SAVENAME:-backbone_ToyGaussianField_ResNet_zeros}"
EPOCHS="${EPOCHS:-500}"
BATCHSIZE="${BATCHSIZE:-128}"
LR="${LR:-1e-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SAVE_INTERVAL="${SAVE_INTERVAL:-25}"
TRAIN_MAXSAMPS="${TRAIN_MAXSAMPS:-200000}"

MODEL_EPOCH="${MODEL_EPOCH:-499}"  # training_script saves epoch indices starting at 0
KERNELSIZES="${KERNELSIZES:-3 5 7 9 11 13 15}"
NSAMPS_CALIB="${NSAMPS_CALIB:-20}"
NSTEPS="${NSTEPS:-20}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-16}"
MAXSAMPS_CALIB="${MAXSAMPS_CALIB:-20000}"

# -------------------------
# Phase 1: Generate toy dataset
# -------------------------
echo "=== Phase 1: Generating toy Gaussian dataset ==="
GEN_CMD=(python scripts/generate_toy_gaussian_field_dataset.py
  --root "$DATA_ROOT"
  --dirname "$DATA_DIRNAME"
  --train_samples "$TRAIN_SAMPLES"
  --valid_samples "$VALID_SAMPLES"
  --image_size "$IMAGE_SIZE"
  --channels "$CHANNELS"
  --alpha "$ALPHA"
  --chunk_size "$CHUNK_SIZE"
)
if [[ -n "$AMPLITUDE" ]]; then
  GEN_CMD+=(--amplitude "$AMPLITUDE")
fi
"${GEN_CMD[@]}"

# -------------------------
# Phase 2: Train boundary-broken ResNet (mode=zeros)
# -------------------------
echo "=== Phase 2: Training boundary-broken ResNet on ${DATASET} ==="
python scripts/training_script.py \
  --dataset "$DATASET" \
  --resnet \
  --mode zeros \
  --epochs "$EPOCHS" \
  --saveinterval "$SAVE_INTERVAL" \
  --homedir "$HOMEDIR" \
  --savename "$SAVENAME" \
  --batchsize "$BATCHSIZE" \
  --lr "$LR" \
  --maxsamps "$TRAIN_MAXSAMPS" \
  --num_workers "$NUM_WORKERS"

MODELFILE="${SAVENAME}_epoch${MODEL_EPOCH}.pt"
if [[ ! -f "${HOMEDIR}/${MODELFILE}" ]]; then
  echo "Expected model not found: ${HOMEDIR}/${MODELFILE}"
  exit 2
fi

# -------------------------
# Phase 3: Cosine scale calibration for LS and bbELS
# -------------------------
echo "=== Phase 3: Cosine scale calibration (LS and bbELS) ==="
python scripts/scales_calibration.py \
  --modelfile "$MODELFILE" \
  --tld "$HOMEDIR" \
  --dataset "$DATASET" \
  --scoremoduletype LS \
  --eval_mode cos \
  --kernelsizes $KERNELSIZES \
  --kfilename "scales_${SAVENAME}_epoch${MODEL_EPOCH}_LS" \
  --nsamps "$NSAMPS_CALIB" \
  --nsteps "$NSTEPS" \
  --scorebatchsize "$SCORE_BATCH_SIZE" \
  --maxsamps "$MAXSAMPS_CALIB"

python scripts/scales_calibration.py \
  --modelfile "$MODELFILE" \
  --tld "$HOMEDIR" \
  --dataset "$DATASET" \
  --scoremoduletype bbELS \
  --eval_mode cos \
  --kernelsizes $KERNELSIZES \
  --kfilename "scales_${SAVENAME}_epoch${MODEL_EPOCH}_bbELS" \
  --nsamps "$NSAMPS_CALIB" \
  --nsteps "$NSTEPS" \
  --scorebatchsize "$SCORE_BATCH_SIZE" \
  --maxsamps "$MAXSAMPS_CALIB"

echo "Done."
