#!/bin/bash
#SBATCH --job-name=resnet_patch_ntk
#SBATCH --output=resnet_patch_ntk_%j.out
#SBATCH --error=resnet_patch_ntk_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=sganguli
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

# Compute a ResNet input-gradient kernel (NTK proxy) restricted to a local patch
# around a single pixel, and compare it offline to exp(-beta ||x - x'||^2).

ROOT="${SLURM_SUBMIT_DIR:-/home/users/hshunt/convolutional_diffusion}"
cd "$ROOT"
export PYTHONPATH="$ROOT:$PYTHONPATH"
source ~/activate_diffusion.sh

DATASET="${DATASET:-celeba}"
MODEL_EPOCH="${MODEL_EPOCH:-0}"
MODELFILE="${MODELFILE:-${ROOT}/checkpoints/backbone_CelebA_ResNet_epoch${MODEL_EPOCH}.pt}"

PIXEL_I="${PIXEL_I:-16}"
PIXEL_J="${PIXEL_J:-16}"
PATCH_SIZE="${PATCH_SIZE:-11}"   # odd, local patch
TIME_STEPS="${TIME_STEPS:-10}"   # space-separated list of discrete time-step indices

N_TRAIN="${N_TRAIN:-256}"
N_TEST="${N_TEST:-256}"
NSTEPS="${NSTEPS:-20}"

OUT_DIR="${OUT_DIR:-${ROOT}/results/ntk}"
mkdir -p "$OUT_DIR"

# Dry run: very small subsets and a single time step, just to check for crashes.
if [[ -n "${DRY_RUN}" ]]; then
  echo "DRY_RUN set: using tiny subsets and a single time step."
  N_TRAIN=4
  N_TEST=4
  TIME_STEPS="10"
fi

echo "ROOT=${ROOT}"
echo "DATASET=${DATASET}"
echo "MODELFILE=${MODELFILE}"
echo "PIXEL=(${PIXEL_I}, ${PIXEL_J}), PATCH_SIZE=${PATCH_SIZE}"
echo "N_TRAIN=${N_TRAIN}, N_TEST=${N_TEST}, NSTEPS=${NSTEPS}"
echo "TIME_STEPS=${TIME_STEPS}"
echo "OUT_DIR=${OUT_DIR}"

set -e

for TS in $TIME_STEPS; do
  OUT_FILE="${OUT_DIR}/patch_ntk_${DATASET}_backbone_CelebA_ResNet_epoch${MODEL_EPOCH}_t${TS}_k${PATCH_SIZE}.pt"
  echo "Running NTK computation for time_step=${TS}, saving to ${OUT_FILE}"
  python scripts/resnet_patch_ntk.py \
    --dataset "$DATASET" \
    --modelfile "$MODELFILE" \
    --n_train "$N_TRAIN" \
    --n_test "$N_TEST" \
    --time_step "$TS" \
    --nsteps "$NSTEPS" \
    --pixel_i "$PIXEL_I" \
    --pixel_j "$PIXEL_J" \
    --patch_size "$PATCH_SIZE" \
    --out_file "$OUT_FILE"
done

echo "ResNet patch NTK job complete."

