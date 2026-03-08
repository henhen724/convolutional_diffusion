#!/bin/bash
#SBATCH --job-name=channel_beta_snr
#SBATCH --output=channel_theory_beta_vs_snr_%j.out
#SBATCH --error=channel_theory_beta_vs_snr_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=sganguli
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

# Run from the directory where `sbatch` was invoked.
# NOTE: Slurm runs scripts from a spool directory, so `$0` can't be used to find the repo.
ROOT="${SLURM_SUBMIT_DIR:-/home/users/hshunt/convolutional_diffusion}"
cd "$ROOT"
export PYTHONPATH="$ROOT:$PYTHONPATH"
source ~/activate_diffusion.sh

DATASET="${DATASET:-celeba}"
TLD="${TLD:-./checkpoints}"
OUT_FILE="${OUT_FILE:-results/channel_theory/beta_vs_snr_center_variance_${DATASET}.pt}"

# Default scale files so the job does not fail when env vars are unset.
# Override SCALE_FILE_LS / SCALE_FILE_ELS when submitting if needed.
if [[ -z "${SCALE_FILE_LS}" && -z "${SCALE_FILE_ELS}" ]]; then
  if [[ "$DATASET" == "celeba" ]]; then
    SCALE_FILE_LS="scales_backbone_CelebA_ResNet_epoch0_LS_median.pt"
    SCALE_FILE_ELS="scales_backbone_CelebA_ResNet_epoch0_ELS_median.pt"
  elif [[ "$DATASET" == "mnist" ]]; then
    SCALE_FILE_LS="scales_MNIST_ResNet_zeros_LS_median.pt"
    SCALE_FILE_ELS="scales_MNIST_ResNet_zeros_ELS_median.pt"
  fi
fi

python scripts/channel_theory_beta_vs_snr_center_variance.py \
  --dataset "$DATASET" \
  --tld "$TLD" \
  --nsteps "${NSTEPS:-20}" \
  --nsamples "${NSAMPLES:-64}" \
  --sample_batch_size "${SAMPLE_BATCH_SIZE:-8}" \
  --out_file "$OUT_FILE" \
  --kernelsizes ${KERNELSIZES:-3 5 7 9 11 13 15} \
  --score_batch_size "${SCORE_BATCH_SIZE:-8}" \
  --max_samples "${MAX_SAMPLES:-200}" \
  ${SCALE_FILE_LS:+--scale_file_ls "$SCALE_FILE_LS"} \
  ${SCALE_FILE_ELS:+--scale_file_els "$SCALE_FILE_ELS"} \
  "$@"
