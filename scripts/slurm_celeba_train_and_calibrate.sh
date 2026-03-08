#!/bin/bash
#SBATCH --job-name=celeba_train_cal
#SBATCH --output=celeba_train_calibrate_%j.out
#SBATCH --error=celeba_train_calibrate_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=sganguli
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# Train ResNet on CelebA with many early checkpoints, then calibrate ELS/LS scales
# (cosine) and run beta vs SNR for each checkpoint.
ROOT="${SLURM_SUBMIT_DIR:-/home/users/hshunt/convolutional_diffusion}"
cd "$ROOT"
export PYTHONPATH="$ROOT:$PYTHONPATH"
source ~/activate_diffusion.sh

DATASET="${DATASET:-celeba}"
SAVENAME="${SAVENAME:-backbone_CelebA_ResNet}"
HOMEDIR="${HOMEDIR:-./checkpoints}"
EPOCHS="${EPOCHS:-40}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1}"
KERNELSIZES="${KERNELSIZES:-3 5 7 9 11 13 15}"
# Epochs to run calibration and beta vs SNR (subset to keep job time reasonable)
CALIB_EPOCHS="${CALIB_EPOCHS:-0 1 2 3 4 5 10 15 20 25 30 35}"

# Dry run: 1 epoch training, calibrate and beta-vs-SNR only for epoch 0 (small data for speed)
if [[ -n "${DRY_RUN}" ]]; then
  EPOCHS=1
  CALIB_EPOCHS="0"
  DRY_RUN_MAXSAMPS="${DRY_RUN_MAXSAMPS:-500}"
  echo "DRY_RUN: training 1 epoch (maxsamps=$DRY_RUN_MAXSAMPS), calibrating only epoch 0"
fi

set -e

echo "=== Phase 1: Training ResNet on ${DATASET} ==="
TRAIN_EXTRA=""
if [[ -n "${DRY_RUN}" ]]; then
  TRAIN_EXTRA="--maxsamps ${DRY_RUN_MAXSAMPS:-500} --suppress"
fi
python scripts/training_script.py \
  --dataset "$DATASET" \
  --resnet \
  --epochs "$EPOCHS" \
  --saveinterval "$SAVE_INTERVAL" \
  --homedir "$HOMEDIR" \
  --savename "$SAVENAME" \
  --batchsize "${BATCHSIZE:-128}" \
  --lr "${LR:-0.0001}" \
  $TRAIN_EXTRA

echo "=== Phase 2: Calibrate ELS and LS scales (cosine) for selected epochs ==="
for ep in $CALIB_EPOCHS; do
  modelfile="${HOMEDIR}/${SAVENAME}_epoch${ep}.pt"
  if [[ ! -f "$modelfile" ]]; then
    echo "Skip epoch $ep: $modelfile not found"
    continue
  fi
  echo "Calibrating ELS for epoch $ep ..."
  python scripts/scales_calibration.py \
    --modelfile "${SAVENAME}_epoch${ep}.pt" \
    --tld "$HOMEDIR" \
    --dataset "$DATASET" \
    --scoremoduletype ELS \
    --eval_mode cos \
    --kernelsizes $KERNELSIZES \
    --kfilename "scales_${SAVENAME}_epoch${ep}_ELS" \
    --nsamps "${NSAMPS_CALIB:-20}" \
    --nsteps "${NSTEPS:-20}" \
    --scorebatchsize "${SCORE_BATCH_SIZE:-16}" \
    --maxsamps "${MAXSAMPS_CALIB:-2000}"
  echo "Calibrating LS for epoch $ep ..."
  python scripts/scales_calibration.py \
    --modelfile "${SAVENAME}_epoch${ep}.pt" \
    --tld "$HOMEDIR" \
    --dataset "$DATASET" \
    --scoremoduletype LS \
    --eval_mode cos \
    --kernelsizes $KERNELSIZES \
    --kfilename "scales_${SAVENAME}_epoch${ep}_LS" \
    --nsamps "${NSAMPS_CALIB:-20}" \
    --nsteps "${NSTEPS:-20}" \
    --scorebatchsize "${SCORE_BATCH_SIZE:-16}" \
    --maxsamps "${MAXSAMPS_CALIB:-2000}"
done

echo "=== Phase 3: Beta vs SNR for each calibrated checkpoint ==="
mkdir -p results/channel_theory
for ep in $CALIB_EPOCHS; do
  scale_ls="${HOMEDIR}/scales_${SAVENAME}_epoch${ep}_LS_median.pt"
  scale_els="${HOMEDIR}/scales_${SAVENAME}_epoch${ep}_ELS_median.pt"
  if [[ ! -f "$scale_ls" || ! -f "$scale_els" ]]; then
    echo "Skip beta vs SNR epoch $ep: scale files missing"
    continue
  fi
  outfile="results/channel_theory/beta_vs_snr_${SAVENAME}_epoch${ep}.pt"
  echo "Beta vs SNR for epoch $ep -> $outfile"
  python scripts/channel_theory_beta_vs_snr_center_variance.py \
    --dataset "$DATASET" \
    --nsteps "${NSTEPS:-20}" \
    --nsamples "${NSAMPLES:-64}" \
    --sample_batch_size "${SAMPLE_BATCH_SIZE:-8}" \
    --out_file "$outfile" \
    --tld "$HOMEDIR" \
    --kernelsizes $KERNELSIZES \
    --scale_file_ls "scales_${SAVENAME}_epoch${ep}_LS_median.pt" \
    --scale_file_els "scales_${SAVENAME}_epoch${ep}_ELS_median.pt" \
    --score_batch_size "${SCORE_BATCH_SIZE:-8}" \
    --max_samples "${MAX_SAMPLES:-200}"
done

echo "Done."
