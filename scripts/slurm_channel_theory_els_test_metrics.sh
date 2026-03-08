#!/bin/bash
# Run channel theory test metrics for ELS or LS.
# Switch with: SCORE_MODULE=ls sbatch ... or SCORE_MODULE=els sbatch ... (default: els)
#SBATCH --job-name=ct_test_metrics
#SBATCH --output=channel_theory_test_metrics_%j.out
#SBATCH --error=channel_theory_test_metrics_%j.err
#SBATCH --time=8:00:00
#SBATCH --partition=sganguli
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

ROOT="${SLURM_SUBMIT_DIR:-/home/users/hshunt/convolutional_diffusion}"
cd "$ROOT"

source ~/activate_diffusion.sh

SCORE_MODULE="${SCORE_MODULE:-els}"
DATASET="${DATASET:-mnist}"
OUT_FILE="${OUT_FILE:-results/channel_theory/${SCORE_MODULE}_test_metrics_${DATASET}.pt}"
python scripts/channel_theory_els_test_metrics.py \
  --score_module "$SCORE_MODULE" \
  --dataset "$DATASET" \
  --nsteps "${NSTEPS:-20}" \
  --ntest "${NTEST:-200}" \
  --test_batch_size "${TEST_BATCH_SIZE:-8}" \
  --out_file "$OUT_FILE" \
  --kernelsizes ${KERNELSIZES:-3 5 7 9 11 13 15} \
  --score_batch_size "${SCORE_BATCH_SIZE:-8}" \
  --max_samples "${MAX_SAMPLES:-500}" \
  "$@"
