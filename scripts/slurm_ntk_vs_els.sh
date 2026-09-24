#!/bin/bash
#SBATCH --job-name=ntk_vs_els
#SBATCH --output=results/logs/ntk_vs_els_%j.out
#SBATCH --error=results/logs/ntk_vs_els_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=sganguli
#SBATCH --mem=256G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# Companion compute for notebooks/ntk_vs_els_samples.ipynb.
#
#   1. fit_ntk_score.py        -- per-noise-level NTK kernel-ridge fit of a trained
#                                 backbone's denoiser, saved as one dtheta per step.
#   2. gen_score_comparison.py -- samples from the trained network, its NTK
#                                 prediction, and a local score machine, all from
#                                 shared seed noise.
#
# --mem is sized for the gradient matrix G: (n_train * channels) x n_params, which
# is ~65 GB in fp32 at n_train=1024 on the 5.3M-parameter CelebA ResNet.
#
# Usage:
#   sbatch scripts/slurm_ntk_vs_els.sh                                  # ResNet + bbELS
#   ARCH=UNet SCORE_TYPE=LS sbatch scripts/slurm_ntk_vs_els.sh          # UNet + LS
#   DRY_RUN=1 sbatch scripts/slurm_ntk_vs_els.sh                        # smoke test

ROOT="${SLURM_SUBMIT_DIR:-/home/users/hshunt/convolutional_diffusion}"
cd "$ROOT" || exit 1
export PYTHONPATH="$ROOT:$PYTHONPATH"

# Same module stack as ~/jupyter_kernel_diffusion_env.sh, so the job and the
# notebook kernel run against identical torch/torchvision builds.
if ! type module &>/dev/null; then
	for _init in /usr/share/lmod/lmod/init/bash /etc/profile.d/modules.sh; do
		[[ -f "$_init" ]] && source "$_init" && break
	done
fi
module purge 2>/dev/null
module load devel math \
	python/3.12.1 \
	py-pytorch/2.4.1_py312 \
	py-torchvision/0.19.1_py312 \
	py-scipy/1.12.0_py312 \
	py-scikit-learn/1.5.1_py312 \
	py-numpy/1.26.3_py312 \
	2>/dev/null
source "${SCRATCH}/diffusion_venv_py312/bin/activate"

DATASET="${DATASET:-celeba}"
ARCH="${ARCH:-ResNet}"
SCORE_TYPE="${SCORE_TYPE:-bbELS}"
MODELFILE="${MODELFILE:-checkpoints/backbone_CelebA_${ARCH}_zeros.pt}"
SCALESFILE="${SCALESFILE:-checkpoints/scales_CelebA_${ARCH}_zeros.pt}"
N_TRAIN="${N_TRAIN:-1024}"
N_SAMPLES="${N_SAMPLES:-8}"
LOCAL_MAX_SAMPLES="${LOCAL_MAX_SAMPLES:-10000}"
# float32 by default: the kernel-ridge coefficients are large where the Gram is
# ill-conditioned (the low-noise steps), and fp16 gradient rows there left dtheta
# reproducing the kernel solution to only ~5% on the ResNet and ~30% on the UNet.
GRAD_DTYPE="${GRAD_DTYPE:-float32}"

OUT_DIR="${OUT_DIR:-${ROOT}/results/ntk_score}"
mkdir -p "$OUT_DIR" "${ROOT}/results/logs"

EXTRA=""
if [[ -n "${DRY_RUN}" ]]; then
	echo "DRY_RUN set: tiny fit and 2 samples."
	EXTRA="--dry_run"
	N_TRAIN=16
fi

TAG="${DATASET}_${ARCH}_${SCORE_TYPE}_N${N_TRAIN}"
FIT_FILE="${OUT_DIR}/ntk_fit_${DATASET}_${ARCH}_N${N_TRAIN}.pt"
SAMP_FILE="${OUT_DIR}/comparison_${TAG}.pt"

echo "ROOT=${ROOT}"
echo "ARCH=${ARCH}  SCORE_TYPE=${SCORE_TYPE}"
echo "MODELFILE=${MODELFILE}  SCALESFILE=${SCALESFILE}"
echo "N_TRAIN=${N_TRAIN}  N_SAMPLES=${N_SAMPLES}  LOCAL_MAX_SAMPLES=${LOCAL_MAX_SAMPLES}"
echo "FIT_FILE=${FIT_FILE}"
echo "SAMP_FILE=${SAMP_FILE}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

set -e

if [[ -f "$FIT_FILE" && -z "${REFIT}" ]]; then
	echo "=== [1/2] NTK fit already present, reusing (set REFIT=1 to redo) ==="
else
	echo "=== [1/2] NTK kernel-ridge fit ==="
	python scripts/fit_ntk_score.py \
		--dataset "$DATASET" \
		--modelfile "$MODELFILE" \
		--n_train "$N_TRAIN" \
		--grad_dtype "$GRAD_DTYPE" \
		--out_file "$FIT_FILE" \
		$EXTRA
fi

echo "=== [2/2] score comparison + sample generation ==="
python scripts/gen_score_comparison.py \
	--dataset "$DATASET" \
	--modelfile "$MODELFILE" \
	--scalesfile "$SCALESFILE" \
	--ntk_fit "$FIT_FILE" \
	--score_module_type "$SCORE_TYPE" \
	--network_label "$ARCH" \
	--n_samples "$N_SAMPLES" \
	--local_max_samples "$LOCAL_MAX_SAMPLES" \
	--out_file "$SAMP_FILE" \
	$EXTRA

echo "Done. Artifacts:"
ls -la "$FIT_FILE" "$SAMP_FILE"
