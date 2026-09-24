#!/bin/bash
#SBATCH --job-name=dit_vpred
#SBATCH --output=results/logs/dit_vpred_%j.out
#SBATCH --error=results/logs/dit_vpred_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=sganguli
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# Train the v-prediction DiT and calibrate the local-score scales against it.
#
#   1. train_dit_vpred.py     -- MinimalDiT on the v objective, saved behind a
#                                VPredToEps wrapper so it presents an eps interface.
#   2. scales_calibration.py  -- per-timestep receptive-field scale k for the local
#                                score machine, matched to this network.
#
# The NTK fit and the three-way comparison then run through the usual driver:
#   MODELFILE=checkpoints/backbone_CelebA_DiT_vpred.pt \
#   SCALESFILE=checkpoints/scales_CelebA_DiT_vpred_median.pt \
#   ARCH=DiT SCORE_TYPE=LS sbatch scripts/slurm_ntk_vs_els.sh
#
# Usage:
#   sbatch scripts/slurm_dit_vpred.sh
#   SKIP_TRAIN=1 sbatch scripts/slurm_dit_vpred.sh   # recalibrate an existing checkpoint
#   DRY_RUN=1 sbatch scripts/slurm_dit_vpred.sh

ROOT="${SLURM_SUBMIT_DIR:-/home/users/hshunt/convolutional_diffusion}"
cd "$ROOT" || exit 1
export PYTHONPATH="$ROOT:$PYTHONPATH"

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
SAVENAME="${SAVENAME:-backbone_CelebA_DiT_vpred}"
EPOCHS="${EPOCHS:-200}"
SCORE_TYPE="${SCORE_TYPE:-LS}"
KERNELSIZES="${KERNELSIZES:-3 5 7 9 11 13 15 17 19}"

mkdir -p "${ROOT}/results/logs" "${ROOT}/checkpoints"

EXTRA=""
CAL_NSAMPS="${CAL_NSAMPS:-10}"
if [[ -n "${DRY_RUN}" ]]; then
	echo "DRY_RUN set: 2 epochs, tiny calibration."
	EXTRA="--dry_run"
	CAL_NSAMPS=2
	KERNELSIZES="3 7"
fi

echo "ROOT=${ROOT}  DATASET=${DATASET}  SAVENAME=${SAVENAME}  EPOCHS=${EPOCHS}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

set -e

if [[ -f "./checkpoints/${SAVENAME}.pt" && -n "${SKIP_TRAIN}" ]]; then
	echo "=== [1/2] SKIP_TRAIN set and checkpoint exists, reusing ==="
else
echo "=== [1/2] train DiT (v-prediction) ==="
python scripts/train_dit_vpred.py \
	--dataset "$DATASET" \
	--epochs "$EPOCHS" \
	--savename "$SAVENAME" \
	--homedir ./checkpoints \
	$EXTRA
fi

# scales_calibration.py joins --tld with --modelfile, so --modelfile is a bare
# filename relative to the checkpoints directory, not a path.
echo "=== [2/2] calibrate ${SCORE_TYPE} scales against the DiT ==="
python scripts/scales_calibration.py \
	--dataset "$DATASET" \
	--modelfile "${SAVENAME}.pt" \
	--kfilename "scales_CelebA_DiT_vpred" \
	--tld ./checkpoints/ \
	--scoremoduletype "$SCORE_TYPE" \
	--kernelsizes $KERNELSIZES \
	--nsteps 20 \
	--nsamps "$CAL_NSAMPS"

echo "Done. Checkpoints:"
ls -la "./checkpoints/${SAVENAME}.pt" ./checkpoints/scales_CelebA_DiT_vpred_*.pt
