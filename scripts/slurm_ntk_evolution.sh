#!/bin/bash
#SBATCH --job-name=ntk_evolution
#SBATCH --output=results/logs/ntk_evolution_%j.out
#SBATCH --error=results/logs/ntk_evolution_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=sganguli
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# How fast does the empirical NTK move away from its value at initialization?
#
# STAGES=epoch  measure drift along the existing epoch-level trajectories
#               (CelebA ResNet 'fresh', every 10 epochs; DiT, every 20 epochs)
# STAGES=dense  train short runs checkpointed at log-spaced optimizer steps and
#               measure drift along those -- the epoch-level series cannot resolve
#               a transient that finishes inside the first saved interval
# STAGES=all    both
#
# Usage:
#   STAGES=epoch sbatch scripts/slurm_ntk_evolution.sh
#   STAGES=dense sbatch scripts/slurm_ntk_evolution.sh

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

STAGES="${STAGES:-all}"
N_EVAL="${N_EVAL:-64}"
TIME_STEPS="${TIME_STEPS:-1 3 5 10 20}"
MAX_STEPS="${MAX_STEPS:-4096}"
OUT_DIR="${ROOT}/results/ntk_score"
mkdir -p "$OUT_DIR" "${ROOT}/results/logs" "${ROOT}/checkpoints/dense"

echo "STAGES=${STAGES}  N_EVAL=${N_EVAL}  TIME_STEPS=${TIME_STEPS}  MAX_STEPS=${MAX_STEPS}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
set -e

if [[ "$STAGES" == "epoch" || "$STAGES" == "all" ]]; then
	echo "=== epoch-level: CelebA ResNet (fresh) ==="
	python scripts/ntk_kernel_evolution.py \
		--pattern 'checkpoints/backbone_CelebA_ResNet_fresh_epoch*.pt' \
		--label ResNet_epochs --n_eval "$N_EVAL" --time_steps $TIME_STEPS \
		--out_file "${OUT_DIR}/kernel_evolution_celeba_ResNet_epochs.pt"

	echo "=== epoch-level: CelebA DiT (v-prediction) ==="
	python scripts/ntk_kernel_evolution.py \
		--pattern 'checkpoints/backbone_CelebA_DiT_vpred_epoch*.pt' \
		--label DiT_epochs --n_eval "$N_EVAL" --time_steps $TIME_STEPS \
		--out_file "${OUT_DIR}/kernel_evolution_celeba_DiT_epochs.pt"
fi

if [[ "$STAGES" == "dense" || "$STAGES" == "all" ]]; then
	if [[ ! -f "checkpoints/dense/backbone_CelebA_resnet_dense_step0.pt" ]]; then
		echo "=== dense training: ResNet ==="
		python scripts/train_dense_checkpoints.py --arch resnet --max_steps "$MAX_STEPS"
	fi
	echo "=== dense: ResNet drift in step time ==="
	python scripts/ntk_kernel_evolution.py \
		--pattern 'checkpoints/dense/backbone_CelebA_resnet_dense_step*.pt' \
		--label ResNet_steps --n_eval "$N_EVAL" --time_steps $TIME_STEPS \
		--out_file "${OUT_DIR}/kernel_evolution_celeba_ResNet_steps.pt"

	if [[ ! -f "checkpoints/dense/backbone_CelebA_dit_dense_step0.pt" ]]; then
		echo "=== dense training: DiT ==="
		python scripts/train_dense_checkpoints.py --arch dit --max_steps "$MAX_STEPS"
	fi
	echo "=== dense: DiT drift in step time ==="
	python scripts/ntk_kernel_evolution.py \
		--pattern 'checkpoints/dense/backbone_CelebA_dit_dense_step*.pt' \
		--label DiT_steps --n_eval "$N_EVAL" --time_steps $TIME_STEPS \
		--out_file "${OUT_DIR}/kernel_evolution_celeba_DiT_steps.pt"
fi

echo "Done."
ls -la "${OUT_DIR}"/kernel_evolution_*.pt
