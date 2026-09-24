#!/bin/bash
#SBATCH --job-name=noise_identity
#SBATCH --output=results/logs/noise_identity_%j.out
#SBATCH --error=results/logs/noise_identity_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=sganguli
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

# Check kambm's low-noise identity (noise averaging == sigma^2 Hf Hf^T regularizer)
# both at the diffusion schedule's own noise levels and, via --sigma_override, at
# artificially small sigma where the second-order expansion must hold if it is right.

ROOT="${SLURM_SUBMIT_DIR:-/home/users/hshunt/convolutional_diffusion}"
cd "$ROOT" || exit 1
export PYTHONPATH="$ROOT:$PYTHONPATH"
if ! type module &>/dev/null; then
	for _init in /usr/share/lmod/lmod/init/bash /etc/profile.d/modules.sh; do
		[[ -f "$_init" ]] && source "$_init" && break
	done
fi
module purge 2>/dev/null
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312 \
	py-scipy/1.12.0_py312 py-numpy/1.26.3_py312 2>/dev/null
source "${SCRATCH}/diffusion_venv_py312/bin/activate"
set -e

echo "=== asymptotic check: artificial sigma sweep at fixed t ==="
python scripts/test_noise_averaging_identity.py --n_points 4 --n_noise 64 --n_probe 16 \
	--sigma_override 0.0001 0.0003 0.001 0.003 0.01 0.03 0.0778 0.3 --override_step 1 \
	--out_file results/ntk_score/noise_identity_sigma_sweep.pt

echo
echo "=== at the schedule's own noise levels ==="
python scripts/test_noise_averaging_identity.py --n_points 6 --n_noise 64 --n_probe 16 \
	--time_steps 1 2 3 5 10 \
	--out_file results/ntk_score/noise_identity_schedule.pt
