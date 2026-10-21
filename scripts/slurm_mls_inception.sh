#!/bin/bash
#SBATCH --job-name=mls_is
#SBATCH --partition=sganguli
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=results/mls_inception/logs/%x_%j.out
# Usage: sbatch [--time=...] scripts/slurm_mls_inception.sh <args for scripts/mls_inception.py>
# Several runs can be packed into one job by separating them with ' ::: '.
ml purge
ml devel math python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312 py-scipy/1.12.0_py312 py-numpy/1.26.3_py312
source $HOME/diffusion_venv_py312/bin/activate
export TORCH_HOME=$SCRATCH/.cache/torch
cd $HOME/convolutional_diffusion
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
run=()
for a in "$@" ":::"; do
	if [ "$a" = ":::" ]; then
		[ ${#run[@]} -gt 0 ] && python scripts/mls_inception.py "${run[@]}"
		run=()
	else
		run+=("$a")
	fi
done
