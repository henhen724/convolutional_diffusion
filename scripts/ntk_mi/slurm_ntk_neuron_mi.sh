#!/bin/bash
#SBATCH --job-name=ntk_neuron_mi
#SBATCH --output=ntk_neuron_mi_%j.out
#SBATCH --error=ntk_neuron_mi_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=sganguli
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/home/users/hshunt/convolutional_diffusion}"
cd "$ROOT"

module purge 2>/dev/null
module load devel math \
    python/3.12.1 \
    py-pytorch/2.4.1_py312 \
    py-numpy/1.26.3_py312 \
    2>/dev/null
source "${HOME}/diffusion_venv_py312/bin/activate"

OUTDIR="${OUTDIR:-$SCRATCH/ntk_mi_results}"
mkdir -p "$OUTDIR"

D="${D:-32}"
M="${M:-2000}"
N="${N:-150}"
N_TRIALS="${N_TRIALS:-100}"
N_STEPS="${N_STEPS:-5000}"
N_SPLITS="${N_SPLITS:-15}"
LR="${LR:-0.2}"
BETA="${BETA:-0.5}"
SEED="${SEED:-0}"

for TASK in supervised denoising; do
    python scripts/ntk_mi/ntk_neuron_mi.py \
        --task "$TASK" \
        --d "$D" --m "$M" --n "$N" \
        --n_trials "$N_TRIALS" --n_steps "$N_STEPS" --n_splits "$N_SPLITS" \
        --lr "$LR" --beta "$BETA" --seed "$SEED" \
        --out "$OUTDIR/${TASK}_d${D}_m${M}_n${N}_v2.npz"
done

echo "Done. Results in $OUTDIR"
