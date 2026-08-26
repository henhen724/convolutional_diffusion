#!/bin/bash
#SBATCH --job-name=ntk_width_scan
#SBATCH --output=ntk_width_scan_%j.out
#SBATCH --error=ntk_width_scan_%j.err
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

OUTDIR="${OUTDIR:-$SCRATCH/ntk_mi_results/width_scan}"
mkdir -p "$OUTDIR"

D="${D:-16}"
N="${N:-150}"
N_STEPS="${N_STEPS:-3000}"
N_TRIALS="${N_TRIALS:-100}"
N_SPLITS="${N_SPLITS:-20}"
LR="${LR:-0.2}"
TASK="${TASK:-supervised}"
SEED="${SEED:-0}"

for M in 20 50 150 500 1500 3000; do
    python scripts/ntk_mi/ntk_neuron_mi.py \
        --task "$TASK" \
        --d "$D" --m "$M" --n "$N" \
        --n_trials "$N_TRIALS" --n_steps "$N_STEPS" --n_splits "$N_SPLITS" \
        --lr "$LR" --seed "$SEED" \
        --out "$OUTDIR/${TASK}_d${D}_m${M}_n${N}.npz"
done

echo "Done. Results in $OUTDIR"
