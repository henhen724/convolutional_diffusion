#!/bin/bash
#SBATCH --job-name=ntk_draws
#SBATCH --output=results/logs/ntk_draws_%j.out
#SBATCH --error=results/logs/ntk_draws_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=sganguli
#SBATCH --mem=300G
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1

# How many noise draws per image before the NTK fit stops improving?
#
# Two image budgets are swept over draws, both topping out at 49152 readout rows:
#
#   n_train=128:  reps 1 .. 64
#   n_train=512:  reps 1 .. 16
#
# Running two budgets is what makes the result interpretable. If R^2 saturates in
# draws at a level that depends on the image count, the asymptote is set by how many
# distinct images the regression saw and extra draws cannot buy past it. If both
# curves converge to the same ceiling, draws and images really are interchangeable.
#
# Low-noise steps only -- that is where the NTK/network gap lives and where the
# earlier sweeps showed the fit is variance-limited.
#
# --mem is sized for G at 24576 rows x 5.33M params in float16 (~262 GB), which is
# also what caps the sweep: sganguli allows at most 32 GB per core, so the memory
# request sets a floor on the CPU request (900G would demand 29 cores, more than the
# GPU nodes have free). Going past ~25k rows needs either sharding G across several
# GPUs or a blocked Gram that recomputes gradients, neither of which is in place.

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

OUT_DIR="${ROOT}/results/ntk_score/draws"
mkdir -p "$OUT_DIR"
STEPS="${STEPS:-1 3}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
set -e

run () {
	local imgs="$1" reps="$2"
	local tag="E_img${imgs}_rep${reps}"
	local out="${OUT_DIR}/fit_${tag}.pt"
	if [[ -f "$out" && -z "${REDO}" ]]; then echo "=== ${tag}: present, skipping ==="; return; fi
	echo "=== ${tag}  (rows = $((imgs * reps * 3))) ==="
	python scripts/fit_ntk_score.py \
		--only_steps $STEPS --n_train "$imgs" --noise_reps "$reps" \
		--grad_dtype float16 --chunk 384 --out_file "$out"
}

# Cheapest first, so a wall-clock overrun still leaves a usable curve.
for r in 1 2 4 8 16 32 64; do run 128 "$r"; done
for r in 1 2 4 8 16;        do run 512 "$r"; done

echo
echo "================ DRAW-SATURATION SUMMARY ================"
python - <<'PY'
import glob, os, torch
rows = []
for f in sorted(glob.glob('results/ntk_score/draws/fit_*.pt')):
    a = torch.load(f, map_location='cpu', weights_only=False)
    for i in sorted(a['diagnostics']):
        d = a['diagnostics'][i]
        rows.append((i, a['n_train'], a['noise_reps'], a['rows'], d['ridge_rel'], d['holdout_r2']))
rows.sort()
print(f"{'step':>5}{'imgs':>6}{'reps':>6}{'rows':>8}{'ridge':>9}{'R2_hold':>10}{'gain':>8}")
prev = {}
for st, im, rp, rw, rg, r2 in rows:
    g = r2 - prev.get((st, im), r2)
    prev[(st, im)] = r2
    print(f'{st:>5}{im:>6}{rp:>6}{rw:>8}{rg:>9.0e}{r2:>+10.4f}{g:>+8.4f}')
PY
