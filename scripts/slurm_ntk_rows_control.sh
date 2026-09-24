#!/bin/bash
#SBATCH --job-name=ntk_rows
#SBATCH --output=results/logs/ntk_rows_%j.out
#SBATCH --error=results/logs/ntk_rows_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=sganguli
#SBATCH --mem=300G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# Control for the noise-realization sweep: is the low-noise NTK fit limited by
# *noise coverage*, or just by the total number of readout rows?
#
# Arm B showed held-out R^2 rising 0.12 -> 0.26 -> 0.43 as a fixed 512-image set
# gained noise draws (1536 -> 3072 -> 6144 rows). But rows grew too. These runs
# match the row counts using extra *images* at one noise draw each:
#
#   6144  rows:  D1 (2048 img x 1)  vs  B3 (512 img x 4)
#   12288 rows:  D2 (4096 img x 1)  vs  D3 (512 img x 8)
#
# If the matched pairs agree, the binding constraint is sample size and noise
# averaging is not special. D2/D3 also extend the scaling curve, which is what
# says whether the NTK would ever catch up given more data.

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

OUT_DIR="${ROOT}/results/ntk_score/hypotheses"
mkdir -p "$OUT_DIR"
STEPS="${STEPS:-1 2 3 5}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
set -e

run () {
	local tag="$1"; shift
	local out="${OUT_DIR}/fit_${tag}.pt"
	if [[ -f "$out" && -z "${REDO}" ]]; then echo "=== ${tag}: present, skipping ==="; return; fi
	echo "=== ${tag} ==="
	python scripts/fit_ntk_score.py --only_steps $STEPS --out_file "$out" "$@"
}

# 6144 rows, via images instead of noise draws
run D1_img2048_rep1 --n_train 2048 --noise_reps 1 --grad_dtype float32

# 12288 rows: images-only vs noise-only. float16 rows here -- 12288 x 5.3M in
# float32 would be 262 GB; the per-noise-level ridge selection absorbs the extra
# gradient round-off (it is chosen on held-out R^2, which is the metric compared).
run D2_img4096_rep1 --n_train 4096 --noise_reps 1 --grad_dtype float16 --chunk 384
run D3_img512_rep8  --n_train 512  --noise_reps 8 --grad_dtype float16 --chunk 384

echo
echo "================ ROW-SCALING SUMMARY ================"
python - <<'PY'
import glob, os, torch
rows = []
for f in sorted(glob.glob('results/ntk_score/hypotheses/fit_*.pt')):
    a = torch.load(f, map_location='cpu', weights_only=False)
    tag = os.path.basename(f)[4:-3]
    for i in sorted(a['diagnostics']):
        d = a['diagnostics'][i]
        rows.append((tag, a['n_train'], a['noise_reps'], a['rows'], i,
                     d['ridge_rel'], d['holdout_r2']))
rows.sort(key=lambda r: (r[4], r[3], r[2]))
print(f"{'step':>5}{'rows':>7}{'imgs':>6}{'reps':>6}  {'config':<20}{'ridge':>9}{'R2_hold':>10}")
for r in rows:
    print(f'{r[4]:>5}{r[3]:>7}{r[1]:>6}{r[2]:>6}  {r[0]:<20}{r[5]:>9.0e}{r[6]:>+10.4f}')
PY
