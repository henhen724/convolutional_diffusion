#!/bin/bash
#SBATCH --job-name=ntk_hypotheses
#SBATCH --output=results/logs/ntk_hypotheses_%j.out
#SBATCH --error=results/logs/ntk_hypotheses_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=sganguli
#SBATCH --mem=256G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# Why does the NTK prediction break down at low noise?
#
# Three hypotheses, tested on the CelebA ResNet at the low-noise steps only
# (--only_steps), which is where the NTK and the trained network disagree.
#
#   A. "too few noise realizations", at FIXED row budget (3072 readout rows):
#      trading clean images for noise draws per image. If averaging over noise is
#      what matters, A4 beats A1 even though A4 sees 8x fewer distinct images.
#
#   B. "too few noise realizations", at FIXED image set (512 images): adding
#      draws strictly adds rows. Separates "more noise coverage" from "more rows".
#
#   C. "the time embedding is hard for the NTK": refit with the time-conditioning
#      pathway frozen out of the kernel entirely. If the fit does not degrade, the
#      NTK was not relying on it.
#
# Held-out R^2 is measured on the same held-out images and the same readout pixels
# in every configuration, so the numbers are directly comparable across rows.

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

MODELFILE="${MODELFILE:-checkpoints/backbone_CelebA_ResNet_zeros.pt}"
STEPS="${STEPS:-1 2 3 5}"
OUT_DIR="${ROOT}/results/ntk_score/hypotheses"
mkdir -p "$OUT_DIR" "${ROOT}/results/logs"

echo "MODELFILE=${MODELFILE}  STEPS=${STEPS}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
set -e

run () {  # run <tag> <extra args...>
	local tag="$1"; shift
	local out="${OUT_DIR}/fit_${tag}.pt"
	if [[ -f "$out" && -z "${REDO}" ]]; then
		echo "=== ${tag}: present, skipping (set REDO=1) ==="
		return
	fi
	echo "=== ${tag} ==="
	python scripts/fit_ntk_score.py \
		--modelfile "$MODELFILE" \
		--only_steps $STEPS \
		--out_file "$out" \
		"$@"
}

# A: fixed 3072 readout rows, trading images for noise draws
run A1_img1024_rep1 --n_train 1024 --noise_reps 1
run A2_img512_rep2  --n_train 512  --noise_reps 2
run A3_img256_rep4  --n_train 256  --noise_reps 4
run A4_img128_rep8  --n_train 128  --noise_reps 8

# B: fixed 512-image set, adding noise draws on top
run B1_img512_rep1  --n_train 512  --noise_reps 1
run B3_img512_rep4  --n_train 512  --noise_reps 4

# C: time-conditioning pathway frozen out of the kernel
run C1_freeze_time  --n_train 1024 --noise_reps 1 --freeze embs. embedding.

echo
echo "================ SUMMARY: held-out R^2 at the readout ================"
python - <<'PY'
import glob, os, torch
rows = []
for f in sorted(glob.glob('results/ntk_score/hypotheses/fit_*.pt')):
    a = torch.load(f, map_location='cpu', weights_only=False)
    tag = os.path.basename(f)[4:-3]
    for i in sorted(a['diagnostics']):
        d = a['diagnostics'][i]
        rows.append((tag, a['n_train'], a['noise_reps'], a['rows'], i, d['beta_t'],
                     d['ridge_rel'], d['holdout_r2'], d['delta_norm']))
print(f"{'config':<20}{'imgs':>6}{'reps':>6}{'rows':>7}{'step':>6}{'beta':>9}"
      f"{'ridge':>9}{'R2_hold':>10}{'|dtheta|':>11}")
for r in rows:
    print(f'{r[0]:<20}{r[1]:>6}{r[2]:>6}{r[3]:>7}{r[4]:>6}{r[5]:>9.4f}'
          f'{r[6]:>9.0e}{r[7]:>+10.4f}{r[8]:>11.3e}')
PY
