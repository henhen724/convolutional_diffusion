#!/bin/bash
#SBATCH --job-name=rhm_refresh
#SBATCH --partition=sganguli
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/rhm_refresh_%j.out

# Re-runs every audit analysis on the widened L=3 side the moment 51244750's
# array finishes, so the deck's numbers are final without anyone polling the
# scheduler for four hours.  Submitted with --dependency=afterok:51244750.
#
# Nothing here is new science: it is the same four scripts already in the
# repo, re-pointed at a results directory that now has 10 rule seeds at
# v=2 and 8 at v=3 for L=3 instead of 3 each.  The one addition is the
# copy_rel sensitivity sweep -- the relative copying-regime cut has a free
# parameter (0.5), and if the depth verdict moved with it the cut would be
# no better than the absolute P floor it replaced.  Partial-output check on
# 2026-11-05, with 7 of 10 v=2 seeds landed: v=2 reads 1.37x..2.06x and v=3
# 0.51x..1.09x across copy_rel in [0.25, 0.8], every interval covering 1,
# and nothing anywhere near the 2.8x the absolute floor produced.
#
# Deliberately does NOT upload: the OneDrive step needs the slide numbers
# read and the headline written by hand.

set -euo pipefail
cd "$HOME/convolutional_diffusion"

ml purge
ml devel math python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312 py-numpy/1.26.3_py312
source "$HOME/diffusion_venv_py312/bin/activate"

OUT=results/rhm_sampling/2026-11-05
mkdir -p "$OUT" logs

echo "######## 1. copying-regime diagnostic"
python scripts/rhm_copy_regime.py --out "$OUT/copy_regime.json" \
    | tee "$OUT/copy_regime.txt"

echo "######## 2. depth test, absolute P floor (the defective cut, kept for the deck)"
python scripts/rhm_depth2d_analysis.py --P_min 10 --copy_rel 0 \
    --out "$OUT/depth2d_pmin10.json" | tee "$OUT/depth2d_pmin10.txt"

echo "######## 3. depth test, copying-regime cut (the one to quote)"
python scripts/rhm_depth2d_analysis.py --P_min 0 --copy_rel 0.5 \
    --out "$OUT/depth2d_copyrel.json" | tee "$OUT/depth2d_copyrel.txt"

echo "######## 4. does the verdict depend on where copy_rel is set?"
{
  for CR in 0.25 0.35 0.5 0.65 0.8; do
    echo "######## copy_rel $CR"
    python scripts/rhm_depth2d_analysis.py --P_min 0 --copy_rel "$CR" \
        | sed -n '/paired depth test/,/reading/p'
  done
} | tee "$OUT/depth2d_copyrel_sens.txt"

echo "######## 5. x* scaling, both |R_L| floors"
python scripts/rhm_xstar_scaling.py --min_RL 200 \
    --out "$OUT/xstar_scaling_mr200.json" | tee "$OUT/xstar_scaling_mr200.txt"
python scripts/rhm_xstar_scaling.py --min_RL 1000 \
    --out "$OUT/xstar_scaling_mr1000.json" | tee "$OUT/xstar_scaling_mr1000.txt"

echo "######## 6. re-render the deck (traces re-run run_cell's exact seeds)"
python scripts/build_audit_slides.py \
    --out notebooks/2026-11-05_rhm_audit_slides.pdf \
    --png_dir /scratch/users/hshunt/convolutional_diffusion/tmp_audit_png

echo "done $(date)"
