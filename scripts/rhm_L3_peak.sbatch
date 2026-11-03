#!/bin/bash
#SBATCH --job-name=rhm_L3pk
#SBATCH --partition=sganguli
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --output=logs/rhm_L3peak_%j.out

# The measurement the deliverable is: where does success peak at L=3, and is
# that peak at P* = Theta(|R_L|)?
#
# 50930718 established that k=7 (the U=1 window at L=3) does make grammaticality
# rise with P -- 0.008 -> 0.234 for v3m2 rs1 over P = 100 -> 1e5 -- which no
# k<=5 run could show. But it stopped at P = 1e5 with success still climbing, so
# it bounds P* from below and does not locate it.
#
# What makes the peak locatable now is that |R_L| is known *exactly* per rule
# seed (nk_exact_L3.json), and the draws differ by an order of magnitude because
# rules are drawn with replacement and collide:
#
#     v=2 m=2:  rs0 4194304   rs1 4194304   rs2 2057216
#     v=3 m=2:  rs0 6291456   rs1  301440   rs2 6291456
#
# So the ladders below are chosen to span P/|R_L| from ~2e-3 to ~2 in every
# case, and rs1 of v3m2 reaches 10. If success peaks at the same P/|R_L| across
# seeds whose |R_L| differ by 20x, the law is P* = Theta(|R_L|), and since
# |R_L| = v m^((n-1)/(s^2-1)) that is exp(Theta(n)) -- the exponential, this
# time from the cost leg (coupon-collecting the N_k distinct windows) rather
# than from the retracted ceiling argument.
#
# The competing hypothesis this also tests: that a *partial* table is more
# constrained than the full grammar, so success peaks well below |R_L| and the
# machine never needs that much data. The L=2 peaks sat at P*/|R_L| = 0.06-0.77,
# which is O(1) but a wide O(1); L=3 says whether that band is a law or an
# artifact of L=2 being small enough to exhaust.
#
# Cost: the table build is what scales with P, and three fixes landed in 55ae0d1
# make P=4e6 fit (chunk before the unfold, pack rows to int64 words, bytes keys
# for novelty). The largest table here is N_7 = 1.8e7 at v3m2 rs0: one-hot 14 GB
# plus the fp16 matching copy 7 GB, which needs the 80 GB H100, not a 40 GB card.
# n_gen is 1024 rather than 128 because the quantity being located is a peak in
# success of order 0.05-0.2, and 128 samples resolves that to +-0.03 at best.

set -euo pipefail
cd "$HOME/convolutional_diffusion"

ml purge
ml devel math python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312 py-numpy/1.26.3_py312
source "$HOME/diffusion_venv_py312/bin/activate"

OUT=results/rhm_sampling/2026-11-03
mkdir -p "$OUT" logs
git rev-parse --short HEAD > "$OUT/provenance_L3peak.txt"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

COMMON="--L 3 --s 2 --n_gen 1024 --gen_batch 512 --nsteps 64 --block_align 1 \
        --pad_mode outside --max_patches 25000000"

# Stage 1, decisive: k=7 at the U=1 scale, ladders crossing P/|R_L| ~ 1.
python scripts/rhm_sampling_complexity.py --v 2 --m 2 --k 7 --rule_seeds 3 $COMMON \
    --P 10000 30000 100000 300000 1000000 2000000 4000000 \
    --out "$OUT/L3peak_v2_m2_k7.json"

python scripts/rhm_sampling_complexity.py --v 3 --m 2 --k 7 --rule_seeds 3 $COMMON \
    --P 10000 30000 100000 300000 1000000 3000000 \
    --out "$OUT/L3peak_v3_m2_k7.json"

# Stage 2, control: k=5 reaches only j*=1 and cannot resolve level-2 structure,
# so if the peak is about the window rather than the data, k=5 stays at
# grammaticality ~0 over the same ladder no matter how large P gets.
python scripts/rhm_sampling_complexity.py --v 2 --m 2 --k 5 --rule_seeds 2 $COMMON \
    --P 10000 100000 1000000 4000000 \
    --out "$OUT/L3peak_v2_m2_k5.json"

echo "done $(date)"
