#!/bin/bash
#SBATCH --job-name=rhm_L3k7
#SBATCH --partition=sganguli
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/rhm_L3k7_%j.out

# The measurement the whole question turns on, and the one no run has made yet.
#
# A k x k window reaches k//2, so it resolves RHM levels up to
# j* = floor(log_s((k+1)/2)). At L=3 that means k=5 reaches only j*=1 and cannot
# see level-2 structure at all; the U=1 member of the family is k = 2 s^(L-1) - 1
# = 7. Every L=3 generation sweep so far ran k <= 5 -- 50903583 included -- which
# is why grammaticality there is 0.000 at every P and every entropy floor. That
# is a statement about the window, not about the sampling complexity.
#
# k=7 was previously excluded on cost: matching by conv2d against ~1e6 patches
# materialises a (b, NP, h, w) tensor and runs a kernel shape cuDNN handles
# badly. The GEMM path added 2026-11-03 does the identical arithmetic as a
# matmul over unfolded windows in fp16 (agreement counts are integers <= k*k, so
# fp16 inputs with fp32 accumulation are exact -- checked against the conv path
# to 0.00e+00 on CPU, and again on GPU in stage 0 below).
#
# Also note what the 2026-11-03 L=2 sweeps revealed: |R_L| = v m^5 is at most
# 31104 there, so by P ~ 1e4 the training set *is* the whole grammar and
# novel-and-grammatical is impossible by construction (succ -> 0 exactly, with
# gram + novel = 1.000). L=2 cannot answer the asymptotic question at large P.
# At L=3, |R_L| = v m^21 is 4.2e6 (v=2,m=2) to 2.1e10 (v=2,m=3), so the whole
# ladder below stays far from exhausting the grammar.

set -euo pipefail
cd "$HOME/convolutional_diffusion"

ml purge
ml devel math python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312 py-numpy/1.26.3_py312
source "$HOME/diffusion_venv_py312/bin/activate"

OUT=results/rhm_sampling/2026-11-03
mkdir -p "$OUT" logs
git rev-parse --short HEAD > "$OUT/provenance_L3k7.txt"

# Stage 0: the GEMM path must reproduce the conv path on the GPU, in fp16, at
# the exact (k, L) this job depends on. If it does not, nothing below is worth
# reading, so fail here.
python - <<'EOF'
import sys, torch
sys.path.insert(0, '.')
from src.utils.random_hierarchy import RandomHierarchyModel2D
from src.utils.rhm_sampling import PatchCountDiscreteScoreMachine
torch.manual_seed(0)
dev = 'cuda'
for v, m, k in [(2, 2, 7), (3, 2, 7), (2, 3, 7), (2, 2, 5)]:
    rhm = RandomHierarchyModel2D(v=v, s=2, L=3, m=m, seed=7)
    g, _ = rhm.sample(400, seed=1)
    kw = dict(vocab_size=v, kernel_size=k, block_align=1, pad_mode='outside',
              device=dev, alpha_smooth=1e-3)
    a = PatchCountDiscreteScoreMachine(g, use_gemm=False, **kw)
    b = PatchCountDiscreteScoreMachine(g, use_gemm=True, **kw)
    xt = torch.randint(0, v + 1, (8, 8, 8), device=dev)
    t = torch.zeros(8, device=dev)
    pa, na, ha = a(t, xt, return_match=True)
    pb, nb, hb = b(t, xt, return_match=True)
    dp, dn = (pa - pb).abs().max().item(), (na - nb).abs().max().item()
    print('v%d m%d k%d NP=%d  dpost=%.3e dn=%.3e' % (v, m, k, a.n_distinct, dp, dn), flush=True)
    assert dp < 1e-5 and dn < 1e-3, 'GEMM path disagrees with conv path'
print('stage 0 ok', flush=True)
EOF

# Stage 1: the ladder at the U=1 scale. nsteps=64 is one reveal per site on the
# 8x8 grid (reveal_one=True), the minimum a full trajectory needs.
LAD="100 300 1000 3000 10000 30000 100000"
for vm in "2 2" "3 2" "2 3"; do
    set -- $vm
    python scripts/rhm_sampling_complexity.py --v $1 --m $2 --L 3 \
        --k 3 5 7 --P $LAD --n_gen 128 --gen_batch 128 --nsteps 64 \
        --rule_seeds 2 --block_align 1 --pad_mode outside \
        --max_patches 2500000 --out "$OUT/L3k7_v$1_m$2.json"
done

echo "done $(date)"
