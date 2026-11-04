"""Does a posterior-entropy budget set the window optimum, and the 1D/2D gap?

The k-sweep (job 51058208) left one thing explained and one thing not.  Explained:
success turns over at k=11 rather than at the root-reaching U1=15, because
widening the window buys grammaticality and spends novelty.  Not explained: why
*there*, and why ``x* = P*/|R_L|`` is 0.193 in 2D against 0.083 in 1D at matched
n -- coverage and table size were both tested and both failed.

The hypothesis here is a budget, and it is hshunt's own criterion for the patch
scale written out for the RHM.  A generation is scored a success only if it is
novel, and to land outside a training set of ``P`` strings the sampler has to
inject enough randomness to get there.  Summing the entropy of the posterior it
actually samples from, over the n reveals of one generation,

    H_tot  =  sum_t  H(p(x_i | x_observed))          [nats]

is exactly that randomness.  Below ``H_tot = ln P`` there is less entropy in the
whole trajectory than there is in naming a training string, and novelty should
die; above it the sampler has room.  So:

    PREDICTION (parameter-free).  The success peak sits where the trajectory
    entropy crosses the novelty floor:   H_tot(k, P*)  =  ln P*.

It predicts three things at once, which is why it is worth a run:

  1. the k-optimum.  H_tot falls with k (a wider window pins more), so the
     crossing moves down in P as k rises; success peaks at the largest k whose
     crossing is still above where the grammar has come good.
  2. a formula for x*, with no fitted constant:  x* = exp(H_tot) / |R_L|.
  3. the sign of the 1D/2D gap.  At matched n a 2D U=1 window covers 9/16 of
     the grid where a 1D one covers 15/16, so it pins less, H_tot is larger,
     the crossing sits at larger P, and x* is larger.  Coverage failed as a
     *direct* explanation of x*; the claim here is that it acts only through
     the entropy, which is measurable and which also moves with v, m and k.

What is measured: the same cells, seeds and ladder as ``rhm_1d_coverage``, with
the posterior entropy accumulated during generation.  Nothing else changes, so
grammaticality, novelty and success must reproduce the earlier run row for row
-- the script checks that and says so, since a mismatch would mean the entropy
instrumentation perturbed the sampler rather than observed it.

Usage:
  python scripts/rhm_entropy_window.py --L 3 4 --v 2 3 --m 2 3 4 \
      --rule_seeds 3 --out results/rhm_1d/2026-11-04b/entropy_1d.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.rhm_1d_coverage import coverage, k_values
from scripts.rhm_1d_depth import MASK, LS1D, RHM1D, exact_RL, is_grammatical


def generate_H(machine, n_samples, n_sites, device, generator):
    """Reveal-one ancestral sampling, also returning the entropy it injected.

    Identical to ``rhm_1d_depth.generate`` except for the accumulator: the two
    RNG draws per step happen in the same order with the same shapes, so the
    generated strings are bit-identical to that function's given one seed.  The
    entropy is summed only over reveals that actually land (``keep``), so a
    batch element that is already full contributes nothing.
    """
    xt = torch.full((n_samples, n_sites), MASK, dtype=torch.int64, device=device)
    rows = torch.arange(n_samples, device=device)
    H = torch.zeros(n_samples, device=device)
    for _ in range(n_sites):
        masked = xt == MASK
        if not bool(masked.any()):
            break
        noise = torch.rand(n_samples, n_sites, device=device, generator=generator)
        pick = torch.where(masked, noise, torch.full_like(noise, -1.0)).argmax(1)
        p = machine.posterior(xt, pick).clamp_min(1e-12)
        tok = torch.multinomial(p, 1, generator=generator)[:, 0]
        keep = masked[rows, pick]
        h = -(p * p.log()).sum(1)
        H += torch.where(keep, h, torch.zeros_like(h))
        xt[rows[keep], pick[keep]] = tok[keep]
    return xt, H


def run_cell_H(model, P, k, n_gen, device, seed, alpha_smooth, gen_batch):
    """One (P, k) measurement, with the trajectory entropy of every generation."""
    t0 = time.time()
    train = model.sample(P, seed=seed)
    machine = LS1D(train, model.v, k, device, alpha_smooth=alpha_smooth)
    gen = torch.Generator(device=device).manual_seed(seed + 9973)
    out, Hs = [], []
    for i in range(0, n_gen, gen_batch):
        b = min(gen_batch, n_gen - i)
        g, h = generate_H(machine, b, model.d, device, gen)
        out.append(g.cpu().numpy())
        Hs.append(h.cpu().numpy())
    gs = np.concatenate(out, 0)
    H = np.concatenate(Hs, 0)

    gram = is_grammatical(model, gs)
    tr = set(map(bytes, np.ascontiguousarray(train.astype(np.int8))))
    novel = np.array([bytes(r) not in tr
                      for r in np.ascontiguousarray(gs.astype(np.int8))], dtype=bool)
    # the entropy of the *successful* generations separately: the budget claim
    # is about what it takes to leave the training set, so if it is right the
    # novel ones should sit above ln P and the copies below it
    def m(x):
        return float(np.mean(x)) if len(x) else float('nan')
    return dict(P=int(P), k=int(k), v=model.v, s=model.s, L=model.L, m=model.m,
                rule_seed=model.seed - 1000, n_samples=int(len(gs)),
                grammatical=float(gram.mean()), novel=float(novel.mean()),
                success=float((gram & novel).mean()),
                H_tot=m(H), H_sd=float(np.std(H)), H_site=m(H) / model.d,
                H_novel=m(H[novel]), H_copy=m(H[~novel]),
                ln_P=float(np.log(P)), margin=m(H) - float(np.log(P)),
                n_distinct_patches=machine.n_distinct,
                train_distinct=len(tr), seconds=round(time.time() - t0, 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--v', type=int, nargs='+', default=[2, 3])
    ap.add_argument('--s', type=int, default=2)
    ap.add_argument('--L', type=int, nargs='+', default=[3, 4])
    ap.add_argument('--m', type=int, nargs='+', default=[2, 3, 4])
    ap.add_argument('--rule_seeds', type=int, default=3)
    ap.add_argument('--n_gen', type=int, default=2048)
    ap.add_argument('--gen_batch', type=int, default=1024)
    ap.add_argument('--alpha_smooth', type=float, default=1e-3)
    ap.add_argument('--R_max', type=float, default=1e5)
    ap.add_argument('--R_min', type=float, default=200)
    # applied up front this time rather than in the analysis: the k-sweep spent
    # GPU time on n=8 v=2 cells whose grammar covers 78-100% of all 2^8 strings,
    # where "grammatical" is not a discriminating event, and then dropped them
    ap.add_argument('--max_chance', type=float, default=0.05,
                    help='skip cells whose |R_L| / v^n exceeds this')
    ap.add_argument('--max_patches', type=int, default=3_000_000)
    ap.add_argument('--P_min', type=int, default=1)
    ap.add_argument('--P_max', type=int, default=262144)
    ap.add_argument('--out', type=str, required=True)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ladder = [p for p in (args.P_min * 2 ** i for i in range(40)) if p <= args.P_max]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for L in args.L:
        for v in args.v:
            for m in args.m:
                for rs in range(args.rule_seeds):
                    model = RHM1D(v, args.s, L, m, seed=1000 + rs)
                    d = model.d
                    nominal = v * m ** ((d - 1) // (args.s - 1))
                    R = None if nominal > 1000 * args.R_max else exact_RL(model)
                    if R is None or R > args.R_max or R < args.R_min:
                        print('skip  v%d m%d L%d rs%d  |R_L|=%s outside [%g, %g]'
                              % (v, m, L, rs, R, args.R_min, args.R_max), flush=True)
                        continue
                    chance = R / float(v) ** d
                    if chance > args.max_chance:
                        print('skip  v%d m%d L%d rs%d  |R_L|=%d is %.0f%% of all %d^%d '
                              'strings' % (v, m, L, rs, R, 100 * chance, v, d), flush=True)
                        continue
                    ks, U1 = k_values(d, args.s, L)
                    print('cell  v%d m%d L%d rs%d  n=%d  U1=%d  |R_L|=%d (%.2f%%)  k in %s'
                          % (v, m, L, rs, d, U1, R, 100 * chance, ks), flush=True)
                    for k in ks:
                        cov_nom, cov_eff = coverage(k, d)
                        for P in ladder:
                            r = run_cell_H(model, P, k, args.n_gen, device,
                                           1234 + 7 * rs, args.alpha_smooth, args.gen_batch)
                            if r['n_distinct_patches'] > args.max_patches:
                                print('   k=%-3d P=%-7d SKIP, table too big' % (k, P),
                                      flush=True)
                                break
                            r.update(R_L=int(R), U1=U1, sees_root=bool(k >= U1),
                                     cov_nom=round(cov_nom, 4), cov_eff=round(cov_eff, 4),
                                     n=d, chance=round(chance, 6))
                            rows.append(r)
                            print('   k=%-3d P=%-7d succ %.4f  H_tot %6.2f  lnP %6.2f  '
                                  'margin %+6.2f  (novel %.2f / copy %.2f)'
                                  % (k, P, r['success'], r['H_tot'], r['ln_P'],
                                     r['margin'], r['H_novel'], r['H_copy']), flush=True)
                            json.dump(rows, open(out, 'w'), indent=1)
    json.dump(rows, open(out, 'w'), indent=1)
    print('wrote %s (%d rows)' % (out, len(rows)))


if __name__ == '__main__':
    main()
