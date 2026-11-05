"""Is the adaptive window's win per-site adaptivity, or just a k(t) schedule?

The s=3 fair test found the "largest window that matched any training patch"
criterion beats the best fixed k by a median 50x, and that no floor on top of it
helps -- neither the entropy floor (``h_min``, 8/8 cells prefer 0) nor the count
floor (``n_min``, monotone down).  So the criterion's whole contribution is the
*choice of k*, and that choice does two separable things:

1. it shrinks the window as the grid fills, a schedule k(t);
2. it picks different k at different sites of the *same* grid.

hshunt's question is about (1) -- "how are you setting the patch size over time?
It should be the maximum length scale which stays above the minimum posterior
entropy" -- so the thing worth knowing is whether (2) earns anything on top.  If
a deterministic k(t) reproduces the adaptive result, the answer to that question
is a curve, the support test is only the device that discovers it, and the whole
machine simplifies.  If it does not, the schedule is a summary statistic of a
rule that is genuinely local.

Three arms, identical grammar, training set, sampler and seed:

``adaptive``   the criterion: largest k with support, no cap.
``cap``        the criterion restricted to the schedule it itself induced.
``sched``      the bare schedule: the capped k at every masked site, support
               test off, so an unmatched site falls back on smoothing rather
               than on a shorter window.
``fixed``      the best fixed k at this cell, for grounding.

The schedule is not chosen by hand.  The adaptive arm records (masked fraction,
mean chosen k) per call; that trace is rounded to the available kernel sizes and
collapsed into breakpoints, which is the fairest possible control -- the same
k(t), applied uniformly instead of per site.  A hand-picked schedule could
always be made to lose.

Registered prediction, before the run: ``cap`` nearly matches ``adaptive``
(the cap only removes kernels the criterion was already declining to use), and
``sched`` falls well short of both, because the sites that cannot match the
capped window are exactly the ones that need a shorter one.  The null that
matters is ``sched ~ adaptive``: that would say per-site adaptivity is worth
nothing and the answer is a curve.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.random_hierarchy import RandomHierarchyModel2D
from src.utils.rhm_grammar import evaluate_samples
from src.utils.discrete_score import (ScheduledDiscreteScoreMachine,
                                      linear_mask_schedule)
from src.utils.rhm_sampling import AdaptiveWindowDiscreteScoreMachine


def schedule_from_trace(trace, kernel_sizes):
    """Round a measured (masked fraction, mean k) trace to a step schedule.

    The trace arrives in call order, so the masked fraction decreases.  Each
    entry is ``(lower edge, k)``: the k holds while the masked fraction is at or
    above that edge, which is what ``cap_at`` reads.  Recording the f at which a
    k *starts* instead would cap the run at the second k from the second call
    onwards -- the first interval would be the single point f = 1.0 -- so the
    control would be tighter than the trace it is meant to reproduce.  The last
    k runs to 0 so the schedule is total.
    """
    ks = np.asarray(sorted(kernel_sizes), dtype=float)
    seq = []
    for f, kbar in trace:
        k = int(ks[int(np.argmin(np.abs(ks - kbar)))])
        if not seq or seq[-1][1] != k:
            seq.append((float(f), k))
    out = [(seq[i + 1][0], seq[i][1]) for i in range(len(seq) - 1)]
    if seq:
        out.append((0.0, seq[-1][1]))
    return out


def run_arm(train, rhm, ksz, n_gen, seed, device, cap=None, require_support=True):
    mach = AdaptiveWindowDiscreteScoreMachine(
        train, vocab_size=rhm.v, kernel_sizes=ksz, h_min=0.0, h_mode='match',
        n_min=1.0, k_cap_schedule=cap, require_support=require_support,
        alpha_smooth=1e-3, device=device, block_align=1, pad_mode='outside')
    sampler = ScheduledDiscreteScoreMachine(
        mach, vocab_size=rhm.v, grid_size=rhm.grid_size,
        default_time_steps=rhm.grid_size ** 2, mask_schedule=linear_mask_schedule,
        mask_token=rhm.v, reveal_one=True)
    gen = torch.Generator(device=device).manual_seed(seed + 9973)
    grids = sampler.sample(n_samples=n_gen, device=device, generator=gen).cpu().numpy()
    res = evaluate_samples(rhm, train, grids)
    kbar = float(np.mean([k for _, k in mach.trace])) if mach.trace else float('nan')
    return res, kbar, list(mach.trace)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--v', type=int, default=2)
    ap.add_argument('--s', type=int, default=3)
    ap.add_argument('--L', type=int, default=2)
    ap.add_argument('--m', type=int, default=2)
    ap.add_argument('--rule_seeds', type=int, default=3)
    ap.add_argument('--P', type=int, nargs='+', default=[300])
    ap.add_argument('--k_adapt', type=int, nargs='+', default=[5, 7, 9])
    ap.add_argument('--k_fixed', type=int, default=7)
    ap.add_argument('--n_gen', type=int, default=4096)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows = []
    for rs in range(args.rule_seeds):
        rhm = RandomHierarchyModel2D(v=args.v, s=args.s, L=args.L, m=args.m,
                                     seed=1000 + rs)
        seed = 1234 + 7 * rs
        for P in args.P:
            train, _ = rhm.sample(P, seed=seed)
            common = dict(train=train, rhm=rhm, n_gen=args.n_gen, seed=seed,
                          device=device)

            ad, ad_k, trace = run_arm(ksz=args.k_adapt, **common)
            sched = schedule_from_trace(trace, args.k_adapt)
            cp, cp_k, _ = run_arm(ksz=args.k_adapt, cap=sched, **common)
            sc, sc_k, _ = run_arm(ksz=args.k_adapt, cap=sched,
                                  require_support=False, **common)
            fx, fx_k, _ = run_arm(ksz=[args.k_fixed], **common)

            row = dict(v=args.v, s=args.s, L=args.L, m=args.m, rule_seed=rs, P=P,
                       k_adapt=args.k_adapt, k_fixed=args.k_fixed,
                       schedule=sched, n_gen=args.n_gen,
                       arms={t: dict(success=float(r['success']),
                                     grammatical=float(r['grammatical']),
                                     novel=float(r['novel']),
                                     level_parse_fraction=r['level_parse_fraction'],
                                     mean_k=kb)
                             for t, (r, kb) in dict(adaptive=(ad, ad_k),
                                                    cap=(cp, cp_k),
                                                    sched=(sc, sc_k),
                                                    fixed=(fx, fx_k)).items()})
            rows.append(row)
            print('v%d m%d rs%d P=%-6d sched=%s' % (args.v, args.m, rs, P, sched))
            for t, (r, kb) in dict(adaptive=(ad, ad_k), cap=(cp, cp_k),
                                   sched=(sc, sc_k), fixed=(fx, fx_k)).items():
                print('    %-9s succ %.4f  gram %.4f  mean k %.3f'
                      % (t, r['success'], r['grammatical'], kb))
            sys.stdout.flush()

            o = ROOT / args.out
            o.parent.mkdir(parents=True, exist_ok=True)
            json.dump(rows, open(o, 'w'))
    print('wrote %s' % (ROOT / args.out))


if __name__ == '__main__':
    main()
