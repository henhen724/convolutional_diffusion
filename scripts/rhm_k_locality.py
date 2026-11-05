"""What drives the per-site window choice: position, time, or the data?

``rhm_k_schedule.py`` showed the criterion's win is per-site -- the same k(t)
curve applied uniformly, at matched mean k, loses 4.7x.  That says the choice
varies across sites of one grid at one time, but not *why*, and there are three
very different answers:

``step``   the masked fraction alone, i.e. the schedule.  Already refuted as a
           complete answer, and measured here as a baseline.
``phase``  the site's offset ``(i mod s, j mod s)`` within its level-1 block.
           A k x k window centred at a site sits across the grammar's blocks in
           a way fixed by that offset, so this is hshunt's straddling question
           in regression form: pixels that see parts of two different rules.
``both``   phase and step jointly, the best *precomputable* rule -- a lookup
           table in (position, time) with no reference to the partial grid.

Whatever ``both`` leaves unexplained is the genuinely data-dependent part: the
machine looking at what has actually been revealed so far.  If ``both`` has R^2
near 1 the criterion is a table and the support test is only a way of building
it; if it does not, the choice has to be made online.

Variance is measured over masked sites only, pooled across calls and samples,
because an unmasked site's k is never used.  R^2 is the usual
``1 - SS_resid / SS_total`` with each model's prediction the group mean, so the
three models are nested and directly comparable.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.random_hierarchy import RandomHierarchyModel2D
from src.utils.discrete_score import (ScheduledDiscreteScoreMachine,
                                      linear_mask_schedule)
from src.utils.rhm_sampling import AdaptiveWindowDiscreteScoreMachine


class KRecorder(AdaptiveWindowDiscreteScoreMachine):
    """Records chosen k, the mask, and the local revealed count at every call.

    The revealed count is how many of the ``K x K`` sites around a site are
    already fixed, ``K`` the largest kernel.  It is the mechanistic candidate
    for the online signal: a window with more of its neighbourhood pinned has
    more to match against and so fails support sooner, which would make the
    rule "shrink the window where the surroundings are already filled in".
    Unlike the block offset it varies sample to sample at fixed position and
    time, which is what a data-dependent signal has to do.
    """

    def __init__(self, *a, side: int, **kw):
        super().__init__(*a, **kw)
        self.side = side
        self.rec = []

    def forward(self, t, xt, label=None, device=None):
        masked = (xt.to(self.machines[0].device) == self.mask_token)
        out = super().forward(t, xt, label=label, device=device)
        K = self.kernel_sizes[-1]
        rev = (~masked).float().reshape(-1, 1, self.side, self.side)
        cnt = torch.nn.functional.conv2d(
            rev, torch.ones(1, 1, K, K, device=rev.device), padding=K // 2)
        # the parent stores the per-site choice it just made, so the accept
        # test lives in one place and this cannot drift from it
        self.rec.append((self._last_kmap.detach().cpu().numpy().copy(),
                         masked.detach().cpu().numpy().copy(),
                         cnt.detach().cpu().numpy().copy()))
        return out


def r2(y, groups):
    """1 - SS_resid/SS_total with each group predicted by its own mean."""
    y = np.asarray(y, dtype=float)
    tot = float(((y - y.mean()) ** 2).sum())
    if tot <= 0:
        return float('nan')
    resid = 0.0
    for g in np.unique(groups):
        sel = groups == g
        resid += float(((y[sel] - y[sel].mean()) ** 2).sum())
    return 1.0 - resid / tot


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
    ap.add_argument('--n_gen', type=int, default=512)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows = []
    for rs in range(args.rule_seeds):
        rhm = RandomHierarchyModel2D(v=args.v, s=args.s, L=args.L, m=args.m,
                                     seed=1000 + rs)
        seed = 1234 + 7 * rs
        side = rhm.grid_size
        for P in args.P:
            train, _ = rhm.sample(P, seed=seed)
            mach = KRecorder(train, vocab_size=rhm.v, kernel_sizes=args.k_adapt,
                             h_min=0.0, h_mode='match', n_min=1.0,
                             alpha_smooth=1e-3, device=device,
                             block_align=1, pad_mode='outside', side=side)
            sampler = ScheduledDiscreteScoreMachine(
                mach, vocab_size=rhm.v, grid_size=side,
                default_time_steps=side ** 2, mask_schedule=linear_mask_schedule,
                mask_token=rhm.v, reveal_one=True)
            gen = torch.Generator(device=device).manual_seed(seed + 9973)
            sampler.sample(n_samples=args.n_gen, device=device, generator=gen)

            ks, steps, phases, nrev = [], [], [], []
            for i, (kmap, msk, cnt) in enumerate(mach.rec):
                kf = kmap.reshape(kmap.shape[0], -1)
                mf = msk.reshape(msk.shape[0], -1)
                cf = cnt.reshape(cnt.shape[0], -1)
                sel = np.nonzero(mf)
                site = sel[1]
                ks.append(kf[sel])
                nrev.append(cf[sel].astype(int))
                steps.append(np.full(site.shape, i))
                phases.append((site // side) % args.s * args.s
                              + (site % side) % args.s)
            ks = np.concatenate(ks)
            steps = np.concatenate(steps)
            phases = np.concatenate(phases)
            nrev = np.concatenate(nrev)
            both = steps * (args.s ** 2) + phases
            # how much the local count adds once the schedule is already known
            step_rev = steps * (nrev.max() + 1) + nrev

            row = dict(v=args.v, s=args.s, L=args.L, m=args.m, rule_seed=rs, P=P,
                       n_gen=args.n_gen, n_obs=int(ks.size),
                       k_mean=float(ks.mean()), k_sd=float(ks.std()),
                       r2_step=r2(ks, steps), r2_phase=r2(ks, phases),
                       r2_both=r2(ks, both), r2_nrev=r2(ks, nrev),
                       r2_step_nrev=r2(ks, step_rev),
                       k_by_phase=[float(ks[phases == p].mean())
                                   for p in range(args.s ** 2)])
            rows.append(row)
            print('v%d m%d rs%d P=%-5d k %.3f+-%.3f | R2 step %.3f phase %.3f both %.3f'
                  ' | nrev %.3f  step+nrev %.3f'
                  % (args.v, args.m, rs, P, row['k_mean'], row['k_sd'],
                     row['r2_step'], row['r2_phase'], row['r2_both'],
                     row['r2_nrev'], row['r2_step_nrev']))
            print('    mean k by phase (i,j mod %d): %s' % (
                args.s, ' '.join('%.2f' % x for x in row['k_by_phase'])))
            sys.stdout.flush()

            o = ROOT / args.out
            o.parent.mkdir(parents=True, exist_ok=True)
            json.dump(rows, open(o, 'w'))
    print('wrote %s' % (ROOT / args.out))


if __name__ == '__main__':
    main()
