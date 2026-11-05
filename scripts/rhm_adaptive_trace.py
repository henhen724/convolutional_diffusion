"""Reveal-by-reveal traces of the adaptive-window machine, for the deck.

Every results deck in this project shows example trajectories rather than only
aggregates, and for this one the trajectory *is* the result: the claim under
test is about how the window size moves along a generation, so a reader has to
be able to watch it move.

The sampler is not reimplemented here.  ``ScheduledDiscreteScoreMachine`` drives
the real machine, and a thin subclass records what the adaptive forward decided
at each call -- the grid it was handed, the per-site chosen ``k``, and the
per-site posterior entropy.  Which site the sampler then revealed is recovered
by diffing consecutive grids, so the trace is read off the actual run instead of
a parallel copy of its logic that could drift from it.
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
from src.utils.rhm_grammar import evaluate_samples, is_grammatical, is_novel
from src.utils.discrete_score import (ScheduledDiscreteScoreMachine,
                                      linear_mask_schedule)
from src.utils.rhm_sampling import AdaptiveWindowDiscreteScoreMachine


class TracingAdaptive(AdaptiveWindowDiscreteScoreMachine):
    """Records (grid in, chosen k per site, posterior entropy per site) per call."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.steps = []

    def forward(self, t, xt, label=None, device=None):
        masked = (xt.to(self.machines[0].device) == self.mask_token)
        chosen = None
        kmap = None
        for k, mach in zip(self.kernel_sizes, self.machines):
            post, n_match, h_match = mach(t, xt, return_match=True)
            if chosen is None:
                chosen = post
                kmap = torch.full_like(n_match, float(k))
                continue
            h = (-(post.clamp_min(1e-12) * post.clamp_min(1e-12).log()).sum(1)
                 if self.h_mode == 'token' else h_match)
            enough = (n_match > 0) if self.n_min <= 1 else (n_match >= self.n_min)
            ok = masked & enough & (h >= self.h_min)
            chosen = torch.where(ok[:, None], post, chosen)
            kmap = torch.where(ok, torch.full_like(kmap, float(k)), kmap)
        p = chosen.clamp_min(1e-12)
        ent = -(p * p.log()).sum(1)
        self.steps.append((xt.detach().cpu().numpy().copy(),
                           kmap.detach().cpu().numpy().copy(),
                           ent.detach().cpu().numpy().copy()))
        if masked.any():
            self.trace.append((float(masked.float().mean()),
                               float(kmap[masked].mean())))
        return chosen


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--v', type=int, default=2)
    ap.add_argument('--s', type=int, default=2)
    ap.add_argument('--L', type=int, default=2)
    ap.add_argument('--m', type=int, default=3)
    ap.add_argument('--rule_seed', type=int, default=0)
    ap.add_argument('--P', type=int, default=30)
    ap.add_argument('--k_adapt', type=int, nargs='+', default=[3, 5])
    ap.add_argument('--h_min', type=float, default=0.1)
    ap.add_argument('--h_mode', default='match')
    ap.add_argument('--n_min', type=float, default=1.0,
                    help='minimum matching training occurrences, the count rival to --h_min')
    ap.add_argument('--n_gen', type=int, default=64)
    ap.add_argument('--k_fixed', type=int, default=None,
                    help='make the control a fixed-window machine at this k instead of the '
                         'adaptive machine at h_min=0. Needed for the s=3 fair test, where the '
                         'best floor turned out to be 0, so floor-vs-control would compare a '
                         'run with itself; the contrast that decides it is adaptive vs the best '
                         'fixed k. Passing one kernel size to the adaptive machine *is* the '
                         'fixed machine -- the override loop never fires -- so this stays one '
                         'code path rather than two.')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rhm = RandomHierarchyModel2D(v=args.v, s=args.s, L=args.L, m=args.m,
                                 seed=1000 + args.rule_seed)
    seed = 1234 + 7 * args.rule_seed
    train, _ = rhm.sample(args.P, seed=seed)

    out = {}
    arms = (('floor', args.h_min, args.n_min, args.k_adapt),
            ('control', 0.0, 1.0,
             [args.k_fixed] if args.k_fixed is not None else args.k_adapt))
    for tag, h_min, n_min, ksz in arms:
        mach = TracingAdaptive(train, vocab_size=rhm.v, kernel_sizes=ksz,
                               h_min=h_min, h_mode=args.h_mode, n_min=n_min,
                               alpha_smooth=1e-3,
                               device=device, block_align=1, pad_mode='outside')
        sampler = ScheduledDiscreteScoreMachine(
            mach, vocab_size=rhm.v, grid_size=rhm.grid_size,
            default_time_steps=rhm.grid_size ** 2, mask_schedule=linear_mask_schedule,
            mask_token=rhm.v, reveal_one=True)
        gen = torch.Generator(device=device).manual_seed(seed + 9973)
        grids = sampler.sample(n_samples=args.n_gen, device=device,
                               generator=gen).cpu().numpy()
        res = evaluate_samples(rhm, train, grids)
        # evaluate_samples returns only the means; the per-sample masks are what
        # a trace page needs, so take them from the same two functions it calls
        gram = np.asarray(is_grammatical(rhm, grids), dtype=bool)
        novel = np.asarray(is_novel(train, grids), dtype=bool)

        # the revealed site at step i is the one that differs between the grid
        # handed to call i and the grid handed to call i+1; the last call has no
        # successor, so the final grid closes the sequence
        xs = [s[0] for s in mach.steps] + [grids]
        traj = []
        for b in range(args.n_gen):
            seq = []
            for i in range(len(mach.steps)):
                a, c = xs[i][b].reshape(-1), xs[i + 1][b].reshape(-1)
                d = np.nonzero(a != c)[0]
                if len(d) != 1:
                    continue
                j = int(d[0])
                seq.append(dict(step=i, site=j,
                                k=float(mach.steps[i][1][b].reshape(-1)[j]),
                                H=float(mach.steps[i][2][b].reshape(-1)[j]),
                                token=int(c[j])))
            traj.append(dict(sample=b, reveals=seq,
                             grid=xs[-1][b].astype(int).tolist(),
                             novel=bool(novel[b]), grammatical=bool(gram[b])))
        out[tag] = dict(h_min=h_min, n_min=n_min, kernel_sizes=list(ksz),
                        success=float(res['success']),
                        grammatical=float(res['grammatical']),
                        novel=float(novel.mean()), traj=traj)
        print('%-8s k=%-12s h_min=%.1f n>=%g  succ %.4f  gram %.4f  novel %.4f  mean k %.3f'
              % (tag, ksz, h_min, n_min, res['success'], res['grammatical'], novel.mean(),
                 float(np.mean([kb for _, kb in mach.trace]))))

    out['config'] = dict(v=args.v, s=args.s, L=args.L, m=args.m,
                         rule_seed=args.rule_seed, P=args.P, seed=seed,
                         k_adapt=args.k_adapt, k_fixed=args.k_fixed, h_mode=args.h_mode,
                         n_min=args.n_min,
                         n_sites=rhm.grid_size ** 2,
                         train_grids=train.astype(int).tolist())
    o = ROOT / args.out
    o.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(o, 'w'))
    print('wrote %s' % o)


if __name__ == '__main__':
    main()
