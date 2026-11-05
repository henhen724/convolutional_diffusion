"""Can a grammar-side feature name the residual that reveal count leaves?

``rhm_k_locality.py`` bounded where the per-site window choice comes from:
position explains nothing (R^2 < 0.001), the local revealed-neighbour count
explains at most 0.47 and subsumes time.  That leaves >= 53% turning on *which*
tokens were revealed rather than how many, and "content-dependent" is a label,
not an account.

This asks for an account.  The obvious candidate is how much *grammatical*
ambiguity a site still has.  At L = 2 the leaf-producing rules are
``rhm.rules[-1]``, shape ``(v, m, s, s)``: ``v * m`` distinct s x s words, one
of which generated the site's own level-1 block.  Counting how many of them
still agree with the revealed sites of that block gives ``n_words``, between 1
(the block is pinned) and ``v * m`` (nothing is known).  Unlike the reveal
count it distinguishes *which* tokens were seen: two blocks with four revealed
sites each can leave four words standing or one.

Registered before the run: ``n_words`` adds at least 0.05 R^2 over the reveal
count alone, and the gain grows with v, because a bigger vocabulary means more
words to tell apart and so more resolution in the feature.  If the gain is
near zero the residual is not grammatical ambiguity at the block scale and the
next place to look is across block boundaries.

Two controls keep the comparison honest.  ``nrev_block`` is the reveal count
*within the same block* -- without it, any gain from ``n_words`` could just be
the block's own fill level re-entering under a grammatical name, since the K x K
count already measured is a different footprint.  And every model is scored
against the same K x K count baseline recomputed here, not against the number
in the other script, so the two are directly subtractable.
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
from src.utils.discrete_score import (ScheduledDiscreteScoreMachine,
                                      linear_mask_schedule)
from src.utils.rhm_sampling import AdaptiveWindowDiscreteScoreMachine
from scripts.rhm_k_locality import r2


class ContentRecorder(AdaptiveWindowDiscreteScoreMachine):
    """Records the chosen k with both count features and the grammar feature."""

    def __init__(self, *a, side: int, words: torch.Tensor, s: int, **kw):
        # mask_token is deliberately not passed through: the parent forwards the
        # raw argument to its sub-machines, so leaving it None keeps this
        # bit-for-bit the machine rhm_k_locality.py measured, and the R^2 here
        # subtracts from the R^2 there.
        super().__init__(*a, **kw)
        self.side = side
        self.s = s
        self.words = words          # (W, s, s), the v*m leaf-producing words
        self.rec = []

    def _block_features(self, xt: torch.Tensor):
        """Per site: compatible level-1 words, and reveals inside its own block."""
        B = xt.shape[0]
        nb = self.side // self.s
        x = xt.reshape(B, nb, self.s, nb, self.s).permute(0, 1, 3, 2, 4)
        msk = (x == self.mask_token)
        # a word survives where every revealed site of the block agrees with it
        agree = msk.unsqueeze(3) | (x.unsqueeze(3) == self.words)
        nw = agree.all(-1).all(-1).sum(-1)              # (B, nb, nb)
        nrevb = (~msk).sum(-1).sum(-1)                  # (B, nb, nb)
        # back to per-site, every site of a block carrying its block's value
        rep = lambda t: (t.repeat_interleave(self.s, 1).repeat_interleave(self.s, 2))
        return rep(nw), rep(nrevb)

    def forward(self, t, xt, label=None, device=None):
        xt = xt.to(self.machines[0].device)
        masked = (xt == self.mask_token)
        out = super().forward(t, xt, label=label, device=device)
        K = self.kernel_sizes[-1]
        grid = xt.reshape(-1, self.side, self.side)
        rev = (~masked).float().reshape(-1, 1, self.side, self.side)
        cnt = torch.nn.functional.conv2d(
            rev, torch.ones(1, 1, K, K, device=rev.device), padding=K // 2)
        nw, nrevb = self._block_features(grid)
        self.rec.append(tuple(a.detach().cpu().numpy().copy() for a in (
            self._last_kmap, masked, cnt, nw, nrevb)))
        return out


def joint(*cols):
    """A single group label per row from several integer columns."""
    out = np.zeros(len(cols[0]), dtype=np.int64)
    for c in cols:
        c = np.asarray(c, dtype=np.int64)
        out = out * (int(c.max()) + 1) + c
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--v', type=int, default=2)
    ap.add_argument('--s', type=int, default=3)
    ap.add_argument('--L', type=int, default=2)
    ap.add_argument('--m', type=int, default=2)
    ap.add_argument('--rule_seeds', type=int, default=3)
    ap.add_argument('--P', type=int, nargs='+', default=[300, 1000])
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
        # the last level is the one that writes leaf tokens, so its rules are
        # the words a revealed leaf block can still be an instance of
        words = torch.as_tensor(rhm.rules[-1].reshape(-1, args.s, args.s),
                                device=device)
        for P in args.P:
            train, _ = rhm.sample(P, seed=seed)
            mach = ContentRecorder(train, vocab_size=rhm.v,
                                   kernel_sizes=args.k_adapt, h_min=0.0,
                                   h_mode='match', n_min=1.0, alpha_smooth=1e-3,
                                   device=device, block_align=1,
                                   pad_mode='outside', side=side, words=words,
                                   s=args.s)
            sampler = ScheduledDiscreteScoreMachine(
                mach, vocab_size=rhm.v, grid_size=side,
                default_time_steps=side ** 2, mask_schedule=linear_mask_schedule,
                mask_token=rhm.v, reveal_one=True)
            gen = torch.Generator(device=device).manual_seed(seed + 9973)
            sampler.sample(n_samples=args.n_gen, device=device, generator=gen)

            ks, steps, nrev, nw, nrevb = [], [], [], [], []
            for i, (kmap, msk, cnt, w, rb) in enumerate(mach.rec):
                flat = lambda a: a.reshape(a.shape[0], -1)
                sel = np.nonzero(flat(msk))
                ks.append(flat(kmap)[sel])
                nrev.append(flat(cnt)[sel].astype(int))
                nw.append(flat(w)[sel].astype(int))
                nrevb.append(flat(rb)[sel].astype(int))
                steps.append(np.full(sel[1].shape, i))
            ks, steps, nrev, nw, nrevb = (np.concatenate(a) for a in
                                          (ks, steps, nrev, nw, nrevb))

            row = dict(v=args.v, s=args.s, L=args.L, m=args.m, rule_seed=rs, P=P,
                       n_gen=args.n_gen, n_obs=int(ks.size),
                       n_words_max=int(args.v * args.m),
                       k_mean=float(ks.mean()), k_sd=float(ks.std()),
                       nw_mean=float(nw.mean()),
                       # the baseline, recomputed here so the deltas subtract
                       r2_nrev=r2(ks, nrev),
                       r2_nw=r2(ks, nw),
                       r2_nrevb=r2(ks, nrevb),
                       r2_nrev_nw=r2(ks, joint(nrev, nw)),
                       r2_nrev_nrevb=r2(ks, joint(nrev, nrevb)),
                       r2_nrev_nrevb_nw=r2(ks, joint(nrev, nrevb, nw)),
                       r2_step_nrev_nw=r2(ks, joint(steps, nrev, nw)),
                       k_by_nw=[float(ks[nw == q].mean()) if (nw == q).any()
                                else float('nan')
                                for q in range(1, args.v * args.m + 1)])
            rows.append(row)
            gain = row['r2_nrev_nw'] - row['r2_nrev']
            ctrl = row['r2_nrev_nrevb'] - row['r2_nrev']
            print('v%d m%d rs%d P=%-5d k %.3f+-%.3f nw %.2f/%d | nrev %.3f'
                  ' +nw %.3f (gain %+.3f) | control +nrevb %+.3f -> all %.3f'
                  % (args.v, args.m, rs, P, row['k_mean'], row['k_sd'],
                     row['nw_mean'], row['n_words_max'], row['r2_nrev'],
                     row['r2_nrev_nw'], gain, ctrl, row['r2_nrev_nrevb_nw']))
            print('    mean k by n_words 1..%d: %s' % (
                args.v * args.m, ' '.join('%.2f' % x for x in row['k_by_nw'])))
            sys.stdout.flush()

            o = ROOT / args.out
            o.parent.mkdir(parents=True, exist_ok=True)
            json.dump(rows, open(o, 'w'))
    print('wrote %s' % (ROOT / args.out))


if __name__ == '__main__':
    main()
