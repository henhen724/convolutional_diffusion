"""Does consistency *between* blocks inform the window choice, as hshunt asked?

``rhm_k_content.py`` named most of the residual the reveal count left: the
level-1 words still compatible with a site's **own** block (``n_words``) add a
median +0.107 R^2 over ``nrev`` across 18 cells.  But ``n_words`` is by
construction blind to everything outside the block, and hshunt's question was
about exactly what it cannot see:

    "When there are pixels that see part of two different grammar rules they
    may be able to check for consistency between blocks in a way the block
    aligned model cannot. [...] if I could tell consistency of a node two
    levels up by looking at the boundaries between the two nodes."

So this measures the two-levels-up constraint directly.  At L = 2 the root
productions are ``rhm.rules[0]``, ``v * m`` distinct s x s arrays of *level-1
symbols*, and the block grid is also s x s, so each production assigns one
level-1 symbol to each block.  Define

  compat[b][sigma] : block b can still be an instance of level-1 symbol sigma
                     (some word of sigma agrees with b's revealed sites);
  surviving        : the root productions p with compat[b][p[b]] for EVERY b;
  n_sym_local[b]   = #{sigma : compat[b][sigma]}        -- block b alone;
  n_sym_root[b]    = #{p[b] : p surviving}              -- block b after every
                     OTHER block has had its say through the root.

``n_sym_root <= n_sym_local`` always, and the gap is precisely the information
that crosses a block boundary.  A window that straddles two blocks is the only
thing that can see it; a block-aligned window cannot.

Registered before this run:

  1. ``n_sym_root`` adds >= 0.02 R^2 over the joint (nrev, nw, n_sym_local)
     baseline, *net of the permutation null* below.  Modest threshold because
     ``nw`` already carries the within-block part and this can only add the
     cross-block remainder.
  2. The substrate exists: ``n_sym_root < n_sym_local`` at more than 10% of
     site-observations.  This is the mechanism check, reported whether or not
     prediction 1 holds -- if cross-block consistency almost never tightens the
     constraint then a null in 1 means "nothing to see", not "the machine
     ignores it", and those are different findings.
  3. Control: ``n_sym_local`` on top of (nrev, nw) adds < 0.02 on its own, since
     it is a coarsening of ``nw`` (symbols instead of words).

Every model is scored against the same baseline recomputed here, so the deltas
subtract against ``rhm_k_content.py``'s.

Note on method: ``RandomHierarchyModel2D.belief_propagation`` gives the *exact*
posterior marginal at every level, which would turn this hard surviving-rule
count into a soft per-block entropy and connect directly to hshunt's entropy
criterion.  Deliberately not used here: the count is what "consistency between
blocks" means literally, it is cheap, and a null result is easier to read
without a second approximation-free-but-unfamiliar quantity in the way.  If the
cross-block gain is real, redoing it in BP entropies is the obvious refinement.
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
from scripts.rhm_k_locality import r2 as r2_ref
from scripts.rhm_k_content import joint


def r2(y, groups):
    """Same group-means R^2 as ``rhm_k_locality.r2``, but O(n) not O(n*groups).

    The reference implementation scans the whole array once per unique group.
    At 1.7M observations and a four-column joint that is the dominant cost, so
    this does it with bincount instead.  ``_check_r2`` below asserts the two
    agree before any cell is scored.
    """
    y = np.asarray(y, dtype=float)
    tot = float(((y - y.mean()) ** 2).sum())
    if tot <= 0:
        return float('nan')
    _, inv = np.unique(groups, return_inverse=True)
    n = np.bincount(inv)
    s = np.bincount(inv, weights=y)
    # SS_resid = sum(y^2) - sum_g (sum_g y)^2 / n_g
    resid = float((y ** 2).sum() - (s ** 2 / n).sum())
    return 1.0 - resid / tot


def _check_r2(seed=0, n=20000):
    """Guard: the fast r2 must reproduce the reference on random input."""
    rng = np.random.default_rng(seed)
    y = rng.integers(5, 10, n).astype(float)
    g = joint(rng.integers(0, 40, n), rng.integers(0, 6, n))
    a, b = r2(y, g), r2_ref(y, g)
    if not np.isclose(a, b, rtol=0, atol=1e-9):
        raise SystemExit('fast r2 disagrees with reference: %.12f vs %.12f'
                         % (a, b))


class CrossBlockRecorder(AdaptiveWindowDiscreteScoreMachine):
    """Records chosen k with the within-block and the across-block features."""

    def __init__(self, *a, side: int, words: torch.Tensor, prods: torch.Tensor,
                 s: int, v: int, **kw):
        # mask_token is deliberately not forwarded: the parent passes the raw
        # argument down to its sub-machines, so leaving it None keeps this the
        # same machine rhm_k_content.py and rhm_k_locality.py measured and the
        # R^2 here subtracts from theirs.
        super().__init__(*a, **kw)
        self.side = side
        self.s = s
        self.v = v
        self.words = words      # (v, m, s, s) leaf-producing words, by symbol
        self.prods = prods      # (v*m, s, s) root productions, in level-1 symbols
        self.rec = []

    def _blocks(self, xt: torch.Tensor):
        """(B, nb, nb, s, s) view of the grid, plus the mask in the same shape."""
        B = xt.shape[0]
        nb = self.side // self.s
        x = xt.reshape(B, nb, self.s, nb, self.s).permute(0, 1, 3, 2, 4)
        return x, (x == self.mask_token), nb

    def _features(self, xt: torch.Tensor):
        """Per site: n_words, n_rev_block, n_sym_local, n_sym_root."""
        x, msk, nb = self._blocks(xt)
        v, m = self.words.shape[0], self.words.shape[1]

        # (B, nb, nb, v, m): does word (sigma, j) agree with every revealed site?
        w = self.words.reshape(1, 1, 1, v, m, self.s, self.s)
        agree = (msk.unsqueeze(3).unsqueeze(4) |
                 (x.unsqueeze(3).unsqueeze(4) == w)).all(-1).all(-1)

        nw = agree.reshape(*agree.shape[:3], v * m).sum(-1)   # (B, nb, nb)
        nrevb = (~msk).sum(-1).sum(-1)                        # (B, nb, nb)
        compat = agree.any(-1)                                # (B, nb, nb, v)
        n_sym_local = compat.sum(-1)                          # (B, nb, nb)

        # A root production survives when every block admits the symbol it
        # assigns there.  nb == s at L = 2, which the caller asserts.
        B, P = x.shape[0], self.prods.shape[0]
        idx = self.prods.reshape(1, P, nb, nb, 1).expand(B, P, nb, nb, 1)
        got = torch.gather(compat.unsqueeze(1).expand(B, P, nb, nb, v), 4,
                           idx).squeeze(-1)                   # (B, P, nb, nb)
        surv = got.all(-1).all(-1)                            # (B, P)

        # distinct symbols still assignable at each block by a surviving rule
        oh = torch.nn.functional.one_hot(self.prods.long(), v).bool()  # (P,nb,nb,v)
        present = (oh.unsqueeze(0) & surv.reshape(B, P, 1, 1, 1)).any(1)
        n_sym_root = present.sum(-1)                          # (B, nb, nb)

        rep = lambda t: (t.repeat_interleave(self.s, 1).repeat_interleave(self.s, 2))
        return rep(nw), rep(nrevb), rep(n_sym_local), rep(n_sym_root)

    def forward(self, t, xt, label=None, device=None):
        xt = xt.to(self.machines[0].device)
        masked = (xt == self.mask_token)
        out = super().forward(t, xt, label=label, device=device)
        K = self.kernel_sizes[-1]
        grid = xt.reshape(-1, self.side, self.side)
        rev = (~masked).float().reshape(-1, 1, self.side, self.side)
        cnt = torch.nn.functional.conv2d(
            rev, torch.ones(1, 1, K, K, device=rev.device), padding=K // 2)
        nw, nrevb, nsl, nsr = self._features(grid)
        self.rec.append(tuple(a.detach().cpu().numpy().copy() for a in (
            self._last_kmap, masked, cnt, nw, nrevb, nsl, nsr)))
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

    if args.L != 2:
        raise SystemExit('the root-production feature is written for L = 2 only')
    _check_r2()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows = []
    for rs in range(args.rule_seeds):
        rhm = RandomHierarchyModel2D(v=args.v, s=args.s, L=args.L, m=args.m,
                                     seed=1000 + rs)
        seed = 1234 + 7 * rs
        side = rhm.grid_size
        nb = side // args.s
        assert nb == args.s, 'root production is s x s; needs nb == s (L = 2)'
        # rules[-1]: symbol -> s x s leaf tokens.  rules[0]: root symbol, choice
        # -> s x s level-1 symbols, one per block.
        words = torch.as_tensor(rhm.rules[-1], device=device)
        prods = torch.as_tensor(rhm.rules[0].reshape(-1, args.s, args.s),
                                device=device)
        for P in args.P:
            train, _ = rhm.sample(P, seed=seed)
            mach = CrossBlockRecorder(train, vocab_size=rhm.v,
                                      kernel_sizes=args.k_adapt, h_min=0.0,
                                      h_mode='match', n_min=1.0,
                                      alpha_smooth=1e-3, device=device,
                                      block_align=1, pad_mode='outside',
                                      side=side, words=words, prods=prods,
                                      s=args.s, v=args.v)
            sampler = ScheduledDiscreteScoreMachine(
                mach, vocab_size=rhm.v, grid_size=side,
                default_time_steps=side ** 2, mask_schedule=linear_mask_schedule,
                mask_token=rhm.v, reveal_one=True)
            gen = torch.Generator(device=device).manual_seed(seed + 9973)
            sampler.sample(n_samples=args.n_gen, device=device, generator=gen)

            cols = {k: [] for k in ('ks', 'nrev', 'nw', 'nrevb', 'nsl', 'nsr',
                                    'step')}
            for i, (kmap, msk, cnt, nw, rb, nsl, nsr) in enumerate(mach.rec):
                flat = lambda a: a.reshape(a.shape[0], -1)
                sel = np.nonzero(flat(msk))
                cols['ks'].append(flat(kmap)[sel])
                cols['step'].append(np.full(sel[1].shape, i))
                for name, arr in (('nrev', cnt), ('nw', nw), ('nrevb', rb),
                                  ('nsl', nsl), ('nsr', nsr)):
                    cols[name].append(flat(arr)[sel].astype(int))
            c = {k: np.concatenate(vv) for k, vv in cols.items()}
            ks = c['ks']

            # Permutation null.  r2 here is a group-means fit, so ANY extra
            # column raises it just by splitting the data finer.  Shuffling
            # nsr within each step preserves its marginal and its coupling to
            # time while destroying its per-site content, so the gain it still
            # shows is the purely mechanical floor the real gain must clear.
            rng = np.random.default_rng(12345)
            nsr_shuf = c['nsr'].copy()
            order = np.argsort(c['step'], kind='stable')
            bounds = np.searchsorted(c['step'][order], np.arange(len(mach.rec) + 1))
            for a, b in zip(bounds[:-1], bounds[1:]):
                if b > a:
                    nsr_shuf[order[a:b]] = rng.permutation(nsr_shuf[order[a:b]])

            base = r2(ks, joint(c['nrev'], c['nw']))
            base_l = r2(ks, joint(c['nrev'], c['nw'], c['nsl']))
            full = r2(ks, joint(c['nrev'], c['nw'], c['nsl'], c['nsr']))
            null = r2(ks, joint(c['nrev'], c['nw'], c['nsl'], nsr_shuf))
            # Is the gain just "this configuration is already globally dead"?
            # nsr == 0 means no root production survives every block at once.
            # That fires at 20-66% of observations and rises steeply with v, so
            # a binary doomed/alive flag could carry the whole gain without any
            # graded cross-block information in it.  r2_dead is that flag in
            # nsr's place; full - dead is what grading the count still buys.
            dead = (c['nsr'] == 0).astype(np.int64)
            r2_dead = r2(ks, joint(c['nrev'], c['nw'], c['nsl'], dead))
            tighter = float((c['nsr'] < c['nsl']).mean())
            row = dict(v=args.v, s=args.s, L=args.L, m=args.m, rule_seed=rs, P=P,
                       n_gen=args.n_gen, n_obs=int(ks.size),
                       k_mean=float(ks.mean()), k_sd=float(ks.std()),
                       nsl_mean=float(c['nsl'].mean()),
                       nsr_mean=float(c['nsr'].mean()),
                       frac_tighter=tighter,
                       frac_dead=float((c['nsr'] == 0).mean()),
                       r2_nrev=r2(ks, c['nrev']),
                       r2_base=base,                 # nrev + nw, the prior best
                       r2_base_nsl=base_l,           # + within-block control
                       r2_full=full,                 # + cross-block feature
                       r2_null=null,                 # + shuffled, the floor
                       r2_dead=r2_dead,              # + binary doomed flag
                       gain=full - base_l,
                       gain_null=null - base_l,
                       gain_dead=r2_dead - base_l,
                       gain_graded=full - r2_dead,
                       r2_nrev_nw_nsr=r2(ks, joint(c['nrev'], c['nw'], c['nsr'])),
                       k_by_tighter=[
                           float(ks[c['nsr'] < c['nsl']].mean())
                           if (c['nsr'] < c['nsl']).any() else float('nan'),
                           float(ks[c['nsr'] == c['nsl']].mean())
                           if (c['nsr'] == c['nsl']).any() else float('nan')])
            rows.append(row)
            print('v%d m%d rs%d P=%-5d k %.3f | base %.3f +nsl %.3f (ctrl %+.3f)'
                  ' +nsr %.3f (GAIN %+.3f, null %+.3f, net %+.3f) | dead flag'
                  ' %+.3f, graded %+.3f | tighter %.1f%% dead %.1f%%'
                  ' | k tight/loose %.2f/%.2f'
                  % (args.v, args.m, rs, P, row['k_mean'], base, base_l,
                     base_l - base, full, full - base_l, null - base_l,
                     full - null, r2_dead - base_l, full - r2_dead,
                     100 * tighter, 100 * row['frac_dead'],
                     row['k_by_tighter'][0], row['k_by_tighter'][1]))
            sys.stdout.flush()

            o = ROOT / args.out
            o.parent.mkdir(parents=True, exist_ok=True)
            json.dump(rows, open(o, 'w'))
    print('wrote %s' % (ROOT / args.out))


if __name__ == '__main__':
    main()
