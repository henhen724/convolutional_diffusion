"""Is the criterion's cross-block dependence really only one bit?

``rhm_k_crossblock.py`` found that the window the LS machine picks at a site
depends on whether the site's level-1 block is still consistent with the OTHER
blocks through the root production -- median +0.196 R^2 over (reveal count,
n_words, n_sym_local), 18/18 cells.  But it also found that a bare binary flag,
"no root production survives every block at once", reproduces that gain to
+0.000 in all 18.  The surviving-symbol COUNT bought nothing over the BIT, even
though the count was >= 2 at 10.6-27.0% of observations.

A hard survivor count is a coarse probe: it asks which symbols are *possible*,
never how probable.  Belief propagation is exact on this tree, so the graded
version of the same quantity is available in closed form:

  h_local[b] = H(z_1[b] | block b's own leaves)       -- normalized up[1]
  h_root[b]  = H(z_1[b] | ALL observed leaves)        -- normalized up[1]*down[1]

h_local is the graded n_sym_local and h_root the graded n_sym_root, computed
from one BP pass.  If the criterion tracks anything softer than the consistency
bit, h_root is where it shows up; if it does not, the one-bit reading stops
being an artifact of a coarse feature and becomes a result.

Registered before this run:

  1. h_root adds >= 0.02 R^2 over (nrev, n_words, h_local, dead), net of a
     null that shuffles it WITHIN each (step, dead) stratum -- so the null
     preserves both the time coupling and the dead/alive split, and the only
     thing it destroys is the graded content.  I am not confident in the
     direction here: the count said zero, and a null result would be the
     stronger finding, because the Bayes-optimal posterior adding nothing over
     one bit is a much harder statement than a coarse count adding nothing.
  2. Correctness cross-check, not a prediction: BP deadness (zero posterior
     mass at level 1) agrees with the enumeration's n_sym_root == 0 at > 99%
     of observations.  These are independent implementations of the same
     predicate, so disagreement means one is wrong and the run is void.
  3. Substrate, reported either way: among ALIVE observations h_root takes
     more than one distinct quantile bin, i.e. the graded feature has
     something to grade.  Without this, 1 collapses to "h_root is constant".

Also carried, UNREGISTERED and labelled so in the output: h_leaf, the exact
posterior entropy over the token at the site itself.  That is the quantity
hshunt named for this criterion -- "the maximum length scale which stays above
the minimum posterior entropy" -- whereas h_root is about the block's symbol.
It costs nothing (same BP pass) and it is the natural place to look if h_root
comes back null: that would say I measured the right idea at the wrong level of
the tree.  It is a diagnostic, not one of the three claims above.

The baseline is recomputed here from the same recorder as rhm_k_crossblock.py,
so the deltas subtract against that script's.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.random_hierarchy import RandomHierarchyModel2D  # noqa: E402
from src.utils.discrete_score import (  # noqa: E402
    ScheduledDiscreteScoreMachine, linear_mask_schedule)
from scripts.rhm_k_content import joint  # noqa: E402
from scripts.rhm_k_crossblock import CrossBlockRecorder, r2, _check_r2  # noqa: E402


def entropy(p, axis=-1):
    """Shannon entropy in nats, with 0 log 0 = 0."""
    q = np.where(p > 0, p, 1.0)
    return -(np.where(p > 0, p, 0.0) * np.log(q)).sum(axis)


def qbin(x, alive, nbins, tol=1e-12):
    """Bin x, keeping the zero atom separate from quantiles of the spread.

    Binning is needed because r2 is a group-means fit on discrete labels.
    Doing it over the alive subset only keeps the dead/alive split in its own
    column (``dead``) rather than smeared across the bins, which is what makes
    "graded on top of the bit" a clean question.

    The atom matters.  h_root is overwhelmingly exactly 0 on alive sites -- the
    root pins the block's symbol uniquely -- with a thin tail of genuinely
    uncertain blocks.  Plain quantile cuts then put far more than 1/nbins of
    the mass below the first cut, every cut lands on 0, np.unique collapses
    them, and the tail (the only part with graded content) gets no resolution
    at all.  Measured on the smoke cell: 33 distinct values, 2 bins.  A null
    gain under that binning would be my discretization, not the feature.  So
    dead -> 0, alive-and-zero -> 1, alive-and-positive -> quantiles of the
    positive part, which is where any grading has to live.
    """
    out = np.zeros(x.shape, dtype=np.int64)
    pos = alive & (x > tol)
    out[alive & ~pos] = 1
    if pos.any():
        xp = x[pos]
        cuts = np.unique(np.quantile(xp, np.linspace(0, 1, nbins + 1)[1:-1]))
        out[pos] = 2 + np.searchsorted(cuts, xp, side='right')
    return out


def shuffle_within(x, strata, rng):
    """Permute x inside each stratum, leaving the per-stratum marginal intact."""
    out = x.copy()
    order = np.argsort(strata, kind='stable')
    vals = strata[order]
    edges = np.flatnonzero(np.r_[True, vals[1:] != vals[:-1], True])
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a > 1:
            out[order[a:b]] = rng.permutation(out[order[a:b]])
    return out


class BPRecorder(CrossBlockRecorder):
    """CrossBlockRecorder plus the exact BP posteriors at level 1."""

    def __init__(self, *a, rhm, **kw):
        super().__init__(*a, **kw)
        self.rhm = rhm

    def _bp(self, grid_np):
        """(h_local, h_root, dead_root, dead_local) per block, and h_leaf per site."""
        rhm, v = self.rhm, self.v
        obs = grid_np != self.mask_token
        leaf_up = np.eye(v, dtype=np.float64)[np.where(obs, grid_np, 0)]
        leaf_up[~obs] = 1.0 / v

        up = rhm._bp_upward(leaf_up)
        down = rhm._bp_downward(up)

        def norm(u, d):
            c = u * d
            mass = c.sum(-1)
            return c / np.where(mass[..., None] > 0, mass[..., None], 1.0), mass <= 0

        # up[1] is already normalized by _bp_upward, so it IS the block-alone
        # posterior under the uniform symbol prior -- the graded n_sym_local.
        loc = up[1]
        dead_local = loc.sum(-1) <= 0          # block's own leaves ungrammatical
        root, dead_root = norm(up[1], down[1])  # graded n_sym_root
        # The exact posterior over the SITE's own token -- the quantity hshunt
        # named when he asked for "the maximum length scale which stays above
        # the minimum posterior entropy".  Per site, not per block.
        leaf, _ = norm(up[rhm.L], down[rhm.L])
        return entropy(loc), entropy(root), dead_root, dead_local, entropy(leaf)

    def forward(self, t, xt, label=None, device=None):
        xt = xt.to(self.machines[0].device)
        masked = (xt == self.mask_token)
        out = super(CrossBlockRecorder, self).forward(
            t, xt, label=label, device=device)
        K = self.kernel_sizes[-1]
        grid = xt.reshape(-1, self.side, self.side)
        rev = (~masked).float().reshape(-1, 1, self.side, self.side)
        cnt = torch.nn.functional.conv2d(
            rev, torch.ones(1, 1, K, K, device=rev.device), padding=K // 2)
        nw, nrevb, nsl, nsr = self._features(grid)

        hl, hr, dr, dl, hleaf = self._bp(grid.detach().cpu().numpy())
        # The level-1 quantities are per block; broadcast them to the sites the
        # block covers.  h_leaf is already per site.
        rep = lambda a: np.repeat(np.repeat(a, self.s, 1), self.s, 2)

        self.rec.append(tuple(
            [a.detach().cpu().numpy().copy()
             for a in (self._last_kmap, masked, cnt, nw, nrevb, nsl, nsr)]
            + [rep(a) for a in (hl, hr, dr, dl)] + [hleaf]))
        return out


def run_cell(args, rs, P, device):
    rhm = RandomHierarchyModel2D(v=args.v, s=args.s, L=args.L, m=args.m,
                                 seed=1000 + rs)
    seed = 1234 + 7 * rs
    side = rhm.grid_size
    assert side // args.s == args.s, 'root production is s x s; needs L = 2'
    words = torch.as_tensor(rhm.rules[-1], device=device)
    prods = torch.as_tensor(rhm.rules[0].reshape(-1, args.s, args.s),
                            device=device)

    train, _ = rhm.sample(P, seed=seed)
    mach = BPRecorder(train, vocab_size=rhm.v, kernel_sizes=args.k_adapt,
                      h_min=0.0, h_mode='match', n_min=1.0, alpha_smooth=1e-3,
                      device=device, block_align=1, pad_mode='outside',
                      side=side, words=words, prods=prods, s=args.s, v=args.v,
                      rhm=rhm)
    sampler = ScheduledDiscreteScoreMachine(
        mach, vocab_size=rhm.v, grid_size=side,
        default_time_steps=side ** 2, mask_schedule=linear_mask_schedule,
        mask_token=rhm.v, reveal_one=True)
    gen = torch.Generator(device=device).manual_seed(seed + 9973)
    sampler.sample(n_samples=args.n_gen, device=device, generator=gen)

    names = ('nrev', 'nw', 'nrevb', 'nsl', 'nsr', 'hl', 'hr', 'dr', 'dl', 'hlf')
    cols = {k: [] for k in names + ('ks', 'step')}
    for i, rec in enumerate(mach.rec):
        kmap, msk = rec[0], rec[1]
        flat = lambda a: a.reshape(a.shape[0], -1)
        sel = np.nonzero(flat(msk))
        cols['ks'].append(flat(kmap)[sel])
        cols['step'].append(np.full(sel[1].shape, i))
        for name, arr in zip(names, rec[2:]):
            cols[name].append(flat(arr)[sel])
    c = {k: np.concatenate(v) for k, v in cols.items()}
    for k in ('nrev', 'nw', 'nrevb', 'nsl', 'nsr'):
        c[k] = c[k].astype(np.int64)
    ks = c['ks']

    dead = c['dr'].astype(np.int64)
    alive = ~c['dr']
    # Registered check 2: two independent implementations of the same predicate.
    agree = float((c['dr'] == (c['nsr'] == 0)).mean())
    hl_bin = qbin(c['hl'], np.ones_like(alive), args.nbins)
    hr_bin = qbin(c['hr'], alive, args.nbins)
    n_hr_bins = int(len(np.unique(hr_bin[alive]))) if alive.any() else 0
    frac_hr_zero = float((c['hr'][alive] <= 1e-12).mean()) if alive.any() else float('nan')
    # Check 3 is about the FEATURE, not the binning: if h_root takes many
    # distinct values but few quantile bins, the bins collapsed a real spread
    # and a null in 1 would be my own discretization.  If it takes few distinct
    # values, h_root is genuinely near-atomic on alive sites and the honest
    # reading of a null in 1 is "there is nothing softer here", not "graded
    # information does not matter".  These are different findings, so measure
    # both rather than inferring one from the other.
    hr_alive = c['hr'][alive]
    n_hr_vals = int(len(np.unique(np.round(hr_alive, 9)))) if alive.any() else 0
    hr_q = ([float(x) for x in np.quantile(hr_alive, [0, .25, .5, .75, 1])]
            if alive.any() else [float('nan')] * 5)

    rng = np.random.default_rng(12345)
    strata = c['step'].astype(np.int64) * 2 + dead
    hr_shuf = shuffle_within(hr_bin, strata, rng)

    base = r2(ks, joint(c['nrev'], c['nw']))
    base_h = r2(ks, joint(c['nrev'], c['nw'], hl_bin))
    base_d = r2(ks, joint(c['nrev'], c['nw'], hl_bin, dead))
    full = r2(ks, joint(c['nrev'], c['nw'], hl_bin, dead, hr_bin))
    null = r2(ks, joint(c['nrev'], c['nw'], hl_bin, dead, hr_shuf))
    # The count's own answer on the same rows, for a like-for-like comparison.
    cnt_d = r2(ks, joint(c['nrev'], c['nw'], c['nsl'], dead))
    cnt_full = r2(ks, joint(c['nrev'], c['nw'], c['nsl'], dead, c['nsr']))

    # UNREGISTERED, exploratory -- added while writing this script, so it does
    # not carry the weight of 1-3 and is reported as a diagnostic.  h_root asks
    # about the block's SYMBOL; h_leaf is the exact posterior over the token at
    # the site being denoised, which is the quantity hshunt actually named:
    # "the maximum length scale which stays above the minimum posterior
    # entropy".  If the criterion tracks a posterior entropy at all, this is
    # the one it should track, and the h_root null would then be telling me I
    # measured the right idea at the wrong level of the tree.
    hlf_bin = qbin(c['hlf'], alive, args.nbins)
    hlf_shuf = shuffle_within(hlf_bin, strata, rng)
    leaf_alone = r2(ks, hlf_bin)
    leaf_full = r2(ks, joint(c['nrev'], c['nw'], hl_bin, dead, hlf_bin))
    leaf_null = r2(ks, joint(c['nrev'], c['nw'], hl_bin, dead, hlf_shuf))
    n_hlf_bins = int(len(np.unique(hlf_bin[alive]))) if alive.any() else 0

    row = dict(v=args.v, s=args.s, L=args.L, m=args.m, rule_seed=rs, P=P,
               n_gen=args.n_gen, n_obs=int(ks.size), nbins=args.nbins,
               k_mean=float(ks.mean()),
               bp_enum_agree=agree,
               frac_dead_bp=float(c['dr'].mean()),
               frac_dead_enum=float((c['nsr'] == 0).mean()),
               frac_dead_local=float(c['dl'].mean()),
               h_local_mean=float(c['hl'].mean()),
               h_root_mean_alive=float(c['hr'][alive].mean()) if alive.any() else float('nan'),
               n_hr_bins_alive=n_hr_bins,
               n_hr_vals_alive=n_hr_vals,
               frac_hr_zero_alive=frac_hr_zero,
               hr_quantiles_alive=hr_q,
               r2_base=base,                  # nrev + nw
               r2_base_hloc=base_h,           # + graded within-block
               r2_base_dead=base_d,           # + the one bit
               r2_full=full,                  # + graded cross-block
               r2_null=null,                  # + shuffled within (step, dead)
               gain_graded=full - base_d,
               gain_graded_net=full - null,
               gain_dead=base_d - base_h,
               r2_cnt_dead=cnt_d,
               r2_cnt_full=cnt_full,
               gain_cnt_graded=cnt_full - cnt_d,
               h_leaf_mean_alive=float(c['hlf'][alive].mean()) if alive.any() else float('nan'),
               n_hlf_bins_alive=n_hlf_bins,
               r2_hleaf_alone=leaf_alone,
               r2_hleaf_full=leaf_full,
               r2_hleaf_null=leaf_null,
               gain_hleaf=leaf_full - base_d,
               gain_hleaf_net=leaf_full - leaf_null)
    print('v%d m%d rs%d P=%-5d k %.3f | base %.3f +hloc %.3f +dead %.3f'
          ' +hroot %.3f (GAIN %+.3f, net %+.3f) | count path %+.3f'
          ' | hr bins %d vals %d zero %.0f%%, dead %.1f%% (agree %.4f)'
          ' | hleaf alone %.3f full %+.3f net %+.3f (bins %d)'
          % (args.v, args.m, rs, P, row['k_mean'], base, base_h, base_d, full,
             full - base_d, full - null, cnt_full - cnt_d, n_hr_bins, n_hr_vals,
             100 * frac_hr_zero, 100 * row['frac_dead_bp'], agree,
             leaf_alone, leaf_full - base_d, leaf_full - leaf_null, n_hlf_bins))
    sys.stdout.flush()
    return row


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
    ap.add_argument('--nbins', type=int, default=8,
                    help='quantile bins for the entropies; fixed so that the '
                         'group cardinality does not drift between cells')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    if args.L != 2:
        raise SystemExit('the root-production feature is written for L = 2 only')
    _check_r2()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows = []
    for rs in range(args.rule_seeds):
        for P in args.P:
            rows.append(run_cell(args, rs, P, device))
            o = ROOT / args.out
            o.parent.mkdir(parents=True, exist_ok=True)
            json.dump(rows, open(o, 'w'))
    print('wrote', ROOT / args.out)


if __name__ == '__main__':
    main()
