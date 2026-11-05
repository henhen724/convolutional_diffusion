"""Does the window criterion read the NEAR coupling, the FAR one, or one bit?

At s=3 L=2 -- where every cross-block result so far was measured -- a site's
level-1 block has exactly one thing above it: the root.  "Consistent with the
root" and "consistent with the other blocks" are therefore the same sentence,
and the finding that the dependence is a single bit (rhm_k_crossblock.py,
rhm_k_bp.py, 18/18 each) cannot distinguish a local consistency check from a
global one.  L=3 separates them.  Writing l* = L-1 for the level of a site's
own smallest grammatical block (3x3 leaves):

  h_local  = H(z_l* | the block's OWN leaves)                      up[l*]
  h_parent = H(z_l* | + its siblings under the same level-(l*-1) parent)
  h_root   = H(z_l* | + every other region, through the root)      up[l*]*down[l*]

At L=3 the parent is a 9x9 region and the root couples nine of them, so
h_parent is the near coupling and h_root - h_parent is the far one.  At L=2 the
parent IS the root, so h_parent and h_root must coincide exactly -- that is a
free correctness check on the machinery, not a result.

Registered before the L=3 run:

  1. NEAR is used: the bit dead_parent ("no symbol survives inside the parent's
     subtree") adds >= 0.02 R^2 over (nrev, n_words, h_local, dead_local), net
     of a null shuffling it within each (step, dead_local) stratum.  Confident:
     this is the L=2 effect, which is a within-parent effect there too.
  2. FAR on top of near: dead_root adds >= 0.02 over the same baseline PLUS
     dead_parent.  No confident direction.  A null means the criterion checks
     consistency only as far as the nearest enclosing block and the L=2 result
     was the near coupling all along, read at the only depth where near and far
     coincide.  A gain means it genuinely reaches the root.
  3. Still one bit at depth: graded h_root adds < 0.02 on top of all the bits,
     replicating rhm_k_bp.py at a depth it has never been tested at.

  Checks, reported either way, that decide whether 1-3 mean anything:
  4. At L=2, h_parent == h_root to 1e-12 at every observation, and the
     recorder's dead_root reproduces rhm_k_crossblock's +0.19-ish bit gain.
     Independent machinery must land on the already-measured number.
  5. Substrate: dead_parent and dead_root each fire at > 5% of observations AND
     disagree with each other at > 5%.  If they never disagree there is no far
     coupling to detect and 2 is vacuous rather than null.

The k ladder must contain a window wide enough to straddle a level-1 boundary
(k > s^(L-1) = 9 at L=3), or 2 is answered by construction: a criterion that
cannot see across a 9x9 region cannot use the far coupling.  --k_adapt is
checked against that and the run refuses if it is too narrow.
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
from src.utils.rhm_sampling import AdaptiveWindowDiscreteScoreMachine  # noqa: E402
from scripts.rhm_k_content import joint  # noqa: E402
from scripts.rhm_k_crossblock import r2, _check_r2  # noqa: E402
from scripts.rhm_k_bp import entropy, qbin, shuffle_within  # noqa: E402


def down_step(rhm, l, parent_down, up):
    """One level of ``_bp_downward``: down[l] (+ up[l+1]) -> down[l+1].

    Lifted out of the method so the PARENT-RESTRICTED message can be built by
    passing a uniform ``parent_down``: that yields P(z_{l+1} | everything
    inside the parent's subtree but outside this node's own), where the full
    down[l+1] also carries everything outside the parent, through the root.
    ``_check_down_step`` asserts this reproduces the method exactly.
    """
    v, s, m = rhm.v, rhm.s, rhm.m
    rule = rhm.rules[l]                      # (v, m, s, s)
    n, nb = parent_down.shape[0], parent_down.shape[1]
    blocks_up = up[l + 1].reshape(n, nb, s, nb, s, v).transpose(0, 1, 3, 2, 4, 5)
    new_blocks = np.zeros((n, nb, nb, s, s, v), dtype=np.float64)
    for a in range(v):
        weight = parent_down[:, :, :, a] * (1.0 / m)
        for wi in range(m):
            for di_t in range(s):
                for dj_t in range(s):
                    loo = np.ones((n, nb, nb), dtype=np.float64)
                    for di in range(s):
                        for dj in range(s):
                            if (di, dj) == (di_t, dj_t):
                                continue
                            loo = loo * blocks_up[:, :, :, di, dj, rule[a, wi, di, dj]]
                    new_blocks[:, :, :, di_t, dj_t, rule[a, wi, di_t, dj_t]] += weight * loo
    out = new_blocks.transpose(0, 1, 3, 2, 4, 5).reshape(n, nb * s, nb * s, v)
    out /= out.sum(-1, keepdims=True).clip(min=1e-300)
    return out


def _check_down_step(v=3, s=3, L=3, m=2, seed=7, n=6):
    """down_step must reproduce rhm._bp_downward at every level, or it is wrong."""
    rhm = RandomHierarchyModel2D(v=v, s=s, L=L, m=m, seed=seed)
    rng = np.random.default_rng(seed)
    x, _ = rhm.sample(n, seed=seed)
    x = np.asarray(x).copy()
    x[rng.random(x.shape) < 0.5] = rhm.v          # erase half the leaves
    obs = x != rhm.v
    leaf_up = np.eye(rhm.v, dtype=np.float64)[np.where(obs, x, 0)]
    leaf_up[~obs] = 1.0 / rhm.v
    up = rhm._bp_upward(leaf_up)
    down = rhm._bp_downward(up)
    for l in range(L):
        mine = down_step(rhm, l, down[l], up)
        if not np.allclose(mine, down[l + 1], rtol=0, atol=1e-12):
            raise SystemExit('down_step disagrees with _bp_downward at l=%d '
                             '(max |d| = %.3e)' % (l, np.abs(mine - down[l + 1]).max()))


class DepthRecorder(AdaptiveWindowDiscreteScoreMachine):
    """Records the chosen k with the local / parent / root BP posteriors.

    General in L, unlike CrossBlockRecorder, whose n_sym_root enumerates root
    productions over nb == s blocks and so only exists at L = 2.
    """

    def __init__(self, *a, side, s, v, rhm, **kw):
        # mask_token deliberately not forwarded: the parent hands the raw
        # argument to its sub-machines, so leaving it None keeps this the same
        # machine the earlier scripts measured.
        super().__init__(*a, **kw)
        self.side, self.s, self.v, self.rhm = side, s, v, rhm
        self.rec = []

    def _words_feat(self, xt):
        """n_words and n_sym_local for each site's own level-(L-1) leaf block."""
        rhm, s = self.rhm, self.s
        B, nb = xt.shape[0], self.side // s
        x = xt.reshape(B, nb, s, nb, s).permute(0, 1, 3, 2, 4)
        msk = (x == self.mask_token)
        words = torch.as_tensor(rhm.rules[-1], device=xt.device)   # (v, m, s, s)
        v, m = words.shape[0], words.shape[1]
        w = words.reshape(1, 1, 1, v, m, s, s)
        agree = (msk.unsqueeze(3).unsqueeze(4) |
                 (x.unsqueeze(3).unsqueeze(4) == w)).all(-1).all(-1)
        nw = agree.reshape(B, nb, nb, v * m).sum(-1)
        nsl = agree.any(-1).sum(-1)
        # Per block -> per site, as CrossBlockRecorder._features does; the
        # recorder indexes everything by masked SITE.
        rep = lambda t: t.repeat_interleave(s, 1).repeat_interleave(s, 2)
        return rep(nw), rep(nsl)

    def _bp(self, grid_np):
        """Per level-(L-1) block: entropies and dead flags, local/parent/root."""
        rhm, v, L = self.rhm, self.v, self.rhm.L
        obs = grid_np != self.mask_token
        leaf_up = np.eye(v, dtype=np.float64)[np.where(obs, grid_np, 0)]
        leaf_up[~obs] = 1.0 / v

        up = rhm._bp_upward(leaf_up)
        down = rhm._bp_downward(up)
        ls = L - 1                                   # the site's own block level

        # Near: start from a UNIFORM prior at the parent, so nothing outside the
        # parent's subtree can reach the node.  At L=2 the parent is the root and
        # down[0] is already uniform, so this must equal the full message.
        n = leaf_up.shape[0]
        nb_par = rhm.s ** (ls - 1)
        unif = np.full((n, nb_par, nb_par, v), 1.0 / v)
        down_par = down_step(rhm, ls - 1, unif, up)

        def post(u, d):
            c = u * d
            mass = c.sum(-1)
            return c / np.where(mass[..., None] > 0, mass[..., None], 1.0), mass <= 0

        loc = up[ls]                                 # already normalized upward
        dead_local = loc.sum(-1) <= 0
        par, dead_par = post(up[ls], down_par)
        root, dead_root = post(up[ls], down[ls])
        return (entropy(loc), entropy(par), entropy(root),
                dead_local, dead_par, dead_root)

    def forward(self, t, xt, label=None, device=None):
        xt = xt.to(self.machines[0].device)
        masked = (xt == self.mask_token)
        out = super().forward(t, xt, label=label, device=device)
        K = self.kernel_sizes[-1]
        grid = xt.reshape(-1, self.side, self.side)
        rev = (~masked).float().reshape(-1, 1, self.side, self.side)
        cnt = torch.nn.functional.conv2d(
            rev, torch.ones(1, 1, K, K, device=rev.device), padding=K // 2)
        nw, nsl = self._words_feat(grid)
        bp = self._bp(grid.detach().cpu().numpy())
        rep = lambda a: np.repeat(np.repeat(a, self.s, 1), self.s, 2)
        self.rec.append(tuple(
            [a.detach().cpu().numpy().copy()
             for a in (self._last_kmap, masked, cnt, nw, nsl)]
            + [rep(a) for a in bp]))
        return out


def run_cell(args, rs, P, device):
    rhm = RandomHierarchyModel2D(v=args.v, s=args.s, L=args.L, m=args.m,
                                 seed=1000 + rs)
    seed, side = 1234 + 7 * rs, rhm.grid_size
    train, _ = rhm.sample(P, seed=seed)
    mach = DepthRecorder(train, vocab_size=rhm.v, kernel_sizes=args.k_adapt,
                         h_min=0.0, h_mode='match', n_min=1.0, alpha_smooth=1e-3,
                         device=device, block_align=1, pad_mode='outside',
                         side=side, s=args.s, v=args.v, rhm=rhm)
    sampler = ScheduledDiscreteScoreMachine(
        mach, vocab_size=rhm.v, grid_size=side,
        default_time_steps=side ** 2, mask_schedule=linear_mask_schedule,
        mask_token=rhm.v, reveal_one=True)
    gen = torch.Generator(device=device).manual_seed(seed + 9973)
    sampler.sample(n_samples=args.n_gen, device=device, generator=gen)

    names = ('nrev', 'nw', 'nsl', 'hl', 'hp', 'hr', 'dl', 'dp', 'dr')
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
    for k in ('nrev', 'nw', 'nsl'):
        c[k] = c[k].astype(np.int64)
    ks = c['ks']

    # Check 4: at L=2 the parent IS the root, so these must be the same number.
    hp_hr_max = float(np.abs(c['hp'] - c['hr']).max())
    dl = c['dl'].astype(np.int64)
    dp = c['dp'].astype(np.int64)
    dr = c['dr'].astype(np.int64)
    alive_l = ~c['dl']
    # Check 5: the far coupling must actually do something beyond the near one.
    disagree = float((c['dp'] != c['dr']).mean())

    hl_bin = qbin(c['hl'], np.ones_like(alive_l), args.nbins)
    hr_bin = qbin(c['hr'], ~c['dr'], args.nbins)
    rng = np.random.default_rng(12345)
    strata = c['step'].astype(np.int64) * 2 + dl
    dp_shuf = shuffle_within(dp, strata, rng)
    dr_shuf = shuffle_within(dr, c['step'].astype(np.int64) * 4 + dl * 2 + dp, rng)

    base = r2(ks, joint(c['nrev'], c['nw'], hl_bin, dl))
    near = r2(ks, joint(c['nrev'], c['nw'], hl_bin, dl, dp))
    near_null = r2(ks, joint(c['nrev'], c['nw'], hl_bin, dl, dp_shuf))
    far = r2(ks, joint(c['nrev'], c['nw'], hl_bin, dl, dp, dr))
    far_null = r2(ks, joint(c['nrev'], c['nw'], hl_bin, dl, dp, dr_shuf))
    graded = r2(ks, joint(c['nrev'], c['nw'], hl_bin, dl, dp, dr, hr_bin))

    row = dict(v=args.v, s=args.s, L=args.L, m=args.m, rule_seed=rs, P=P,
               n_gen=args.n_gen, n_obs=int(ks.size), nbins=args.nbins,
               side=side, k_adapt=list(args.k_adapt), k_mean=float(ks.mean()),
               hp_hr_maxdiff=hp_hr_max,
               frac_dead_local=float(c['dl'].mean()),
               frac_dead_parent=float(c['dp'].mean()),
               frac_dead_root=float(c['dr'].mean()),
               frac_parent_root_disagree=disagree,
               n_hr_bins_alive=int(len(np.unique(hr_bin[~c['dr']])))
               if (~c['dr']).any() else 0,
               r2_base=base, r2_near=near, r2_near_null=near_null,
               r2_far=far, r2_far_null=far_null, r2_graded=graded,
               gain_near=near - base, gain_near_net=near - near_null,
               gain_far=far - near, gain_far_net=far - far_null,
               gain_graded=graded - far)
    print('v%d m%d rs%d P=%-5d L%d k %.3f | base %.3f +near %.3f (%+.3f net %+.3f)'
          ' +far %.3f (%+.3f net %+.3f) +graded %+.3f | dead l/p/r %.0f/%.0f/%.0f%%'
          ' disagree %.1f%% | hp-hr %.1e'
          % (args.v, args.m, rs, P, args.L, row['k_mean'], base, near,
             near - base, near - near_null, far, far - near, far - far_null,
             graded - far, 100 * row['frac_dead_local'],
             100 * row['frac_dead_parent'], 100 * row['frac_dead_root'],
             100 * disagree, hp_hr_max))
    sys.stdout.flush()
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--v', type=int, default=2)
    ap.add_argument('--s', type=int, default=3)
    ap.add_argument('--L', type=int, default=3)
    ap.add_argument('--m', type=int, default=2)
    ap.add_argument('--rule_seeds', type=int, default=3)
    ap.add_argument('--P', type=int, nargs='+', default=[300])
    ap.add_argument('--k_adapt', type=int, nargs='+', default=[5, 11, 17])
    ap.add_argument('--n_gen', type=int, default=128)
    ap.add_argument('--nbins', type=int, default=8)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    if args.L < 2:
        raise SystemExit('need L >= 2 for a block to have a parent')
    # Prediction 2 is answered by construction unless some window can straddle a
    # level-(L-2) boundary -- a criterion that cannot see past its own parent
    # region cannot use the far coupling, so a null would mean nothing.
    need = args.s ** (args.L - 1)
    if args.L > 2 and max(args.k_adapt) <= need:
        raise SystemExit('k_adapt tops out at %d, which cannot straddle a %d-wide '
                         'level-1 region: prediction 2 would be vacuous'
                         % (max(args.k_adapt), need))
    _check_r2()
    _check_down_step(v=args.v, s=args.s, L=args.L, m=args.m)

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
