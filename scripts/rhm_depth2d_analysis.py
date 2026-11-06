"""Is x* = P*/|R_L| depth-stable in 2D at FIXED grammar shape (v, m)?

Section A's `P* = 0.21 |R_L|` needs x* not to drift with depth.  The evidence
for that today is L=2 pooled over nine (v,m) combos (0.1925) against L=3 pooled
over two (0.2075) -- and those two sets of combos are disjoint, while x* varies
7.6x across (v,m) medians.  Pooling over disjoint cells cannot tell depth from
shape.  On the 1D side the same pooling looked flat and the paired comparison
was not: x* falls with depth in 15/15 (v,m) combos, median ~2.8x, which is why
`P* = Theta(|R_L|)` fails in 1D.

So this pairs within (v,m).  51244089 supplies the missing 2D L=2 m=2 peaks at
v=2,3 (m=2 had only ever been run at L=2 for v=4); L=3 m=2 at v=2,3 is already
published.  v=4 is the control: it is the one m=2 cell that exists at L=2 under
both the published settings and the new ones, so a disagreement there means the
ladder or the seed count moved the answer, not depth.

Two measurement details that the published pipeline leaves on the floor:

  Duplicate ladder points.  `L3k7_v*_m*.json` (n_gen=128) and
  `L3peak_v*_k7.json` (n_gen=1024) both cover P = 1e4, 3e4, 1e5, so after the
  sort by P every L=3 curve carries three adjacent pairs of points measuring the
  same thing at an 8x difference in precision.  `rhm_peak_analysis.curves` keeps
  both, which puts a zigzag right where a 3-point parabola is most fragile.
  Here the duplicates are pooled into one binomial (counts add), which is what
  they are.

  Peak estimator.  Reported under both the published 3-point fit (A) and the
  inverse-variance fit over the whole top (B) from `rhm_peak_noise`, because
  A's bootstrap bias blows up on exactly the sparse L=3 curves this comparison
  depends on (v3m2 rs1: 0.130).

Uncertainty is a nested bootstrap: resample each cell's success counts to get
that cell's P* replicates, resample which cells are in the median, and carry
both through the L=2 / L=3 ratio.  A ratio CI that covers 1 means no resolvable
depth effect -- which, given the L=3 side is 3 seeds at n_gen=1024 with
per-cell spread up to 3.5x, is not the same as depth-stability.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))

from rhm_peak_noise import U1, est_A, est_B, geo_sd  # noqa: E402


def load_RL(paths):
    out = {}
    for p in paths:
        for f in glob.glob(str(p)):
            for r in json.load(open(f)):
                out[(r['v'], r['m'], r['s'], r['L'], r['rule_seed'])] = r['R_L']
    return out


def copy_mask(Ps, succ, gram, td, R, copy_rel):
    """Leading-edge cut on the copying regime, in units of the cell's own curve.

    An absolute `P >= 10` floor is the wrong shape for this: the peak sits at
    P* ~ 0.2 |R_L|, and v=2 m=2 L=2 has |R_L| = 38..64, so the floor lands on
    the peak of exactly the cells the depth test needs.  `rhm_copy_regime.py`
    shows 4 of 29 L=2 cells have 0.2 |R_L| < 10, all of them at v=2 -- which is
    the one v where the paired test claimed an effect.

    The regime itself is visible without mentioning P.  Among the grammatical
    outputs, the novelty rate relative to a uniform draw from the support,

        ratio = (success / grammatical) / (1 - train_distinct / |R_L|),

    runs 0.006 -> 0.35 across the v=4 ladder and 0.19 -> 0.50 across v=2's: the
    machine starts out replaying its training set and then spreads out.  It
    never reaches 1 at L=2 (it does at L=3, ~0.92), so a fixed threshold on
    `ratio` is not transferable -- it would push v=4's argmax from 100 to 500
    and v=3's off the ladder.  What transfers is the comparison to the cell's
    OWN plateau: take `ref` as the median ratio over the top half of the curve
    and drop the LEADING run of points below `copy_rel * ref`.  Leading only --
    the zero-success tail at large P is real and defines the falling side.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        nov_g = np.where(gram > 0, succ / np.where(gram > 0, gram, 1.0), np.nan)
        chance = 1.0 - np.minimum(td, R) / R
        ratio = np.where(chance > 0, nov_g / np.where(chance > 0, chance, 1.0), np.nan)
    keep = np.ones(len(Ps), bool)
    if copy_rel <= 0 or succ.max() <= 0:
        return keep, ratio
    top = succ >= 0.5 * succ.max()
    ref = np.nanmedian(ratio[top]) if np.isfinite(ratio[top]).any() else np.nan
    if not np.isfinite(ref) or ref <= 0:
        return keep, ratio
    ok = np.isfinite(ratio) & (ratio >= copy_rel * ref)
    first = int(np.argmax(ok)) if ok.any() else len(Ps)
    keep[:first] = False
    return keep, ratio


def load_curves(specs, RL, P_min=0.0, copy_rel=0.0):
    """[(tag, glob)] -> {(tag, v, m, s, L, rs, k): (P, succ, n)} at k = U1."""
    grouped = defaultdict(list)
    for tag, pat in specs:
        for f in glob.glob(str(ROOT / pat)):
            for r in json.load(open(f)):
                if 'success' not in r:
                    continue
                if r.get('block_align', 1) != 1 or not isinstance(r.get('k'), int):
                    continue
                if r['k'] != U1(r['s'], r['L']):
                    continue
                key = (tag, r['v'], r['m'], r['s'], r['L'], r['rule_seed'], r['k'])
                grouped[key].append(r)

    out = {}
    for key, rows in grouped.items():
        if key[1:6] not in RL:
            continue
        # Pool duplicate ladder points: same P measured twice is one binomial
        # with the counts added, not two observations to be interleaved.
        by_P = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
        for r in rows:
            n = float(r.get('n_gen', r.get('n_samples', 4096)))
            e = by_P[r['P']]
            e[0] += r['success'] * n
            e[1] += n
            e[2] += r.get('grammatical', float('nan')) * n
            e[3] = max(e[3], float(r.get('train_distinct', r['P'])))
        order = sorted(by_P)
        Ps = np.array(order, float)
        hits = np.array([by_P[p][0] for p in order], float)
        ns = np.array([by_P[p][1] for p in order], float)
        gram = np.array([by_P[p][2] for p in order], float)
        td = np.array([by_P[p][3] for p in order], float)
        succ = hits / ns
        keep, _ = copy_mask(Ps, succ, gram / ns, td, float(RL[key[1:6]]), copy_rel)
        keep &= Ps >= P_min
        if keep.sum() < 3:
            continue
        out[key] = (Ps[keep], succ[keep], ns[keep])
    return out


def cell_estimates(P, S, N, n_boot, frac, rng):
    """Point estimates plus bootstrap replicates of P*, for A and B.

    B runs with fallback=True here: its rejections track curve shape, so
    dropping those cells would take the median over a biased subsample.
    """
    ng = float(np.median(N))
    pa, pb = est_A(P, S, ng), est_B(P, S, ng, frac, fallback=True)
    ba = np.empty(n_boot)
    bb = np.empty(n_boot)
    for j in range(n_boot):
        # Per-point n, so the sparse end of an L=3 curve is resampled at the
        # precision it was actually measured with.
        Sb = rng.binomial(N.astype(int), np.clip(S, 0, 1)) / N
        ba[j] = est_A(P, Sb, ng)
        bb[j] = est_B(P, Sb, ng, frac, fallback=True)
    return pa, pb, ba, bb


def med_ratio_ci(num_cells, den_cells, n_boot, rng, q=(2.5, 97.5)):
    """Nested bootstrap on median(num) / median(den) across cells.

    Resamples which cells enter the median AND which bootstrap replicate each
    contributes, so cell-to-cell spread and within-cell estimator noise both
    land in the interval.
    """
    if not num_cells or not den_cells:
        return (np.nan, np.nan, np.nan)
    draws = []
    for _ in range(n_boot):
        vals = []
        for cells in (num_cells, den_cells):
            idx = rng.integers(0, len(cells), len(cells))
            picked = []
            for i in idx:
                reps = cells[i]
                reps = reps[np.isfinite(reps)]
                if len(reps):
                    picked.append(reps[rng.integers(0, len(reps))])
            vals.append(np.median(picked) if picked else np.nan)
        if np.isfinite(vals[0]) and np.isfinite(vals[1]) and vals[1] > 0:
            draws.append(vals[0] / vals[1])
    if len(draws) < 20:
        return (np.nan, np.nan, np.nan)
    d = np.array(draws)
    return (float(np.median(d)), float(np.percentile(d, q[0])),
            float(np.percentile(d, q[1])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--old', default='results/rhm_sampling/2026-11-03')
    ap.add_argument('--new', default='results/rhm_sampling/2026-11-05')
    ap.add_argument('--boot', type=int, default=400)
    ap.add_argument('--P_min', type=float, default=10,
                    help='drop ladder points below this before locating the '
                         'peak. 51244089 ran down to P=2 and found a spike '
                         'there that is the copying regime, not the peak the '
                         'scaling law is about: v4m2 rs3 has success 0.0070 at '
                         'P=3 against 0.0032 on the broad hump at P=100, with '
                         'grammaticality 0.238 at P=3 falling to 0.020 by P=8. '
                         'With two or three training grids the machine has '
                         'almost no window dictionary to recombine, so it '
                         'near-copies -- grammatical, and novel only because '
                         'the training set is too small to match against. The '
                         'spike is sharp (one ladder point) and seed-specific, '
                         'so it is a lottery over which few grids were drawn, '
                         'not a scale. Every published 2D L=2 ladder starts at '
                         'P=10, so this floor also makes the v=4 control a '
                         'like-for-like comparison. Set 0 to see it, and '
                         'prefer --copy_rel, which cuts the same regime '
                         'without censoring small-|R_L| cells.')
    ap.add_argument('--copy_rel', type=float, default=0.0,
                    help='cut the copying regime by its own signature instead '
                         'of by an absolute P. Drops the leading run of ladder '
                         'points whose novelty-among-grammatical rate is below '
                         'this fraction of the cell\'s plateau value; see '
                         'copy_mask and scripts/rhm_copy_regime.py. 0.5 is the '
                         'tested setting. Combine with --P_min 0.')
    ap.add_argument('--frac', type=float, default=0.5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    old, new = ROOT / a.old, ROOT / a.new
    RL = load_RL([old / 'nk_exact_L*.json', new / 'nk_exact_L*.json'])
    curves = load_curves([
        ('L2new', a.new + '/depth2d_L2_v*_m2.json'),
        ('L2pub', a.old + '/P_v*_m2.json'),
        ('L3pub', a.old + '/L3peak_v*_k7.json'),
        ('L3pub', a.old + '/L3k7_v*_m*.json'),
        # Same tag on purpose: 51244750's extra seeds run the published
        # settings exactly (n_gen 1024, nsteps 64, k=7, block_align 1,
        # pad_mode outside), so they are more of that population, not a new
        # one. Only the seeds and two ladder points differ.
        ('L3pub', a.new + '/L3seed_v*_m2_k7.json'),
    ], RL, a.P_min, a.copy_rel)
    if not curves:
        raise SystemExit('no curves; has 51244089 finished?')

    rng = np.random.default_rng(a.seed)
    recs, reps = [], {}
    for key in sorted(curves):
        tag, v, m, s, L, rs, k = key
        if m != 2:
            continue
        P, S, N = curves[key]
        R = RL[key[1:6]]
        pa, pb, ba, bb = cell_estimates(P, S, N, a.boot, a.frac, rng)
        recs.append(dict(tag=tag, v=v, m=m, s=s, L=L, rule_seed=rs, k=k, R_L=R,
                         n_P=len(P), n_gen=float(np.median(N)),
                         succ_max=float(S.max()),
                         x_A=(pa / R) if np.isfinite(pa) else float('nan'),
                         x_B=(pb / R) if np.isfinite(pb) else float('nan'),
                         boot_sd_A=geo_sd(ba), boot_sd_B=geo_sd(bb)))
        reps[(tag, v, L, rs)] = (ba / R, bb / R)

    print('cells: %d   (m=2 only, k=U1, block_align=1)   P_min=%g copy_rel=%g'
          % (len(recs), a.P_min, a.copy_rel))
    print()
    print('--- per cell ---')
    print('%-7s %3s %3s %4s %6s %7s %6s %9s %9s %8s %8s'
          % ('src', 'v', 'L', 'rs', '|R_L|', 'n_gen', 'nP', 'x* A', 'x* B',
             'sd A', 'sd B'))
    for r in recs:
        f = lambda z: ('%.4f' % z) if np.isfinite(z) else '  -   '
        g = lambda z: ('%.2fx' % z) if np.isfinite(z) else '  -  '
        print('%-7s %3d %3d %4d %6d %7.0f %6d %9s %9s %8s %8s'
              % (r['tag'], r['v'], r['L'], r['rule_seed'], r['R_L'],
                 r['n_gen'], r['n_P'], f(r['x_A']), f(r['x_B']),
                 g(r['boot_sd_A']), g(r['boot_sd_B'])))

    def group(tag, v, L, which):
        out = []
        for (t, vv, LL, rs), (ra, rb) in reps.items():
            if t == tag and vv == v and LL == L:
                out.append(ra if which == 'A' else rb)
        return [o for o in out if np.isfinite(o).any()]

    def point(tag, v, L, which):
        xs = [r['x_A'] if which == 'A' else r['x_B'] for r in recs
              if r['tag'] == tag and r['v'] == v and r['L'] == L]
        xs = [x for x in xs if np.isfinite(x)]
        return (float(np.median(xs)), len(xs)) if xs else (float('nan'), 0)

    print()
    print('--- control: v=4 m=2 L=2, new settings vs published ---')
    print('%-4s %-28s %-28s %s' % ('est', 'new (8 seeds, n_gen 16384)',
                                   'published (n_gen 8192)', 'ratio new/pub'))
    for w in ('A', 'B'):
        pn, cn = point('L2new', 4, 2, w)
        pp, cp = point('L2pub', 4, 2, w)
        r, lo, hi = med_ratio_ci(group('L2new', 4, 2, w), group('L2pub', 4, 2, w),
                                 a.boot, rng)
        print('%-4s %-28s %-28s %.2fx  [%.2f, %.2f]'
              % (w, '%.4f  (%d cells)' % (pn, cn), '%.4f  (%d cells)' % (pp, cp),
                 r, lo, hi))

    print()
    print('--- paired depth test: x*(L=2) / x*(L=3) at fixed (v, m=2) ---')
    print('registered 2: if 2D behaves like 1D, x*(L=3) < x*(L=2), i.e. ratio > 1')
    print('%-4s %3s %-22s %-22s %s'
          % ('est', 'v', 'L=2', 'L=3', 'ratio L2/L3 [95% CI]'))
    verdict, counts = {}, {}
    for w in ('A', 'B'):
        for v in (2, 3):
            n2 = group('L2new', v, 2, w) or group('L2pub', v, 2, w)
            t2 = 'L2new' if group('L2new', v, 2, w) else 'L2pub'
            p2, c2 = point(t2, v, 2, w)
            p3, c3 = point('L3pub', v, 3, w)
            r, lo, hi = med_ratio_ci(n2, group('L3pub', v, 3, w), a.boot, rng)
            verdict[(w, v)] = (r, lo, hi)
            counts[(w, v)] = (c2, c3)
            print('%-4s %3d %-22s %-22s %.2fx  [%.2f, %.2f]%s'
                  % (w, v, '%.4f (%d)' % (p2, c2), '%.4f (%d)' % (p3, c3),
                     r, lo, hi,
                     '' if not np.isfinite(lo) else
                     ('   <- excludes 1' if (lo > 1 or hi < 1) else '   covers 1')))

    print()
    print('--- reading ---')
    # "Does the interval cover 1" is the wrong question on its own: with few
    # seeds nothing will ever exclude 1, and that is not evidence of stability.
    # The question that decides it is whether a 1D-SIZED effect is still
    # allowed, so the bar is the matched one-step 1D median, read from
    # depth1d_noise_1step.json when it is there.
    one_step = {'A': 2.52, 'B': 2.59}
    f1d = ROOT / a.new / 'depth1d_noise_1step.json'
    if f1d.exists():
        combos = json.load(open(f1d))['combos']
        for w in ('A', 'B'):
            rs = [c[w]['ratio'] for c in combos
                  if np.isfinite(c[w]['ratio']) and c[w]['ratio'] > 0]
            if rs:
                one_step[w] = float(np.median(rs))

    sig = [k for k, (r, lo, hi) in verdict.items()
           if np.isfinite(lo) and (lo > 1 or hi < 1)]
    up = [k for k in sig if verdict[k][0] > 1]
    excl = [k for k, (r, lo, hi) in verdict.items()
            if np.isfinite(hi) and hi < one_step[k[0]]]
    print('L=3 cells per (est, v): %s'
          % ',  '.join('%s v%d: %d' % (w, v, counts[(w, v)][1])
                       for (w, v) in sorted(counts)))
    print('A 1D-sized step (%.2fx A / %.2fx B) is excluded in %d of 4 pairs.'
          % (one_step['A'], one_step['B'], len(excl)))
    if len(excl) == 4 and not up:
        print('So x* does NOT fall with depth in 2D the way it does in 1D.')
        print('This is a demonstration, not a failure to resolve one, and')
        print('section A stands as written.')
    elif not sig:
        print('No depth effect resolvable, and the 1D size is not excluded')
        print('everywhere either, so this decides nothing on its own.')
    else:
        print('Resolved in %d of 4 (estimator, v) pairs; %d with x*(L=3) lower,'
              % (len(sig), len(up)))
        print('which is the 1D direction.  If this holds, section A\'s')
        print('P* = 0.21 |R_L| is depth-dependent and needs restating.')

    if a.out:
        p = ROOT / a.out
        p.parent.mkdir(parents=True, exist_ok=True)
        json.dump(dict(cells=recs,
                       verdict={'%s_v%d' % k: v for k, v in verdict.items()}),
                  open(p, 'w'), indent=1)
        print('\nwrote %s' % p)


if __name__ == '__main__':
    main()
