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


def load_curves(specs, RL, P_min=0.0):
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
        by_P = defaultdict(lambda: [0.0, 0.0])
        for r in rows:
            n = float(r.get('n_gen', 4096))
            by_P[r['P']][0] += r['success'] * n
            by_P[r['P']][1] += n
        Ps = np.array(sorted(by_P), float)
        hits = np.array([by_P[p][0] for p in sorted(by_P)], float)
        ns = np.array([by_P[p][1] for p in sorted(by_P)], float)
        keep = Ps >= P_min
        if keep.sum() < 3:
            continue
        out[key] = (Ps[keep], (hits / ns)[keep], ns[keep])
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
                         'like-for-like comparison. Set 0 to see it.')
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
    ], RL, a.P_min)
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

    print('cells: %d   (m=2 only, k=U1, block_align=1)' % len(recs))
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
    verdict = {}
    for w in ('A', 'B'):
        for v in (2, 3):
            n2 = group('L2new', v, 2, w) or group('L2pub', v, 2, w)
            t2 = 'L2new' if group('L2new', v, 2, w) else 'L2pub'
            p2, c2 = point(t2, v, 2, w)
            p3, c3 = point('L3pub', v, 3, w)
            r, lo, hi = med_ratio_ci(n2, group('L3pub', v, 3, w), a.boot, rng)
            verdict[(w, v)] = (r, lo, hi)
            print('%-4s %3d %-22s %-22s %.2fx  [%.2f, %.2f]%s'
                  % (w, v, '%.4f (%d)' % (p2, c2), '%.4f (%d)' % (p3, c3),
                     r, lo, hi,
                     '' if not np.isfinite(lo) else
                     ('   <- excludes 1' if (lo > 1 or hi < 1) else '   covers 1')))

    print()
    print('--- reading ---')
    sig = [k for k, (r, lo, hi) in verdict.items()
           if np.isfinite(lo) and (lo > 1 or hi < 1)]
    if not sig:
        print('No depth effect resolvable at this power.  With the L=3 side at')
        print('3 seeds and n_gen=1024 (per-cell spread up to 3.5x), only an')
        print('effect of roughly the 1D size (~2.8x) would clear these bars, so')
        print('this is NOT a demonstration that x* is depth-stable in 2D.')
    else:
        up = [k for k in sig if verdict[k][0] > 1]
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
