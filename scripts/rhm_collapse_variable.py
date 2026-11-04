"""Which quantity does P* actually track -- |R_L|, or the window table's size?

The depth probes left a gap.  ``P* = 0.21 |R_L|`` holds in 2D with a free
exponent of 0.965 +- 0.041, but in 1D the same fit gives 0.689 +- 0.045 and the
ratio x* = P*/|R_L| is 0.050 against 2D's 0.193 at matched n=16.  So |R_L| is
not the whole story, and the 0.21 had to be scoped to 2D.

|R_L| counts whole samples.  The machine never sees a whole sample: it sees
windows, ``n`` of them per training sample, drawn from the ``N_k`` distinct
windows the grammar admits at window size k.  Its table is full when

    P * n  ~  N_k        i.e.   P_sat ~ N_k / n

and nothing it can do improves after that.  So the natural candidate is

    y* = P* n / N_k

which, unlike x*, knows about the window geometry -- and the window geometry is
the one thing that genuinely differs between 1D and 2D at matched n.  If y* is
common to both dimensions where x* is 3.9x apart, the law generalises and the
2D-specific 0.21 becomes a consequence rather than a brute constant.

Nothing here needs a new run: every 1D sweep row already records
``n_distinct_patches`` at each P, and the 2D side has exact saturation curves
from ``rhm_window_count_exact``.  N_k is read as the plateau of those curves,
and a cell whose curve has not plateaued is reported and excluded rather than
being given a lower bound dressed up as a value.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def geo(x):
    """Median and geometric sd; ratios are log-normal-ish so means mislead."""
    x = np.asarray(x, float)
    return float(np.median(x)), (float(np.exp(np.log(x).std(ddof=1))) if len(x) > 1 else 1.0)


def saturated(curve, tol=0.02):
    """Plateau value of a (P, n_distinct) curve, or None if it is still climbing.

    ``tol`` is the fractional gain allowed over the last doubling of P.  A table
    that is still growing gives a lower bound on N_k, not N_k, and a lower bound
    would bias y* upward by an unknown factor -- so those cells are dropped.
    """
    if len(curve) < 2:
        return None
    (_, a), (_, b) = curve[-2], curve[-1]
    return float(b) if b > 0 and (b - a) / b <= tol else None


def nk_1d(sweep_rows):
    """N_k per 1D cell, from the top of its own P ladder."""
    by = {}
    for r in sweep_rows:
        key = (r['v'], r['m'], r['s'], r['L'], r['rule_seed'])
        by.setdefault(key, []).append((r['P'], r['n_distinct_patches']))
    return {k: saturated(sorted(v)) for k, v in by.items()}


def nk_2d(exact_paths, curve_paths):
    """N_k per 2D (cell, k), exact where it exists and sampled only as a fallback.

    ``rhm_window_count_exact`` enumerates the whole grammar by odometer, so its
    N_k is the true count and is keyed by k as well as by cell -- which matters
    here because the peak's k differs by depth (3 at L=2, 7 at L=3).  The
    sampled saturation curves cover far fewer cells and only bound N_k from
    below, so they are used only where no exact count exists, and flagged.
    """
    out = {}
    for p in exact_paths:
        if not p.exists():
            continue
        for r in json.load(open(p)):
            out[(r['v'], r['m'], r['s'], r['L'], r['rule_seed'], r['k'])] = (float(r['N_k']), True)
    for p in curve_paths:
        if not p.exists():
            continue
        for r in json.load(open(p)):
            key = (r['v'], r['m'], r['s'], r['L'], r['rule_seed'], r['k'])
            sat = saturated(r['curve'])
            if key not in out and sat:
                out[key] = (sat, False)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--peaks_1d', default='results/rhm_1d/2026-11-03/peaks.json')
    ap.add_argument('--sweep_1d', default='results/rhm_1d/2026-11-03/sweep.json')
    ap.add_argument('--peaks_2d', default='results/rhm_sampling/2026-11-03/peak_summary.json')
    ap.add_argument('--exact_2d', nargs='+',
                    default=['results/rhm_sampling/2026-11-03/nk_exact_L2.json',
                             'results/rhm_sampling/2026-11-03/nk_exact_L3.json'])
    ap.add_argument('--counts_2d', nargs='+',
                    default=['results/rhm_sampling/2026-11-03/counts_u1_L2.json',
                             'results/rhm_sampling/2026-11-03/counts_u1_L3.json'])
    ap.add_argument('--out', default='results/rhm_1d/2026-11-04/collapse_variable.json')
    args = ap.parse_args()

    N1 = nk_1d(json.load(open(ROOT / args.sweep_1d)))
    N2 = nk_2d([ROOT / p for p in args.exact_2d], [ROOT / p for p in args.counts_2d])

    keep = ('v', 'm', 's', 'L', 'rule_seed', 'k', 'P_star', 'R_L', 'ratio', 'succ_max')
    pts = []
    for p in json.load(open(ROOT / args.peaks_1d)):
        if not p['bracketed']:
            continue
        key = (p['v'], p['m'], p['s'], p['L'], p['rule_seed'])
        pts.append(dict(dim=1, n=p['s'] ** p['L'], N_k=N1.get(key), exact=False,
                        **{q: p[q] for q in keep}))
    for p in json.load(open(ROOT / args.peaks_2d)):
        if not p['bracketed']:
            continue
        nk, ex = N2.get((p['v'], p['m'], p['s'], p['L'], p['rule_seed'], p['k']), (None, False))
        pts.append(dict(dim=2, n=p['s'] ** (2 * p['L']), N_k=nk, exact=ex,
                        **{q: p[q] for q in keep}))

    have = [q for q in pts if q['N_k']]
    print('%d bracketed peaks; %d have a saturated N_k (1D %d, 2D %d, of which %d exact)'
          % (len(pts), len(have), sum(q['dim'] == 1 for q in have),
             sum(q['dim'] == 2 for q in have), sum(q['exact'] for q in have)))
    miss2 = [q for q in pts if q['dim'] == 2 and not q['N_k']]
    if miss2:
        print('  2D cells without a saturated count (excluded): %s'
              % sorted({(q['v'], q['m'], q['L']) for q in miss2}))
    if not any(q['dim'] == 2 for q in have):
        print('NO 2D cells matched -- the comparison cannot be made; stopping.')
        return

    for q in have:
        q['x_star'] = q['P_star'] / q['R_L']            # the current 2D law
        q['y_star'] = q['P_star'] * q['n'] / q['N_k']   # the table-saturation law
        q['Nk_over_RL'] = q['N_k'] / q['R_L']

    print('\n--- 1. the two candidate ratios, by dimension and depth ---')
    print('%-4s %-5s %-6s %-11s %-9s %-11s %-9s %s'
          % ('dim', 'n', 'cells', 'med x*', 'geo sd', 'med y*', 'geo sd', 'med N_k/|R_L|'))
    groups = sorted({(q['dim'], q['n']) for q in have})
    for dim, n in groups:
        sel = [q for q in have if q['dim'] == dim and q['n'] == n]
        mx, sx = geo([q['x_star'] for q in sel])
        my, sy = geo([q['y_star'] for q in sel])
        mr, _ = geo([q['Nk_over_RL'] for q in sel])
        print('%-4dD %-5d %-6d %-11.4f %-9.2fx %-11.4f %-9.2fx %.3f'
              % (dim, n, len(sel), mx, sx, my, sy, mr))

    print('\n--- 2. does y* collapse the dimensions where x* does not? ---')
    # the whole point: x* is 3.9x apart at matched n.  If y* is not, the window
    # table is the right bookkeeping and |R_L| was a 2D-only proxy for it.
    for lab, fld in (('x* = P*/|R_L|', 'x_star'), ('y* = P*n/N_k', 'y_star')):
        a = [q[fld] for q in have if q['dim'] == 1]
        b = [q[fld] for q in have if q['dim'] == 2]
        ma, sa = geo(a)
        mb, sb = geo(b)
        allm, alls = geo(a + b)
        print('  %-16s 1D %.4f (%.2fx, n=%d)   2D %.4f (%.2fx, n=%d)   '
              'gap %.2fx   pooled sd %.2fx'
              % (lab, ma, sa, len(a), mb, sb, len(b), max(ma, mb) / min(ma, mb), alls))

    # matched n is the cleanest single comparison, free of any depth trend
    m1 = [q for q in have if q['dim'] == 1 and q['n'] == 16]
    m2 = [q for q in have if q['dim'] == 2 and q['n'] == 16]
    if m1 and m2:
        print('\n  at matched n=16 (%d 1D cells, %d 2D):' % (len(m1), len(m2)))
        for lab, fld in (('x*', 'x_star'), ('y*', 'y_star')):
            a, b = np.median([q[fld] for q in m1]), np.median([q[fld] for q in m2])
            print('    %-3s 1D %.4f  vs  2D %.4f   -> %.2fx apart'
                  % (lab, a, b, max(a, b) / min(a, b)))

    print('\n--- 3. free fit P* ~ N_k^c, pooled over both dimensions ---')
    # if the saturation reading is right this exponent is 1 and the intercept
    # is ~1/n; a c far from 1 means N_k is no better a bookkeeping than |R_L|
    lN = np.log([q['N_k'] / q['n'] for q in have])
    lP = np.log([q['P_star'] for q in have])
    c, a0 = np.polyfit(lN, lP, 1)
    res = lP - (c * lN + a0)
    se = float(np.sqrt((res ** 2).sum() / (len(lN) - 2) / ((lN - lN.mean()) ** 2).sum()))
    print('  P* ~ (N_k/n)^%.3f +- %.3f  (1 is %.1f sd away),  prefactor %.3f'
          % (c, se, abs(c - 1) / se, np.exp(a0)))
    lR = np.log([q['R_L'] for q in have])
    b, b0 = np.polyfit(lR, lP, 1)
    resb = lP - (b * lR + b0)
    seb = float(np.sqrt((resb ** 2).sum() / (len(lR) - 2) / ((lR - lR.mean()) ** 2).sum()))
    print('  P* ~ |R_L|^%.3f +- %.3f  (1 is %.1f sd away)   [same cells, for contrast]'
          % (b, seb, abs(b - 1) / seb))
    print('  residual scatter: %.2fx about the N_k fit, %.2fx about the |R_L| fit'
          % (np.exp(res.std(ddof=1)), np.exp(resb.std(ddof=1))))

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(have, open(out, 'w'), indent=1)
    print('\nwrote %s (%d cells)' % (out, len(have)))


if __name__ == '__main__':
    main()
