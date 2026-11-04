"""Why the adaptive window loses: the floor's two effects, separated.

``rhm_adaptive_score.py`` establishes that the adaptive machine scores below the
best fixed ``k`` on novel-and-grammatical in 20 of 21 paired cells.  That is a
verdict, not a mechanism.  Two things could produce it:

1. the *selection* is bad -- the chosen ``k`` is simply worse than the best
   fixed one, and adaptivity adds nothing over the fixed window it imitates; or
2. the *adaptivity* is bad -- varying ``k`` across sites and steps costs
   something a fixed window of the same mean size does not pay.

They are told apart by comparing the adaptive machine against the fixed ``k``
it actually selects, on the same cell and the same P.  This script does that,
and prints the induced ``k(t)`` schedule, which is the object hshunt's criterion
is really about: the schedule is recorded per masked fraction, so it can be read
in trajectory order (masked 1.0 first, 0.0 last) rather than assumed.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CELL = ('v', 's', 'L', 'm', 'rule_seed')


def load(dirs):
    fixed, adapt = [], []
    for d in dirs:
        for f in sorted((ROOT / d).glob('*.json')):
            try:
                rows = json.load(open(f))
            except (ValueError, IOError):
                continue
            if not isinstance(rows, list):
                continue
            for r in rows:
                if (not isinstance(r, dict) or 'success' not in r
                        or 'k' not in r or any(c not in r for c in CELL)):
                    continue
                (adapt if r['k'] == 'adapt' else fixed).append(r)
    return fixed, adapt


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dirs', nargs='+',
                    default=['results/rhm_sampling/2026-11-02',
                             'results/rhm_sampling/2026-11-03'])
    args = ap.parse_args()
    fixed, adapt = load(args.dirs)

    # index fixed by (cell, k, P) so a comparison can be exact rather than
    # interpolated: an adaptive row is only ever compared with a fixed row at
    # the identical grammar, window and training-set size
    fx = {}
    for r in fixed:
        fx[(tuple(r[c] for c in CELL), r['k'], r['P'])] = r
    ks_avail = sorted(set(k for (_, k, _) in fx))
    print('fixed k available: %s\n' % ks_avail)

    print('=== adaptive (h_min=0) against the fixed k it imitates, same cell and P ===')
    print('  the adaptive machine picks k~4.5 of a possible {1,3,5}, so k=5 is the')
    print('  like-for-like window and k=3 is the best fixed one at L=2 (= U1).\n')
    print('  v s L m rs      P   adapt   fix k=3   fix k=5   adapt-vs-k5')
    rows = defaultdict(list)
    for r in adapt:
        if r['h_min'] != 0.0 or r['h_mode'] != 'match':
            continue
        c = tuple(r[x] for x in CELL)
        f3 = fx.get((c, 3, r['P']))
        f5 = fx.get((c, 5, r['P']))
        if f3 is None and f5 is None:
            continue
        rows[c].append((r, f3, f5))
    n_cmp = n_win = 0
    deltas = []
    for c in sorted(rows):
        # the P at which the adaptive machine does best on this cell, which is
        # the only rung its own P* would be read from
        r, f3, f5 = max(rows[c], key=lambda t: t[0]['success'])
        s3 = '%.4f' % f3['success'] if f3 else '   -  '
        s5 = '%.4f' % f5['success'] if f5 else '   -  '
        d = ''
        if f5:
            n_cmp += 1
            delta = r['success'] - f5['success']
            deltas.append(delta)
            n_win += delta > 0
            d = '%+.4f' % delta
        print('  %d %d %d %d %2d %6d  %.4f    %s    %s    %s'
              % (c[0], c[1], c[2], c[3], c[4], r['P'], r['success'], s3, s5, d))
    if n_cmp:
        deltas.sort()
        print('\n  adaptive beats fixed k=5 at matched P: %d / %d cells '
              '(median delta %+.4f)' % (n_win, n_cmp, deltas[n_cmp // 2]))

    print('\n=== induced k(t), in trajectory order (masked 1.0 -> 0.0) ===')
    print('  one L=2 cell per floor, h_mode=match, v2 m3 rs0, at its best P\n')
    sel = [r for r in adapt
           if r['h_mode'] == 'match' and (r['v'], r['m'], r['L'], r['rule_seed']) == (2, 3, 2, 0)]
    by_floor = defaultdict(list)
    for r in sel:
        by_floor[r['h_min']].append(r)
    print('  h_min    P    succ    gram   novel   ' +
          '  '.join('%.1f' % f for f, _ in sorted(sel[0]['k_schedule'], reverse=True)))
    for h in sorted(by_floor):
        r = max(by_floor[h], key=lambda q: q['success'])
        sched = sorted(r['k_schedule'], reverse=True)
        print('  %4.1f  %6d  %.4f  %.4f  %.4f   %s'
              % (h, r['P'], r['success'], r['grammatical'], r['novel'],
                 '  '.join('%.2f' % kk for _, kk in sched)))
    print('\n  (columns are masked fraction: 1.0 is the first reveal, 0.1 the last)')


if __name__ == '__main__':
    main()
