"""The fair test: does the entropy floor earn its keep when a schedule has room?

Every earlier scoring of hshunt's patch-size criterion used ``s = 2``, where
``U1 = 2 s^(L-1) - 1`` equals the grid side minus one, so the only odd window in
``[U1, side]`` is ``U1`` itself and "largest k above a floor" was choosing
between one real rung and the whole grid.  ``s = 3, L = 2`` is a 9x9 grid with
``U1 = 5`` and three genuine rungs, 5 / 7 / 9.

Three questions, in the order they decide the verdict:

1. Where does the *fixed* optimum sit -- at U1, above it, or at the whole grid?
   At s = 2 it was always U1, which is what made the criterion redundant.
2. Does the adaptive machine beat the best fixed window, per cell, on the shared
   ladder?  This is the registered prediction in ``rhm_adaptive_s3.sbatch``.
3. Does the induced ``k(t)`` actually use the extra rungs, or does it collapse
   to two values again?  A schedule that only ever picks 9 then 5 has not used
   the room, and a win would then be about the choice set, not the schedule.

Reads whatever ``s3_*_L2_v*_m*.json`` files exist, so it is safe to run against
a sweep that is still filling in.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CELL = ('v', 's', 'L', 'm', 'rule_seed')


def load(d):
    rows = []
    for f in sorted((ROOT / d).glob('s3_*.json')):
        try:
            rs = json.load(open(f))
        except (ValueError, IOError):
            continue
        if isinstance(rs, list):
            rows += [r for r in rs if isinstance(r, dict) and 'success' in r
                     and not r.get('skipped') and all(c in r for c in CELL)]
    return rows


def best(rs):
    return max(rs, key=lambda r: r['success']) if rs else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', default='results/rhm_sampling/2026-11-04')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    rows = load(args.dir)
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[tuple(r[c] for c in CELL)]['adapt' if r['k'] == 'adapt' else r['k']].append(r)

    cells = sorted(c for c in by if 'adapt' in by[c] and any(isinstance(k, int) for k in by[c]))
    if not cells:
        print('no scored cells yet in %s' % args.dir)
        return
    U1 = 2 * cells[0][1] ** (cells[0][2] - 1) - 1
    print('%d cells, U1 = %d, rungs %s\n'
          % (len(cells), U1, sorted(k for k in by[cells[0]] if isinstance(k, int))))

    print('=== peak success on the shared ladder ===')
    print('  v  m rs | best fixed k   succ | adapt h_min  k_mean   succ | adapt - fixed')
    table, w_fix, n_bind, at_u1 = [], 0, 0, 0
    for c in cells:
        fixed = {k: v for k, v in by[c].items() if isinstance(k, int)}
        # one ladder for every arm, so no peak is read off a rung the others
        # never visited -- the adaptive arm's ladder is the reference
        Ps = set(r['P'] for r in by[c]['adapt'])
        for v in fixed.values():
            Ps &= set(r['P'] for r in v)
        if not Ps:
            continue
        fb = {k: best([r for r in v if r['P'] in Ps]) for k, v in fixed.items()}
        f = max((r for r in fb.values() if r), key=lambda r: r['success'])
        a = best([r for r in by[c]['adapt'] if r['P'] in Ps])
        ctl = best([r for r in by[c]['adapt'] if r['P'] in Ps and r['h_min'] == 0])
        w_fix += a['success'] > f['success']
        n_bind += a['h_min'] > 0
        at_u1 += f['k'] == U1
        print('  %d %2d %2d | %11d %6.4f | %11.1f %7.2f %6.4f | %+.4f'
              % (c[0], c[3], c[4], f['k'], f['success'], a['h_min'], a['k_mean'],
                 a['success'], a['success'] - f['success']))
        row = dict(zip(CELL, c))
        row.update(fixed_k=f['k'], fixed_succ=f['success'], fixed_P=f['P'],
                   fixed_gram=f['grammatical'], fixed_novel=f['novel'],
                   adapt_succ=a['success'], adapt_P=a['P'], adapt_hmin=a['h_min'],
                   adapt_kmean=a['k_mean'], adapt_gram=a['grammatical'],
                   adapt_novel=a['novel'], adapt_ksched=a['k_schedule'],
                   ctl_succ=None if ctl is None else ctl['success'],
                   per_k={str(k): (r['success'] if r else None) for k, r in fb.items()})
        table.append(row)

    n = len(table)
    srt = lambda xs: sorted(xs)[len(xs) // 2] if xs else float('nan')
    print('\n=== verdict ===')
    print('  adaptive beats the best fixed k: %d / %d cells  (median ratio %.3f)'
          % (w_fix, n, srt([r['adapt_succ'] / r['fixed_succ'] for r in table if r['fixed_succ']])))
    print('  its best floor binds (h_min>0):  %d / %d' % (n_bind, n))
    print('  the fixed optimum sits at U1:    %d / %d  (at s=2 this was every cell)' % (at_u1, n))
    print('  median delta gram %+.3f, delta novel %+.3f'
          % (srt([r['adapt_gram'] - r['fixed_gram'] for r in table]),
             srt([r['adapt_novel'] - r['fixed_novel'] for r in table])))

    print('\n=== does the schedule use the extra rungs? ===')
    print('  k(t) at each cell\'s best floor, first reveal -> last:')
    for r in table:
        s = sorted(r['adapt_ksched'], reverse=True)          # masked 1.0 first
        ks = [k for _, k in s]
        print('    v%d m%d rs%d  h=%.1f  %s   (span %.2f)'
              % (r['v'], r['m'], r['rule_seed'], r['adapt_hmin'],
                 ' '.join('%.2f' % k for k in ks), max(ks) - min(ks)))

    if args.out:
        o = ROOT / args.out
        o.parent.mkdir(parents=True, exist_ok=True)
        json.dump(table, open(o, 'w'), indent=1)
        print('\nwrote %s (%d rows)' % (o, n))


if __name__ == '__main__':
    main()
