"""Does restricting the adaptive window to ``k >= U1`` rescue the entropy floor?

Three machines on the same grammars, ladder and seeds:

``fixed k=3``
    the best fixed window at L=2, which is exactly ``U1 = 2 s^(L-1) - 1``.
``adapt {1,3,5}``
    the 2026-11-03 machine (50903583).  Its floor can select windows below U1,
    which cannot see the root and so cannot be grammatical at any P.
``adapt {3,5}``
    the 2026-11-04 machine (51098252), one variable changed: k=1 removed.

The prediction on record in ``rhm_adaptive_u1.sbatch`` is that the floor's
novelty gain survives the restriction while the grammaticality collapse does
not, so ``{3,5}`` with a binding floor beats both ``h_min=0`` and fixed k=3.
The null stated alongside it is that ``{3,5}`` merely reproduces fixed k=3, in
which case U1 already *is* the schedule and the criterion adds nothing at L=2.

Peaks are read per cell over the shared ladder, which is the same quantity
``P*`` comes from.  Grammaticality and novelty are carried along separately
because the two machines can reach the same success from opposite directions.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CELL = ('v', 's', 'L', 'm', 'rule_seed')


def load(dirs):
    rows = []
    for d in dirs:
        for f in sorted((ROOT / d).glob('*.json')):
            try:
                rs = json.load(open(f))
            except (ValueError, IOError):
                continue
            if not isinstance(rs, list):
                continue
            for r in rs:
                if (not isinstance(r, dict) or 'success' not in r or 'k' not in r
                        or any(c not in r for c in CELL)):
                    continue
                rows.append(r)
    return rows


def arm(r):
    """Which of the three machines a row belongs to, or None."""
    if r['k'] == 'adapt':
        ks = tuple(r['k_set'])
        if ks == (1, 3, 5):
            return 'a135'
        if ks == (3, 5):
            return 'a35'
        return None
    return 'f3' if r['k'] == 3 else None


def best(rs):
    return max(rs, key=lambda r: r['success']) if rs else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dirs', nargs='+',
                    default=['results/rhm_sampling/2026-11-02',
                             'results/rhm_sampling/2026-11-03',
                             'results/rhm_sampling/2026-11-04'])
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    by = defaultdict(lambda: defaultdict(list))
    for r in load(args.dirs):
        a = arm(r)
        if a and r['L'] == 2:
            by[tuple(r[c] for c in CELL)][a].append(r)

    cells = sorted(c for c in by if {'f3', 'a135', 'a35'} <= set(by[c]))
    print('cells with all three arms at L=2: %d\n' % len(cells))
    if not cells:
        print('nothing to compare')
        return

    print('=== peak success on the shared ladder ===')
    print('  v s L m rs   fixed k=3   adapt{1,3,5}   adapt{3,5}   {3,5} vs k=3   floor used')
    table, w_f3, w_a135, n_bind = [], 0, 0, 0
    for c in cells:
        # one ladder for all three arms, so no peak is read off a rung the
        # others never visited
        Ps = set.intersection(*(set(r['P'] for r in by[c][a])
                                for a in ('f3', 'a135', 'a35')))
        if not Ps:
            continue
        g = {a: best([r for r in by[c][a] if r['P'] in Ps]) for a in ('f3', 'a135', 'a35')}
        d = g['a35']['success'] - g['f3']['success']
        w_f3 += d > 0
        w_a135 += g['a35']['success'] > g['a135']['success']
        n_bind += g['a35']['h_min'] > 0
        print('  %d %d %d %d %2d     %.4f        %.4f       %.4f       %+.4f       h>=%.1f'
              % (c[0], c[1], c[2], c[3], c[4], g['f3']['success'],
                 g['a135']['success'], g['a35']['success'], d, g['a35']['h_min']))
        row = dict(zip(CELL, c))
        for a in ('f3', 'a135', 'a35'):
            row.update({a + '_succ': g[a]['success'], a + '_P': g[a]['P'],
                        a + '_gram': g[a]['grammatical'], a + '_novel': g[a]['novel']})
        row['a35_hmin'] = g['a35']['h_min']
        row['a35_kmean'] = g['a35'].get('k_mean')
        table.append(row)

    n = len(table)
    print('\n=== verdict ===')
    print('  adapt{3,5} beats fixed k=3:      %d / %d cells' % (w_f3, n))
    print('  adapt{3,5} beats adapt{1,3,5}:   %d / %d cells' % (w_a135, n))
    print('  its best floor binds (h_min>0):  %d / %d cells' % (n_bind, n))
    for lo, hi, lbl in (('f3', 'a35', '{3,5} - k=3'), ('a135', 'a35', '{3,5} - {1,3,5}')):
        ds = sorted(r[hi + '_succ'] - r[lo + '_succ'] for r in table)
        rs = sorted(r[hi + '_succ'] / r[lo + '_succ']
                    for r in table if r[lo + '_succ'] > 0)
        print('  median %-16s %+.4f   (median ratio %.3f over %d cells)'
              % (lbl, ds[n // 2], rs[len(rs) // 2] if rs else float('nan'), len(rs)))
    dg = sorted(r['a35_gram'] - r['f3_gram'] for r in table)
    dn = sorted(r['a35_novel'] - r['f3_novel'] for r in table)
    print('  vs k=3 at their peaks: median delta gram %+.4f, delta novel %+.4f'
          % (dg[n // 2], dn[n // 2]))

    print('\n=== the floor sweep on {3,5}, one cell (v2 m3 rs0) ===')
    c0 = (2, 2, 2, 3, 0)
    if c0 in by:
        print('  h_min      P    succ    gram   novel   k_mean')
        for h in sorted(set(r['h_min'] for r in by[c0]['a35'])):
            r = best([q for q in by[c0]['a35'] if q['h_min'] == h])
            print('  %5.1f %6d  %.4f  %.4f  %.4f   %.3f'
                  % (h, r['P'], r['success'], r['grammatical'], r['novel'],
                     r.get('k_mean', float('nan'))))

    if args.out:
        o = ROOT / args.out
        o.parent.mkdir(parents=True, exist_ok=True)
        json.dump(table, open(o, 'w'), indent=1)
        print('\nwrote %s (%d rows)' % (o, len(table)))


if __name__ == '__main__':
    main()
