"""Score the adaptive-window LS machine on novel-and-grammatical, not grammaticality.

``AdaptiveWindowDiscreteScoreMachine`` picks ``k`` per site and per step as the
largest window whose posterior entropy is still above a floor ``h_min`` --
hshunt's criterion.  The 2026-11-03 sweep (job 50903583) measured it, but every
reading of that sweep since has quoted ``grammatical`` alone, which is the one
number the criterion is *not* about: a machine that copies its training set
scores 1.0 on grammaticality and 0.0 on the thing we care about.  The driver
recorded ``novel`` and ``success`` all along, so this is a re-read of existing
rows, not a new sweep.

The comparison has to be paired or it says nothing.  Fixed-k and adaptive rows
are matched on the grammar cell ``(v, s, L, m, rule_seed)`` *and* on P, because
realized ``|R_L|`` varies by orders of magnitude between rule seeds and the two
sweeps do not share a full ladder.  Only P values present on both sides are
compared, and the script prints how many that is so a thin overlap is visible
rather than hidden inside an average.

The headline is peak success over the shared ladder, since that is the quantity
``P*`` is read off and the one the whole project measures.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CELL = ('v', 's', 'L', 'm', 'rule_seed')


def load_rows(dirs):
    """Every row carrying a success score, tagged fixed or adaptive."""
    fixed, adapt, files = [], [], []
    for d in dirs:
        for f in sorted((ROOT / d).glob('*.json')):
            try:
                rows = json.load(open(f))
            except (ValueError, IOError):
                continue
            if not isinstance(rows, list):
                continue
            n_f = n_a = 0
            for r in rows:
                if not isinstance(r, dict) or 'success' not in r or 'k' not in r:
                    continue
                if any(c not in r for c in CELL):
                    continue
                r['_src'] = f.name
                if r['k'] == 'adapt':
                    adapt.append(r)
                    n_a += 1
                else:
                    fixed.append(r)
                    n_f += 1
            if n_f or n_a:
                files.append((f.name, n_f, n_a))
    return fixed, adapt, files


def cell_of(r):
    return tuple(r[c] for c in CELL)


def best(rows, key='success'):
    return max(rows, key=lambda r: r[key]) if rows else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dirs', nargs='+',
                    default=['results/rhm_sampling/2026-11-02',
                             'results/rhm_sampling/2026-11-03'])
    ap.add_argument('--out', default=None, help='write the paired table as JSON')
    args = ap.parse_args()

    fixed, adapt, files = load_rows(args.dirs)
    print('=== inventory ===')
    for name, n_f, n_a in files:
        print('  %-28s fixed %4d  adapt %4d' % (name, n_f, n_a))
    print('  total: %d fixed rows, %d adaptive rows' % (len(fixed), len(adapt)))
    print('  fixed k values:   %s' % sorted(set(r['k'] for r in fixed)))
    print('  adaptive k sets:  %s' % sorted(set(tuple(r['k_set']) for r in adapt)))
    print('  adaptive floors:  %s' % sorted(set((r['h_mode'], r['h_min']) for r in adapt)))

    f_by = defaultdict(list)
    for r in fixed:
        f_by[cell_of(r)].append(r)
    a_by = defaultdict(list)
    for r in adapt:
        a_by[cell_of(r)].append(r)

    shared = sorted(set(f_by) & set(a_by))
    print('\n=== cells ===')
    print('  adaptive cells %d, fixed cells %d, both %d'
          % (len(a_by), len(f_by), len(shared)))
    if not shared:
        print('  no paired cells; nothing to compare')
        return

    # Per cell: the best fixed k and the best adaptive floor, both restricted to
    # the P values the two sweeps actually share, so the peak is read off one
    # ladder.  A cell whose overlap is a single rung cannot bracket a peak and
    # is reported but flagged.
    table = []
    for c in shared:
        fr, ar = f_by[c], a_by[c]
        Ps = sorted(set(r['P'] for r in fr) & set(r['P'] for r in ar))
        if not Ps:
            continue
        fr = [r for r in fr if r['P'] in Ps]
        ar = [r for r in ar if r['P'] in Ps]
        bf, ba = best(fr), best(ar)
        # best adaptive among floors that actually bind, i.e. h_min > 0; h_min=0
        # is the control ("largest window that matched anything") and answering
        # whether the *floor* helps means comparing against it too
        ctl = best([r for r in ar if r['h_min'] == 0.0])
        flo = best([r for r in ar if r['h_min'] > 0.0])
        row = dict(zip(CELL, c))
        row.update(
            n_P=len(Ps), P_lo=min(Ps), P_hi=max(Ps),
            fixed_k=bf['k'], fixed_P=bf['P'], fixed_succ=bf['success'],
            fixed_gram=bf['grammatical'], fixed_novel=bf['novel'],
            adapt_succ=ba['success'], adapt_P=ba['P'],
            adapt_hmode=ba['h_mode'], adapt_hmin=ba['h_min'],
            adapt_gram=ba['grammatical'], adapt_novel=ba['novel'],
            adapt_kmean=ba.get('k_mean'),
            ctl_succ=ctl['success'] if ctl else None,
            floor_succ=flo['success'] if flo else None,
            floor_hmin=flo['h_min'] if flo else None,
            floor_hmode=flo['h_mode'] if flo else None,
        )
        table.append(row)

    print('\n=== paired peak success on the shared ladder ===')
    hdr = ('  v s L m rs  nP   best fixed            best adaptive                  '
           'floor vs control')
    print(hdr)
    for r in table:
        print('  %d %d %d %d %2d  %2d   k=%-2s succ %.4f      %s h>=%.1f succ %.4f '
              '(k~%.2f)   %s'
              % (r['v'], r['s'], r['L'], r['m'], r['rule_seed'], r['n_P'],
                 r['fixed_k'], r['fixed_succ'], r['adapt_hmode'], r['adapt_hmin'],
                 r['adapt_succ'],
                 r['adapt_kmean'] if r['adapt_kmean'] is not None else float('nan'),
                 ('%.4f vs %.4f' % (r['floor_succ'], r['ctl_succ']))
                 if r['floor_succ'] is not None and r['ctl_succ'] is not None
                 else 'n/a'))

    n = len(table)
    wins = sum(1 for r in table if r['adapt_succ'] > r['fixed_succ'])
    fwins = sum(1 for r in table
                if r['floor_succ'] is not None and r['ctl_succ'] is not None
                and r['floor_succ'] > r['ctl_succ'])
    nf = sum(1 for r in table if r['floor_succ'] is not None and r['ctl_succ'] is not None)
    print('\n=== verdict ===')
    print('  adaptive beats best fixed k:  %d / %d cells' % (wins, n))
    print('  a binding floor beats h_min=0: %d / %d cells' % (fwins, nf))
    if n:
        ms = sorted(r['adapt_succ'] - r['fixed_succ'] for r in table)
        print('  median succ delta (adapt - fixed): %+.4f  [min %+.4f, max %+.4f]'
              % (ms[n // 2], ms[0], ms[-1]))
    nz = [r for r in table if r['fixed_succ'] > 0]
    if nz:
        rr = sorted(r['adapt_succ'] / r['fixed_succ'] for r in nz)
        print('  median succ ratio over %d cells with nonzero fixed: %.3f'
              % (len(nz), rr[len(nz) // 2]))
    # Which side of the ledger moved: a machine can gain success by becoming
    # more grammatical or by copying less, and the floor is supposed to do the
    # second.  Reported separately because they have opposite mechanisms.
    dg = sorted(r['adapt_gram'] - r['fixed_gram'] for r in table)
    dn = sorted(r['adapt_novel'] - r['fixed_novel'] for r in table)
    print('  median delta grammatical %+.4f ; median delta novel %+.4f'
          % (dg[n // 2], dn[n // 2]))

    if args.out:
        out = ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(table, open(out, 'w'), indent=1)
        print('\nwrote %s (%d rows)' % (out, len(table)))


if __name__ == '__main__':
    main()
