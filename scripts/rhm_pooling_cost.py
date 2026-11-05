"""Does pooling the phases cost the machine its *support*, or its *sampler*?

``PatchCountDiscreteScoreMachine`` can hold one patch table for the whole grid
(``block_align=1``, pooled: every window placement votes into the same table, so
the machine is translation invariant) or one table per offset mod ``s``
(``block_align=2``, per-phase: a window at phase ``p`` is only ever matched
against windows seen at phase ``p``).  Pooling is strictly weaker as a
constraint -- the pooled table is the union of the phase tables, so any grid the
pooled machine can emit without leaving its table, the per-phase machine can
too, and usually more.

Two very different things could explain why the per-phase sampler scores higher:

*Support.*  Pooling admits more non-grammatical grids into the set the machine
cannot rule out.  ``rhm_straddle.py`` measures that set exactly at ``L=2`` by
enumerating every ``v^(d*d)`` grid: ``lv_pooled`` and ``lv_aligned`` are its
sizes, and ``pool_cost = lv_pooled / lv_aligned`` (line 165) is the factor by
which pooling inflates it.  If support were the whole story, the per-phase
machine's grammaticality advantage would be exactly ``pool_cost``, because
``ceil_pooled = R_L / lv_pooled`` is the grammaticality a sampler drawing
uniformly from its own locally-valid set would reach.

*Sampler.*  Pooling also changes the posterior *weights*, not just their
support: the pooled count for a window mixes placements from all ``s^2`` phases,
so even where the two tables have the same support the pooled machine weights
the continuations wrongly.  Nothing in ``pool_cost`` sees this.

The two are multiplicatively separable.  Writing ``eff = gram / ceil`` for how
efficiently a sampler converts its locally-valid set into grammatical output,

    gram_aligned / gram_pooled  =  pool_cost  x  (eff_aligned / eff_pooled)

so the measured ratio divided by ``pool_cost`` is the sampler term, with no
model in between.  This script computes both factors on matched cells.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CELL = ('v', 's', 'L', 'm', 'rule_seed', 'k')
DIRS = ['results/rhm_sampling/2026-11-02', 'results/rhm_sampling/2026-11-03',
        'results/rhm_sampling/2026-11-04']
STRADDLE = ['results/rhm_sampling/2026-11-03/straddle_v2.json',
            'results/rhm_sampling/2026-11-03/straddle_v3.json',
            'results/rhm_sampling/2026-11-03/straddle_levels_L2_v2.json',
            'results/rhm_sampling/2026-11-03/straddle_levels_L2_v3.json']


def load_runs(dirs):
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
                if (isinstance(r, dict) and 'success' in r and not r.get('skipped')
                        and isinstance(r.get('k'), int) and all(c in r for c in CELL)):
                    rows.append(r)
    return rows


def load_straddle(paths):
    out = {}
    for p in paths:
        f = ROOT / p
        if not f.exists():
            continue
        for r in json.load(open(f)):
            if 'pool_cost' not in r or 'lv_aligned' not in r:
                continue
            out[tuple(r[c] for c in CELL)] = r
    return out


def med(xs):
    xs = sorted(x for x in xs if x == x)
    if not xs:
        return float('nan')
    h = len(xs) // 2
    return xs[h] if len(xs) % 2 else 0.5 * (xs[h - 1] + xs[h])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pad_mode', default='outside')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    runs = [r for r in load_runs(DIRS)
            if r.get('pad_mode') == args.pad_mode and r.get('block_align') in (1, 2, 3)]
    by = defaultdict(dict)
    for r in runs:
        by[tuple(r[c] for c in CELL)].setdefault(r['block_align'], []).append(r)

    strad = load_straddle(STRADDLE)
    keys = sorted(k for k in by if 1 in by[k] and 2 in by[k])
    print('%d run rows, %d (cell,k) with both alignments, %d with exact support\n'
          % (len(runs), len(keys), sum(k in strad for k in keys)))
    if not keys:
        return

    print('=== peak over the shared P ladder, per-phase (ba=2) vs pooled (ba=1) ===')
    print('   v  s  L  m rs  k |  gram_2  gram_1  ratio |  succ_2  succ_1  ratio |'
          ' pool_cost  sampler')
    table = []
    for key in keys:
        # the same ladder for both arms, so no peak is read off a P the other
        # never visited
        Ps = set(r['P'] for r in by[key][1]) & set(r['P'] for r in by[key][2])
        if not Ps:
            continue
        pk = {ba: max((r for r in by[key][ba] if r['P'] in Ps), key=lambda r: r['success'])
              for ba in (1, 2)}
        gk = {ba: max((r for r in by[key][ba] if r['P'] in Ps),
                      key=lambda r: r['grammatical']) for ba in (1, 2)}
        g1, g2 = gk[1]['grammatical'], gk[2]['grammatical']
        s1, s2 = pk[1]['success'], pk[2]['success']
        st = strad.get(key)
        pc = st['pool_cost'] if st else float('nan')
        gr = g2 / g1 if g1 else float('nan')
        row = dict(zip(CELL, key))
        row.update(gram_pooled=g1, gram_phase=g2, gram_ratio=gr,
                   succ_pooled=s1, succ_phase=s2,
                   succ_ratio=s2 / s1 if s1 else float('nan'),
                   pool_cost=pc, sampler_term=gr / pc if pc == pc and pc else float('nan'),
                   n_P=len(Ps), P_gram_pooled=gk[1]['P'], P_gram_phase=gk[2]['P'])
        if st:
            row.update(R_L=st['R_L'], lv_pooled=st['lv_pooled'], lv_aligned=st['lv_aligned'],
                       ceil_pooled=st['ceil_pooled'], ceil_aligned=st['ceil_aligned'],
                       eff_pooled=g1 / st['ceil_pooled'] if st['ceil_pooled'] else float('nan'),
                       eff_phase=g2 / st['ceil_aligned'] if st['ceil_aligned'] else float('nan'))
        table.append(row)
        print('  %2d %2d %2d %2d %2d %2d | %7.4f %7.4f %6.2f | %7.4f %7.4f %6.2f | %9s %8s'
              % (row['v'], row['s'], row['L'], row['m'], row['rule_seed'], row['k'],
                 g2, g1, gr, s2, s1, row['succ_ratio'],
                 '%.3f' % pc if pc == pc else '-',
                 '%.2f' % row['sampler_term'] if row['sampler_term'] == row['sampler_term'] else '-'))

    have = [r for r in table if r['pool_cost'] == r['pool_cost']]
    print('\n=== the decomposition  gram_2/gram_1 = pool_cost x sampler ===')
    print('  cells with exact support: %d' % len(have))
    print('  median measured grammaticality ratio : %6.2fx' % med([r['gram_ratio'] for r in have]))
    print('  median support term (pool_cost)      : %6.2fx' % med([r['pool_cost'] for r in have]))
    print('  median sampler term (the remainder)  : %6.2fx' % med([r['sampler_term'] for r in have]))
    print('  median peak-success ratio            : %6.2fx  (L=2: %.2fx over %d, L=3: %.2fx over %d)'
          % (med([r['succ_ratio'] for r in table]),
             med([r['succ_ratio'] for r in table if r['L'] == 2]),
             sum(r['L'] == 2 for r in table),
             med([r['succ_ratio'] for r in table if r['L'] == 3]),
             sum(r['L'] == 3 for r in table)))
    print('\n  how far each sampler sits below its own support ceiling (gram / ceil):')
    print('    pooled   %.4f   (ceil %.4f, gram %.4f)'
          % (med([r['eff_pooled'] for r in have]), med([r['ceil_pooled'] for r in have]),
             med([r['gram_pooled'] for r in have])))
    print('    per-phase %.4f  (ceil %.4f, gram %.4f)'
          % (med([r['eff_phase'] for r in have]), med([r['ceil_aligned'] for r in have]),
             med([r['gram_phase'] for r in have])))

    # What does the sampler term scale with? Pooling mixes s^2 phases into one
    # table -- 4 at s=2, 9 at s=3 -- so if the cost is phase confusion it grows
    # with s, and if it is per-site weight error it does not. Phase-aware means
    # block_align = s, so this reads ba = s against ba = 1 at every cell that
    # has both, which covers s=2 and s=3 in one pass. No exact support exists at
    # s=3 (v^81 grids), so this is the measured ratio only.
    print('\n=== does the cost grow with the number of phases mixed? ===')
    print('   s  phases | cells | median gram ratio | median succ ratio')
    for s in sorted({k[1] for k in by}):
        ks = [k for k in by if k[1] == s and 1 in by[k] and s in by[k]]
        rows = []
        for key in ks:
            Ps = set(r['P'] for r in by[key][1]) & set(r['P'] for r in by[key][s])
            if not Ps:
                continue
            g = {ba: max((r for r in by[key][ba] if r['P'] in Ps),
                         key=lambda r: r['grammatical'])['grammatical'] for ba in (1, s)}
            p = {ba: max((r for r in by[key][ba] if r['P'] in Ps),
                         key=lambda r: r['success'])['success'] for ba in (1, s)}
            if g[1] and p[1]:
                rows.append((g[s] / g[1], p[s] / p[1]))
        if rows:
            print('  %2d  %6d | %5d | %17.2fx | %16.2fx'
                  % (s, s * s, len(rows), med([r[0] for r in rows]),
                     med([r[1] for r in rows])))

    if args.out:
        o = ROOT / args.out
        o.parent.mkdir(parents=True, exist_ok=True)
        json.dump(table, open(o, 'w'), indent=1)
        print('\nwrote %s (%d rows)' % (o, len(table)))


if __name__ == '__main__':
    main()
